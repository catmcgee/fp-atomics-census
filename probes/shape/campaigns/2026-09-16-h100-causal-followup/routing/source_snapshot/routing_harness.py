#!/usr/bin/env python3
"""Run graph-safe actual MoE-routing observation in fresh single-GPU processes.

Each condition starts a new ``LLM`` engine.  The optional frozen shape hook is
only an execution witness: vLLM's native ``enable_return_routed_experts``
capturer is the sole source for selected-expert IDs and counts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from routing_telemetry import (
    RoutingTelemetryError,
    compare_hook_hashes,
    compare_matched_samples,
    digest_token_ids,
    hook_evidence,
    summarise_selected_experts,
    validate_dynamic_inputs,
    validate_graph_dispatch,
)

DEFAULT_PROMPTS = [
    "Compute exactly: 17 multiplied by 19 equals",
    "Write one vivid sentence about a glacier at sunrise.",
]
EXPECTED_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
EXPECTED_VLLM = "98dff2a81d747d1dba01a47f939f48c3526d4206"
EXPECTED_VLLM_FILES = {
    "config/compilation.py": "3a79ea08d48d44879ac8cbfee1c7d88f9bd72927d9bd12eee31743e8da8a4d7e",
    "config/scheduler.py": "f1f70b83527ee2ccc78bf186d418739d2616de1f96834ad0bbb65f42b6a41787",
    "config/vllm.py": "788430b9211a5b0ccda0be565ad2a0f27e9069370f35d2b1f6a333a6ca135186",
    "engine/arg_utils.py": "950cb3b650c081d6d36d43d73e3ecd669b44eabd4fe8395b92e399c7a89b8011",
    "entrypoints/llm.py": "52de4ac99489e004ef6c61d0bedc84aa96020dd58b8bd1ae500814b548b2b83e",
    "model_executor/layers/fused_moe/routed_experts_capturer.py": "cd006bc1a1703418cca3b589fafb2abc4268a086fb3db044bbb64348aeac8b13",
    "model_executor/layers/fused_moe/router/fused_moe_router.py": "f182f30712983f61815f8df947af597277d660cd2891066fcb13789d55f7353e",
    "sampling_params.py": "2aba9ebd1c3921d601dcf94ac6a1f726e5172fe991e313cd5a69d3eaa02013fa",
    "v1/worker/gpu_model_runner.py": "3970d7b764f80847842e10fc20f0e3f4f3da84cd97d042bb92db6e99147337e2",
    "v1/core/sched/scheduler.py": "abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983",
    "v1/engine/output_processor.py": "80a01067f4b3b351239506a3a1754dafdfc83cd162352d95545ba8ebed44c447",
    "outputs.py": "346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425",
}
EXPECTED_HOOK_SHA256 = "5bc003293fef95e525658226cfb05b2ad28199cb7010ebb09bbd9c37954e357f"
ACK_TIMEOUT_SECONDS = 180


def source_attestation(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise RoutingTelemetryError(f"source root is missing: {source_root}")
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip()
    if (head, dirty) != (EXPECTED_SOURCE, ""):
        raise RoutingTelemetryError(
            f"source gate failed: head={head}, dirty={bool(dirty)}"
        )
    hook_path = source_root / "probes/shape/shape_hook_pkg/shape_hook/__init__.py"
    hook_digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    if hook_digest != EXPECTED_HOOK_SHA256:
        raise RoutingTelemetryError(f"frozen shape-hook bytes differ: {hook_digest}")
    vllm_spec = importlib.util.find_spec("vllm")
    hook_spec = importlib.util.find_spec("shape_hook")
    if vllm_spec is None or vllm_spec.origin is None:
        raise RoutingTelemetryError("installed vllm package is unavailable")
    vllm_package = Path(vllm_spec.origin).resolve().parent
    actual = {}
    for relative, expected in EXPECTED_VLLM_FILES.items():
        path = vllm_package / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[relative] = digest
        if digest != expected:
            raise RoutingTelemetryError(f"installed vllm bytes differ for {relative}: {digest}")
    expected_hook_init = (
        source_root / "probes/shape/shape_hook_pkg/shape_hook/__init__.py"
    ).resolve()
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RoutingTelemetryError("installed vllm distribution metadata is unavailable") from exc
    installed = {
        "vllm": str(Path(vllm_spec.origin).resolve()),
        "shape_hook": (
            None
            if hook_spec is None or hook_spec.origin is None
            else str(Path(hook_spec.origin).resolve())
        ),
    }
    if vllm_version != "0.29.0":
        raise RoutingTelemetryError(f"installed vllm version is {vllm_version}, expected 0.29.0")
    if installed["shape_hook"] != str(expected_hook_init):
        raise RoutingTelemetryError(f"installed shape_hook is not the frozen source tree: {installed}")
    return {
        "source_commit": head,
        "reviewed_vllm_commit": EXPECTED_VLLM,
        "vllm_version": vllm_version,
        "vllm_files": actual,
        "shape_hook_sha256": hook_digest,
        "installed_modules": installed,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(output: Path, condition: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", f"{output.stem}_{condition}").strip("_")
    if not value:
        raise RoutingTelemetryError("unable to construct a handoff label")
    return value


def _snapshot_sources(args: argparse.Namespace) -> Path:
    root = args.output.parent.resolve() / "source_snapshot"
    if root.exists():
        raise RoutingTelemetryError(f"refusing to reuse source snapshot: {root}")
    root.mkdir(parents=True)
    sources = {
        "routing_harness.py": Path(__file__).resolve(),
        "routing_telemetry.py": Path(__file__).resolve().with_name("routing_telemetry.py"),
        "shape_hook_init.py": (
            args.source_root.resolve()
            / "probes/shape/shape_hook_pkg/shape_hook/__init__.py"
        ),
    }
    for name, source in sources.items():
        if not source.is_file():
            raise RoutingTelemetryError(f"source snapshot input is missing: {source}")
        shutil.copy2(source, root / name)
    return root


def _handoff_manifest(
    data_root: Path, label: str, targets: list[Path]
) -> Path:
    data_root = data_root.resolve()
    ack = data_root / "acks" / f"{label}.json"
    manifest = data_root / "handoff" / label / "manifest.json"
    if ack.exists():
        raise RoutingTelemetryError(f"stale ACK exists: {ack}")
    if manifest.parent.exists():
        raise RoutingTelemetryError(f"refusing to reuse handoff path: {manifest.parent}")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in targets:
        target = target.resolve()
        candidates = [target] if target.is_file() else sorted(
            path for path in target.rglob("*") if path.is_file()
        )
        if not candidates:
            raise RoutingTelemetryError(f"handoff target has no files: {target}")
        for path in candidates:
            if path.is_symlink():
                raise RoutingTelemetryError(f"handoff symlink is forbidden: {path}")
            try:
                relative = path.relative_to(data_root).as_posix()
            except ValueError as exc:
                raise RoutingTelemetryError(
                    f"handoff file is outside output data root: {path}"
                ) from exc
            if relative in seen:
                continue
            seen.add(relative)
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    files.sort(key=lambda row: row["path"])
    payload = {"schema": 1, "label": label, "files": files}
    write_json(
        manifest,
        {**payload, "sha256": hashlib.sha256(canonical_json(payload)).hexdigest()},
    )
    return manifest


def _wait_ack(data_root: Path, label: str, manifest: Path) -> None:
    ack = data_root / "acks" / f"{label}.json"
    expected = json.loads(manifest.read_text())["sha256"]
    print(
        f"HANDOFF_READY label={label} manifest={manifest} sha256={expected}",
        flush=True,
    )
    deadline = time.monotonic() + ACK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if ack.is_file():
            value = json.loads(ack.read_text())
            required = {
                "schema": 1,
                "label": label,
                "handoff_manifest_sha256": expected,
                "verified_manifest_sha256": expected,
            }
            if value != required:
                raise RoutingTelemetryError(f"invalid ACK: {ack}")
            print(f"HANDOFF_ACKED label={label}", flush=True)
            return
        time.sleep(1)
    raise RoutingTelemetryError(f"ACK timeout after {ACK_TIMEOUT_SECONDS}s: {label}")


def _request_sample(
    output: Any, index: int, num_experts: int, num_layers: int, top_k: int
) -> dict[str, Any]:
    if len(output.outputs) != 1:
        raise RoutingTelemetryError(f"prompt {index} returned {len(output.outputs)} choices, expected one")
    choice = output.outputs[0]
    prompt_ids = list(output.prompt_token_ids)
    generated_ids = list(choice.token_ids)
    routing_data = choice.routed_experts
    if routing_data is None:
        raise RoutingTelemetryError("vLLM returned no routed experts; telemetry was not active")
    # The final sampled token has not been forwarded, so it has no routing row.
    # Keep only real rows: static capture capacity must not reach a count/hash.
    expected_forward_tokens = max(1, len(prompt_ids) + len(generated_ids) - 1)
    returned_rows = int(routing_data.shape[0])
    if returned_rows < expected_forward_tokens:
        raise RoutingTelemetryError(
            "routing tensor has fewer rows than the known forwarded sequence: "
            f"{returned_rows} < {expected_forward_tokens}"
        )
    valid_tokens = expected_forward_tokens
    routing = summarise_selected_experts(
        routing_data, num_experts=num_experts, valid_tokens=valid_tokens
    )
    if routing["num_layers"] != num_layers or routing["experts_per_token"] != top_k:
        raise RoutingTelemetryError(
            "native routing shape does not match the pinned model: "
            f"layers={routing['num_layers']}, top_k={routing['experts_per_token']}"
        )
    if any(value < 2 for value in routing["per_layer_distinct_experts"]):
        raise RoutingTelemetryError(
            "at least one layer contains only one expert ID across all real tokens; "
            "cannot exclude an unwritten/static capture buffer"
        )
    return {
        "prompt_index": index,
        "prompt_token_count": len(prompt_ids),
        "generated_token_count": len(generated_ids),
        "prompt_token_ids_sha256": digest_token_ids(prompt_ids),
        "output_token_ids": generated_ids,
        "output_token_ids_sha256": digest_token_ids(generated_ids),
        "routing": routing,
    }


def _read_hook_records(path: Path) -> list[dict[str, Any]]:
    rank0 = path / "rank0.jsonl"
    if not rank0.is_file():
        raise RoutingTelemetryError(f"shape hook did not create {rank0}")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(rank0.read_text().splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RoutingTelemetryError(f"malformed shape-hook JSON at {rank0}:{line_no}") from exc
        if not isinstance(item, dict):
            raise RoutingTelemetryError(f"non-object shape-hook JSON at {rank0}:{line_no}")
        records.append(item)
    return records


def _prepare_hook_dir(root: Path | None, condition: str) -> Path | None:
    if root is None:
        return None
    path = root / condition
    if path.exists():
        raise RoutingTelemetryError(f"refusing to reuse existing shape-hook directory: {path}")
    path.mkdir(parents=True)
    # The plugin is loaded when vLLM is first imported.  Its output routine reads
    # this variable per forward, so separate engine instances can have separate
    # immutable condition directories.
    os.environ["SHAPE_HOOK_OUT"] = str(path)
    return path


def _shutdown_llm(llm: Any) -> str | None:
    """Shut down supported vLLM engines without assuming a public LLM API.

    The reviewed vLLM 0.29 ``LLM`` wrapper has no ``shutdown`` method.  Its
    ``LLMEngine.engine_core`` is an ``EngineCoreClient`` and does expose the
    supported shutdown operation.  Prefer a future public wrapper method when
    available, then use the reviewed 0.29 path.  A fresh worker process remains
    the final cleanup boundary when neither API exists.
    """
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return "LLM.shutdown"
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return "LLM.llm_engine.engine_core.shutdown"
    return None


def run_condition(
    *,
    condition: str,
    model: str,
    revision: str,
    prompts: list[str],
    max_tokens: int,
    cudagraph_mode: str,
    enable_telemetry: bool,
    num_experts: int,
    num_layers: int,
    top_k: int,
    shape_hook_root: Path | None,
    condition_output: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    hook_dir = _prepare_hook_dir(shape_hook_root, condition)
    # Delayed import keeps --help and CPU schema tests free of CUDA effects.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        revision=revision,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=2048,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=False,
        compilation_config={"mode": 0, "cudagraph_mode": cudagraph_mode},
        enable_return_routed_experts=enable_telemetry,
    )
    resolved_async_scheduling = (
        llm.llm_engine.vllm_config.scheduler_config.async_scheduling
    )
    if resolved_async_scheduling is not False:
        _shutdown_llm(llm)
        raise RoutingTelemetryError(
            "synchronous scheduler request did not resolve false: "
            f"{resolved_async_scheduling!r}"
        )
    result: dict[str, Any] = {
        "schema": 1,
        "condition": condition,
        "status": "engine_created",
        "source_attestation": source,
        "engine_controls": {
            "seed": 0,
            "max_model_len": 2048,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "enforce_eager": False,
            "compilation_config": {"mode": 0, "cudagraph_mode": cudagraph_mode},
            "enable_return_routed_experts": enable_telemetry,
        },
        "samples": [],
        "hook_evidence": None,
    }
    write_json(condition_output, result)
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=max_tokens, seed=0, ignore_eos=True),
            use_tqdm=False,
        )
        if not enable_telemetry:
            samples = []
            for index, output in enumerate(outputs):
                sample = {
                    "prompt_index": index,
                    "prompt_token_count": len(output.prompt_token_ids),
                    "generated_token_count": len(output.outputs[0].token_ids),
                    "prompt_token_ids_sha256": digest_token_ids(list(output.prompt_token_ids)),
                    "output_token_ids": list(output.outputs[0].token_ids),
                    "output_token_ids_sha256": digest_token_ids(list(output.outputs[0].token_ids)),
                    "routing": None,
                }
                samples.append(sample)
                result.update(status="samples_collecting", samples=samples)
                write_json(condition_output, result)
        else:
            samples = []
            for index, output in enumerate(outputs):
                samples.append(
                    _request_sample(output, index, num_experts, num_layers, top_k)
                )
                result.update(status="samples_collecting", samples=samples)
                write_json(condition_output, result)
        # Single-process engine mode means this also flushes an in-process hook.
        # Worker-side writes have already completed before generate returns.
        if hook_dir is not None:
            try:
                import shape_hook
                shape_hook.flush()
            except Exception as exc:  # do not accept a missing witness silently
                raise RoutingTelemetryError("unable to flush frozen shape hook") from exc
            evidence = hook_evidence(_read_hook_records(hook_dir), require_hashes=True)
        else:
            evidence = None
        result.update(status="complete", samples=samples, hook_evidence=evidence)
        write_json(condition_output, result)
        return result
    finally:
        _shutdown_llm(llm)


def _worker(args: argparse.Namespace, prompts: list[str]) -> int:
    try:
        source = source_attestation(args.source_root)
        settings = {
            "graph-control": ("FULL", False),
            "graph-telemetry": ("FULL", True),
            "eager-telemetry": ("NONE", True),
        }
        cudagraph_mode, enable_telemetry = settings[args.worker_condition]
        run_condition(
            condition=args.worker_condition,
            model=args.model,
            revision=args.revision,
            prompts=prompts,
            max_tokens=args.max_tokens,
            cudagraph_mode=cudagraph_mode,
            enable_telemetry=enable_telemetry,
            num_experts=args.num_experts,
            num_layers=args.num_layers,
            top_k=args.top_k,
            shape_hook_root=args.shape_hook_root,
            condition_output=args.output,
            source=source,
        )
        return 0
    except Exception as exc:  # retain prior condition/sample JSON plus the failure
        failure = args.output.with_name(args.output.name + ".failure.json")
        write_json(
            failure,
            {
                "schema": 1,
                "condition": args.worker_condition,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        print(f"condition failed: {exc}", file=sys.stderr)
        return 1


def _condition_command(
    args: argparse.Namespace, condition: str, output: Path, worker_script: Path
) -> list[str]:
    command = [
        sys.executable,
        str(worker_script),
        "--worker-condition",
        condition,
        "--source-root",
        str(args.source_root),
        "--model",
        args.model,
        "--revision",
        args.revision,
        "--num-experts",
        str(args.num_experts),
        "--num-layers",
        str(args.num_layers),
        "--top-k",
        str(args.top_k),
        "--max-tokens",
        str(args.max_tokens),
        "--shape-hook-root",
        str(args.shape_hook_root),
        "--output",
        str(output),
    ]
    if args.prompts_json:
        command.extend(["--prompts-json", str(args.prompts_json)])
    return command


def _run_condition_process(command: list[str], log: Path, timeout: int) -> int:
    with log.open("w") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
            # A condition is isolated only if all descendants leave its process
            # group.  The leader can exit while an engine child is still alive.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return returncode
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    return returncode
                time.sleep(0.05)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            handle.write(f"\nHARNESS_TIMEOUT seconds={timeout}\n")
            return 124


def _aggregate_report(args: argparse.Namespace, conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {
        "graph-control": {"cudagraph_mode": "FULL", "telemetry": False},
        "graph-telemetry": {"cudagraph_mode": "FULL", "telemetry": True},
        "eager-telemetry": {"cudagraph_mode": "NONE", "telemetry": True},
    }
    for name, settings in expected.items():
        condition = conditions.get(name)
        if not isinstance(condition, dict) or condition.get("status") != "complete":
            raise RoutingTelemetryError(f"condition {name} is absent or incomplete")
        controls = condition.get("engine_controls")
        if not isinstance(controls, dict) or controls.get("compilation_config") != {
            "mode": 0,
            "cudagraph_mode": settings["cudagraph_mode"],
        } or controls.get("enable_return_routed_experts") is not settings["telemetry"]:
            raise RoutingTelemetryError(f"condition {name} does not retain the requested controls")
    attestations = [conditions[name].get("source_attestation") for name in expected]
    if not isinstance(attestations[0], dict) or any(item != attestations[0] for item in attestations[1:]):
        raise RoutingTelemetryError("condition source attestations differ")
    graph_control = conditions["graph-control"]
    graph_telemetry = conditions["graph-telemetry"]
    eager_telemetry = conditions["eager-telemetry"]
    graph_hook_dispatch = validate_graph_dispatch(
        graph_telemetry["hook_evidence"], expect_graph=True, expect_compile="0"
    )
    report = {
        "schema": 1,
        "source": "vllm 0.29 enable_return_routed_experts",
        "model": args.model,
        "revision": args.revision,
        "expected_routing_shape": {
            "num_experts": args.num_experts,
            "num_layers": args.num_layers,
            "experts_per_token": args.top_k,
        },
        "source_attestation": attestations[0],
        "requested_graph_mode": True,
        "conditions": {
            "graph_control": graph_control,
            "graph_telemetry": graph_telemetry,
            "eager_telemetry": eager_telemetry,
        },
        "instrumentation_preserves_graph_tokens": compare_matched_samples(
            graph_control["samples"], graph_telemetry["samples"], compare_routing=False
        ),
        "instrumentation_preserves_graph_hidden_logits": compare_hook_hashes(
            graph_control["hook_evidence"], graph_telemetry["hook_evidence"]
        ),
        "graph_telemetry_dispatch": graph_hook_dispatch,
        "eager_telemetry_dispatch": validate_graph_dispatch(
            eager_telemetry["hook_evidence"], expect_graph=False, expect_compile="0"
        ),
        "graph_matches_eager": compare_matched_samples(
            graph_telemetry["samples"], eager_telemetry["samples"], compare_routing=True
        ),
        "dynamic_graph_inputs": validate_dynamic_inputs(graph_telemetry["samples"]),
    }
    report["valid"] = (
        report["instrumentation_preserves_graph_tokens"]["all_equal"]
        and report["instrumentation_preserves_graph_hidden_logits"]["all_equal"]
        and report["graph_telemetry_dispatch"]["valid"]
        and report["eager_telemetry_dispatch"]["valid"]
        and report["graph_matches_eager"]["all_equal"]
        and report["dynamic_graph_inputs"]["valid"]
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    ap.add_argument("--revision", default="ec052fda178e241c7c443468d2fa1db6618996be")
    ap.add_argument("--num-experts", type=int, default=60)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--prompts-json", type=Path)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--shape-hook-root", type=Path, required=True,
                    help="new parent directory for graph-control/graph-telemetry/eager-telemetry hook logs")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--require-ack",
        action="store_true",
        help="gate each condition on an exact durable-storage ACK (180 second timeout)",
    )
    ap.add_argument("--condition-timeout", type=int, default=1200)
    ap.add_argument(
        "--worker-condition",
        choices=("graph-control", "graph-telemetry", "eager-telemetry"),
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()
    prompts = json.loads(args.prompts_json.read_text()) if args.prompts_json else DEFAULT_PROMPTS
    if not isinstance(prompts, list) or len(prompts) < 2 or any(not isinstance(prompt, str) for prompt in prompts):
        raise SystemExit("prompts must be a JSON list of at least two strings")
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be at least 2 to exercise graph decode replay")
    if args.num_experts <= 0 or args.num_layers <= 0 or args.top_k <= 0:
        raise SystemExit("--num-experts, --num-layers and --top-k must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing --output: {args.output}")
    if args.worker_condition:
        return _worker(args, prompts)
    if args.shape_hook_root.exists():
        raise SystemExit(f"refusing to reuse existing --shape-hook-root: {args.shape_hook_root}")
    condition_root = args.output.with_name(args.output.stem + ".conditions")
    if condition_root.exists():
        raise SystemExit(f"refusing to reuse condition output root: {condition_root}")
    condition_root.mkdir(parents=True)
    data_root = args.output.parent.resolve()
    worker_script = Path(__file__).resolve()
    source_snapshot = None
    if args.require_ack:
        if not args.output.is_absolute() or not args.shape_hook_root.is_absolute():
            raise SystemExit("--require-ack requires absolute --output and --shape-hook-root")
        try:
            args.shape_hook_root.resolve().relative_to(data_root)
        except ValueError as exc:
            raise SystemExit("--shape-hook-root must be inside the --output parent") from exc
        for reserved in (data_root / "handoff", data_root / "COMPLETE.json"):
            if reserved.exists():
                raise SystemExit(f"refusing to reuse ACK protocol path: {reserved}")
        (data_root / "acks").mkdir(exist_ok=True)
        source_snapshot = _snapshot_sources(args)
        worker_script = source_snapshot / "routing_harness.py"
    conditions = {}
    acked_labels: list[str] = []
    for condition in ("graph-control", "graph-telemetry", "eager-telemetry"):
        output = condition_root / f"{condition}.json"
        log = condition_root / f"{condition}.log"
        returncode = _run_condition_process(
            _condition_command(args, condition, output, worker_script),
            log,
            args.condition_timeout,
        )
        if returncode:
            failure = args.output.with_name(args.output.name + ".failure.json")
            write_json(
                failure,
                {
                    "schema": 1,
                    "failed_condition": condition,
                    "returncode": returncode,
                    "condition_output": str(output),
                    "condition_log": str(log),
                    "completed_conditions": sorted(conditions),
                },
            )
            return returncode
        conditions[condition] = json.loads(output.read_text())
        if conditions[condition].get("status") != "complete":
            failure = args.output.with_name(args.output.name + ".failure.json")
            write_json(
                failure,
                {
                    "schema": 1,
                    "failed_condition": condition,
                    "error": "condition process returned success without complete evidence",
                    "completed_conditions": sorted(conditions),
                    "acked_labels": acked_labels,
                },
            )
            return 1
        if args.require_ack:
            assert source_snapshot is not None
            try:
                label = _label(args.output, condition)
                manifest = _handoff_manifest(
                    data_root,
                    label,
                    [output, log, args.shape_hook_root / condition, source_snapshot],
                )
                _wait_ack(data_root, label, manifest)
                acked_labels.append(label)
            except Exception as exc:
                failure = args.output.with_name(args.output.name + ".failure.json")
                write_json(
                    failure,
                    {
                        "schema": 1,
                        "failed_condition": condition,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "completed_conditions": sorted(conditions),
                        "acked_labels": acked_labels,
                    },
                )
                print(f"handoff failed: {exc}", file=sys.stderr)
                return 1
    try:
        report = _aggregate_report(args, conditions)
    except Exception as exc:
        failure = args.output.with_name(args.output.name + ".failure.json")
        write_json(
            failure,
            {
                "schema": 1,
                "failed_condition": "aggregate",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "completed_conditions": sorted(conditions),
            },
        )
        print(f"aggregate failed: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, report)
    if args.require_ack:
        write_json(
            data_root / "COMPLETE.json",
            {
                "schema": 1,
                "status": "valid" if report["valid"] else "invalid",
                "valid": report["valid"],
                "output": args.output.resolve().relative_to(data_root).as_posix(),
                "acked_labels": acked_labels,
            },
        )
    print(json.dumps({"valid": report["valid"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
