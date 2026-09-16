#!/usr/bin/env python3
"""Locate the first vLLM #56900 divergence at opaque MoE boundaries.

The controller runs fresh-process compiled/eager cells.  Workers install
recorders *after* model construction on each instantiated MoERunner's
``_forward_impl``.  That method is entered by the existing opaque
``torch.ops.vllm.moe_forward_shared`` custom op, so the recorder does not add a
new operation to the Dynamo/Inductor graph.

This is a diagnostic, not an accuracy oracle.  The controller includes an
uninstrumented compiled cell by default and reports whether recording preserves
its generated tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any, Callable


MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
SHORT = [
    "The verifier re-runs a sampled computation and compares hashes.",
    "Floating-point addition is not associative, so",
    "def fibonacci(n):",
    "In 1854 the",
]
LONG = [
    " ".join(["The verifier records the batch shape at every step and replays it later."] * k)
    for k in (3, 6, 9, 12)
]
PROMPTS = [prompt for pair in zip(SHORT, LONG) for prompt in pair] * 2


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=repr) + "\n")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_snapshot(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "root": str(root),
        "file_count": len(files),
        "files": [str(path.relative_to(root)) for path in files],
    }


def module_provenance() -> dict[str, Any]:
    """Record concrete distributions and import entry points for each cell."""
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for distribution in ("torch", "triton", "vllm", "tokenspeed-triton"):
        try:
            package = importlib.metadata.distribution(distribution)
            record = package.locate_file(
                next((file for file in package.files or () if str(file).endswith("RECORD")), "")
            )
            details: dict[str, Any] = {
                "distribution_version": package.version,
                "distribution_location": str(package.locate_file("")),
                "record_sha256": sha256_bytes(record.read_bytes()) if record.is_file() else None,
                "direct_url": package.read_text("direct_url.json"),
            }
        except importlib.metadata.PackageNotFoundError:
            details = {"distribution_version": None}
        if distribution != "tokenspeed-triton":
            try:
                module = importlib.import_module(distribution)
                module_file = getattr(module, "__file__", None)
                details.update(
                    {
                        "module_version": getattr(module, "__version__", None),
                        "module_file": module_file,
                        "module_file_sha256": sha256_bytes(Path(module_file).read_bytes())
                        if module_file and Path(module_file).is_file()
                        else None,
                        "import_error": None,
                    }
                )
            except BaseException as error:
                details["import_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        result[distribution] = details
    result["triton_package_providers"] = importlib.metadata.packages_distributions().get(
        "triton", []
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        capture_output=True,
        check=False,
    )
    result["pip_freeze"] = {
        "returncode": freeze.returncode,
        "stdout": freeze.stdout,
        "stderr": freeze.stderr,
    }
    return result


def version_matches(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    return actual == expected if "+" in expected else actual.split("+", 1)[0] == expected


def normalise_layer_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def layer_index(name: str) -> int | None:
    parts = name.split(".")
    try:
        position = parts.index("layers")
        return int(parts[position + 1])
    except (ValueError, IndexError):
        return None


def tensor_summary(torch: Any, tensor: Any, *, row_hashes: bool) -> dict[str, Any]:
    """Copy one tensor to CPU and produce exact byte and optional row digests."""
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    result: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "numel": cpu.numel(),
        "nbytes": len(raw),
        "sha256": sha256_bytes(raw),
    }
    if cpu.numel():
        numeric = cpu.float()
        result["stats"] = {
            "finite": int(torch.isfinite(numeric).sum().item()),
            "min": float(numeric.min().item()),
            "max": float(numeric.max().item()),
            "mean": float(numeric.mean().item()),
            "l2": float(torch.linalg.vector_norm(numeric).item()),
        }
    else:
        result["stats"] = {"finite": 0, "min": None, "max": None, "mean": None, "l2": 0.0}
    if row_hashes and cpu.ndim >= 1:
        result["row_sha256"] = [
            sha256_bytes(row.contiguous().view(torch.uint8).numpy().tobytes())
            for row in cpu.reshape(cpu.shape[0], -1)
        ]
    return result


def duplicate_block_report(row_hashes: list[str], prompt_lengths: list[int]) -> dict[str, Any]:
    """Compare the duplicated prompt blocks in a packed prefill tensor."""
    if len(prompt_lengths) != 16:
        return {"valid": False, "reason": "expected 16 prompt lengths"}
    offsets = [0]
    for length in prompt_lengths:
        offsets.append(offsets[-1] + length)
    if offsets[-1] != len(row_hashes):
        return {
            "valid": False,
            "reason": "row count does not equal packed prompt-token count",
            "rows": len(row_hashes),
            "prompt_tokens": offsets[-1],
        }
    pairs = []
    for left in range(8):
        right = left + 8
        left_rows = row_hashes[offsets[left] : offsets[left + 1]]
        right_rows = row_hashes[offsets[right] : offsets[right + 1]]
        pairs.append(left_rows == right_rows)
    return {"valid": True, "pairs_equal": pairs, "pairs_equal_count": sum(pairs)}


class BoundaryRecorder:
    def __init__(
        self,
        torch: Any,
        output_dir: Path,
        prompt_lengths: list[int],
        save_tensors: bool,
    ) -> None:
        self.torch = torch
        self.output_dir = output_dir
        self.prompt_lengths = prompt_lengths
        self.save_tensors = save_tensors
        self.events: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.raw_dir = output_dir / "tensors"
        if save_tensors:
            self.raw_dir.mkdir()

    def capture(
        self,
        layer: str,
        call: int,
        stage: str,
        tensor: Any,
        *,
        row_hashes: bool = True,
    ) -> None:
        summary = tensor_summary(self.torch, tensor, row_hashes=row_hashes)
        if row_hashes and call == 0:
            summary["duplicate_blocks"] = duplicate_block_report(
                summary.get("row_sha256", []), self.prompt_lengths
            )
        event = {"layer": layer, "layer_index": layer_index(layer), "call": call, "stage": stage}
        event["tensor"] = summary
        if self.save_tensors:
            filename = f"{len(self.events):04d}_{normalise_layer_name(layer)}_call{call}_{stage}.pt"
            self.torch.save(tensor.detach().contiguous().cpu(), self.raw_dir / filename)
            event["tensor_file"] = f"tensors/{filename}"
        self.events.append(event)

    def install(self, runners: list[tuple[str, Any]]) -> None:
        for layer, runner in runners:
            original = runner._forward_impl
            original_select = runner.router.select_experts
            active: dict[str, Any] = {}

            def select_wrapper(
                router_self: Any,
                *args: Any,
                __original: Callable[..., Any] = original_select,
                __active: dict[str, Any] = active,
                **kwargs: Any,
            ) -> Any:
                del router_self
                result = __original(*args, **kwargs)
                if __active.get("enabled") and isinstance(result, tuple) and len(result) >= 2:
                    __active["topk_weights"] = result[0].detach().clone()
                    __active["topk_ids"] = result[1].detach().clone()
                return result

            runner.router.select_experts = MethodType(select_wrapper, runner.router)

            def forward_wrapper(
                runner_self: Any,
                hidden_states: Any,
                router_logits: Any,
                shared_experts_input: Any,
                input_ids: Any = None,
                *,
                __layer: str = layer,
                __original: Callable[..., Any] = original,
                __active: dict[str, Any] = active,
            ) -> Any:
                del runner_self
                call = self.counts.get(__layer, 0)
                self.counts[__layer] = call + 1
                hidden_before = hidden_states.detach().clone()
                router_before = router_logits.detach().clone()
                same_input_storage = bool(
                    shared_experts_input is not None
                    and hidden_states.data_ptr() == shared_experts_input.data_ptr()
                    and hidden_states.shape == shared_experts_input.shape
                    and hidden_states.stride() == shared_experts_input.stride()
                    and hidden_states.storage_offset()
                    == shared_experts_input.storage_offset()
                )
                shared_before = (
                    hidden_before
                    if same_input_storage
                    else shared_experts_input.detach().clone()
                    if shared_experts_input is not None
                    else None
                )
                alias = {
                    "hidden_shared_same_object": hidden_states is shared_experts_input,
                    "hidden_shared_same_data_ptr": bool(
                        shared_experts_input is not None
                        and hidden_states.data_ptr() == shared_experts_input.data_ptr()
                    ),
                    "hidden_shared_same_view": same_input_storage,
                    "hidden_data_ptr": hidden_states.data_ptr(),
                    "shared_data_ptr": shared_experts_input.data_ptr()
                    if shared_experts_input is not None
                    else None,
                }
                __active.clear()
                __active["enabled"] = True
                try:
                    result = __original(
                        hidden_states, router_logits, shared_experts_input, input_ids
                    )
                finally:
                    __active["enabled"] = False
                self.torch.cuda.synchronize()
                self.capture(__layer, call, "moe_input", hidden_before)
                if shared_before is not None and not same_input_storage:
                    self.capture(__layer, call, "shared_input", shared_before)
                self.capture(__layer, call, "router_logits", router_before)
                if "topk_weights" in __active:
                    self.capture(__layer, call, "topk_weights", __active["topk_weights"])
                    self.capture(__layer, call, "topk_ids", __active["topk_ids"])
                values = result if isinstance(result, tuple) else (result,)
                names = ("shared_output", "routed_output") if len(values) == 2 else ("routed_output",)
                for name, value in zip(names, values):
                    self.capture(__layer, call, name, value)
                self.capture(__layer, call, "moe_input_after", hidden_states)
                self.events[-1]["call_metadata"] = alias
                return result

            runner._forward_impl = MethodType(forward_wrapper, runner)

    def validate(self, selected_layers: list[str]) -> None:
        """Reject partial or ambiguous traces before comparing them."""
        keys = [(event["call"], event["layer"], event["stage"]) for event in self.events]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise RuntimeError(f"duplicate trace event keys: {duplicates}")
        expected_stages = {
            "moe_input",
            "router_logits",
            "topk_weights",
            "topk_ids",
            "shared_output",
            "routed_output",
            "moe_input_after",
        }
        calls: dict[tuple[str, int], set[str]] = {}
        for event in self.events:
            calls.setdefault((event["layer"], event["call"]), set()).add(event["stage"])
        missing_layers = sorted(set(selected_layers) - {layer for layer, _ in calls})
        incomplete = {
            f"{layer}:call{call}": sorted(expected_stages - stages)
            for (layer, call), stages in calls.items()
            if not expected_stages <= stages
        }
        if missing_layers or incomplete:
            raise RuntimeError(
                f"incomplete boundary trace: missing_layers={missing_layers}, "
                f"missing_stages={incomplete}"
            )


def discover_runners(config: Any, selected_layers: set[int] | None) -> list[tuple[str, Any]]:
    context = config.compilation_config.static_forward_context
    runners = []
    for name, value in context.items():
        if not (hasattr(value, "_forward_impl") and hasattr(value, "router")):
            continue
        index = layer_index(str(name))
        if selected_layers is None or index in selected_layers:
            runners.append((str(name), value))
    runners.sort(key=lambda item: (layer_index(item[0]) is None, layer_index(item[0]), item[0]))
    return runners


def output_summary(outputs: list[Any]) -> dict[str, Any]:
    tokens = [list(output.outputs[0].token_ids) for output in outputs]
    return {
        "token_ids": tokens,
        "duplicate_pairs": [tokens[index] == tokens[index + 8] for index in range(8)],
        "texts": [output.outputs[0].text for output in outputs],
    }


def worker(arguments: argparse.Namespace) -> None:
    run_dir = arguments.run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"refusing existing worker directory: {run_dir}")
    cache_root = run_dir / "cache"
    cache_root.mkdir(parents=True)
    environment = {
        "VLLM_PLUGINS": "",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
    }
    os.environ.pop("VLLM_DEBUG_DUMP_PATH", None)
    os.environ.update(environment)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "cache_before.json", cache_snapshot(cache_root))
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt

        write_json(run_dir / "provenance.json", module_provenance())

        if not version_matches(vllm.__version__, arguments.expected_vllm):
            raise RuntimeError(f"expected vLLM {arguments.expected_vllm}, got {vllm.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        compilation = {
            "mode": 3 if arguments.compile == "on" else 0,
            "cudagraph_mode": "NONE",
        }
        llm = LLM(
            model=MODEL,
            tensor_parallel_size=1,
            seed=0,
            max_model_len=2048,
            enable_prefix_caching=False,
            enforce_eager=False,
            compilation_config=compilation,
            revision=REVISION,
            tokenizer_revision=REVISION,
        )
        config = llm.llm_engine.vllm_config
        resolved = {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "compile_requested": arguments.compile,
            "mode": config.compilation_config.mode.value,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "use_v2_model_runner": config.use_v2_model_runner,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
        }
        write_json(run_dir / "resolved.json", resolved)
        expected_mode = 3 if arguments.compile == "on" else 0
        if (
            resolved["mode"] != expected_mode
            or resolved["cudagraph_mode"] != "NONE"
            or resolved["use_v2_model_runner"] is not True
        ):
            raise RuntimeError(f"configuration mismatch: {resolved}")

        tokenizer = llm.get_tokenizer()
        prompt_ids = [tokenizer.encode(prompt) for prompt in PROMPTS]
        runners = discover_runners(config, set(arguments.layers) if arguments.layers else None)
        recorder = None
        if arguments.record:
            if not runners:
                raise RuntimeError("no selected MoERunner instances found")
            recorder = BoundaryRecorder(
                torch,
                run_dir,
                [len(tokens) for tokens in prompt_ids],
                arguments.save_tensors,
            )
            recorder.install(runners)
        batch = [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_ids]
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=arguments.max_tokens,
            logprobs=5,
            ignore_eos=True,
        )
        outputs = llm.generate(batch, sampling, use_tqdm=False)
        write_json(
            run_dir / "result.json",
            {
                "model": MODEL,
                "revision": REVISION,
                "compile": arguments.compile,
                "record": arguments.record,
                "prompts": PROMPTS,
                "prompt_token_ids": prompt_ids,
                "output": output_summary(outputs),
                "selected_runners": [name for name, _ in runners],
            },
        )
        if recorder is not None:
            recorder.validate([name for name, _ in runners])
            write_json(
                run_dir / "trace.json",
                {
                    "schema": 1,
                    "capture_boundary": "MoERunner._forward_impl inside existing opaque moe_forward_shared op",
                    "events": recorder.events,
                },
            )
        write_json(run_dir / "cache_after.json", cache_snapshot(cache_root))
    except BaseException as error:
        write_json(
            run_dir / "failure.json",
            {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        write_json(run_dir / "cache_after.json", cache_snapshot(cache_root))
        raise


def event_key(event: dict[str, Any]) -> tuple[int, str, str]:
    return int(event["call"]), str(event["layer"]), str(event["stage"])


def compare_traces(compiled: dict[str, Any], eager: dict[str, Any]) -> dict[str, Any]:
    for name, trace in (("compiled", compiled), ("eager", eager)):
        keys = [event_key(event) for event in trace["events"]]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"{name} trace has duplicate event keys: {duplicates}")
    left = {event_key(event): event for event in compiled["events"]}
    right = {event_key(event): event for event in eager["events"]}
    # Preserve recorder order.  Alphabetical stage order would incorrectly put
    # an output before an input and could mislabel the first boundary.
    keys = list(left)
    keys.extend(key for key in right if key not in left)
    comparisons = []
    first_difference = None
    first_differing_layer_call = None
    differing_stages_at_first_layer: list[str] = []
    for key in keys:
        a, b = left.get(key), right.get(key)
        missing = "compiled" if a is None else "eager" if b is None else None
        same = bool(a and b and a["tensor"]["sha256"] == b["tensor"]["sha256"])
        row_differences = None
        if a and b:
            ar = a["tensor"].get("row_sha256")
            br = b["tensor"].get("row_sha256")
            if ar is not None and br is not None and len(ar) == len(br):
                row_differences = sum(x != y for x, y in zip(ar, br))
        item = {
            "call": key[0],
            "layer": key[1],
            "layer_index": layer_index(key[1]),
            "stage": key[2],
            "identical": same,
            "missing": missing,
            "rows_differing": row_differences,
        }
        comparisons.append(item)
        if not same and first_difference is None:
            first_difference = item
        layer_call = (key[0], key[1])
        if not same and first_differing_layer_call is None:
            first_differing_layer_call = layer_call
        if not same and layer_call == first_differing_layer_call:
            differing_stages_at_first_layer.append(key[2])
    return {
        "compiled_events": len(compiled["events"]),
        "eager_events": len(eager["events"]),
        "first_difference": first_difference,
        "first_differing_layer_call": {
            "call": first_differing_layer_call[0],
            "layer": first_differing_layer_call[1],
            "layer_index": layer_index(first_differing_layer_call[1]),
            "differing_stages": differing_stages_at_first_layer,
        }
        if first_differing_layer_call is not None
        else None,
        "comparisons": comparisons,
    }


def compare_raw_tensors(
    output_dir: Path, compiled: dict[str, Any], eager: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compute elementwise metrics for a targeted ``--save-tensors`` pair."""
    import torch

    left = {event_key(event): event for event in compiled["events"]}
    right = {event_key(event): event for event in eager["events"]}
    comparisons = []
    for key in left:
        a, b = left[key], right.get(key)
        if b is None or "tensor_file" not in a or "tensor_file" not in b:
            continue
        x = torch.load(
            output_dir / "compiled-recorded" / a["tensor_file"],
            map_location="cpu",
            weights_only=True,
        )
        y = torch.load(
            output_dir / "eager-recorded" / b["tensor_file"],
            map_location="cpu",
            weights_only=True,
        )
        item: dict[str, Any] = {
            "call": key[0],
            "layer": key[1],
            "layer_index": layer_index(key[1]),
            "stage": key[2],
            "shape_equal": x.shape == y.shape,
            "dtype_equal": x.dtype == y.dtype,
        }
        if x.shape == y.shape:
            unequal = x != y
            item["elements"] = x.numel()
            item["elements_differing"] = int(unequal.sum().item())
            if x.is_floating_point() and x.numel():
                delta = (x.float() - y.float()).abs()
                item["max_abs"] = float(delta.max().item())
                item["mean_abs"] = float(delta.mean().item())
                item["l2_delta"] = float(torch.linalg.vector_norm(delta).item())
        comparisons.append(item)
    return comparisons


def compare_cells(output_dir: Path, cells: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for cell in cells:
        path = output_dir / cell["name"] / "result.json"
        if cell["returncode"] == 0 and path.is_file():
            results[cell["name"]] = json.loads(path.read_text())
    report: dict[str, Any] = {"schema": 1, "cells": cells}
    baseline = results.get("compiled-baseline")
    compiled = results.get("compiled-recorded")
    eager = results.get("eager-recorded")
    if baseline and compiled:
        report["instrumentation_preserves_compiled_tokens"] = (
            baseline["output"]["token_ids"] == compiled["output"]["token_ids"]
        )
    if compiled and eager:
        left = compiled["output"]["token_ids"]
        right = eager["output"]["token_ids"]
        report["compiled_vs_eager"] = {
            "identical_outputs": sum(a == b for a, b in zip(left, right)),
            "duplicate_pairs_compiled": compiled["output"]["duplicate_pairs"],
            "duplicate_pairs_eager": eager["output"]["duplicate_pairs"],
        }
        ct = json.loads((output_dir / "compiled-recorded" / "trace.json").read_text())
        et = json.loads((output_dir / "eager-recorded" / "trace.json").read_text())
        report["trace_comparison"] = compare_traces(ct, et)
        if any("tensor_file" in event for event in ct["events"]):
            report["raw_tensor_comparison"] = compare_raw_tensors(output_dir, ct, et)
    return report


def stop_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_cell(command: list[str], stdout: Path, stderr: Path, timeout: int) -> tuple[int, bool]:
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, text=True, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            stop_group(process)
            return 124, True
        except BaseException:
            stop_group(process)
            raise


def handoff_manifest(output_dir: Path, cell_name: str) -> Path:
    """Describe every byte that must be durably copied for one completed cell."""
    label = cell_name.replace("-", "_")
    handoff_dir = output_dir / "handoff" / label
    if handoff_dir.exists():
        raise RuntimeError(f"refusing existing handoff directory: {handoff_dir}")
    handoff_dir.mkdir(parents=True)
    script_copy = handoff_dir / Path(__file__).name
    script_copy.write_bytes(Path(__file__).read_bytes())
    controller_snapshot = handoff_dir / "controller_manifest.json"
    controller_snapshot.write_bytes((output_dir / "manifest.json").read_bytes())
    comparison_snapshot = handoff_dir / "comparison_before_ack.json"
    comparison_snapshot.write_bytes((output_dir / "comparison.json").read_bytes())
    targets = [
        output_dir / cell_name,
        output_dir / f"{cell_name}.stdout.log",
        output_dir / f"{cell_name}.stderr.log",
        script_copy,
        controller_snapshot,
        comparison_snapshot,
    ]
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in targets:
        if not target.exists():
            continue
        candidates = (
            [target]
            if target.is_file()
            else sorted(path for path in target.rglob("*") if path.is_file())
        )
        for path in candidates:
            if path.is_symlink():
                raise RuntimeError(f"handoff evidence contains symlink: {path}")
            relative = path.relative_to(output_dir).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            files.append(
                {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    files.sort(key=lambda item: item["path"])
    payload = {"schema": 1, "label": label, "files": files}
    manifest = handoff_dir / "manifest.json"
    write_json(manifest, {**payload, "sha256": sha256_bytes(canonical_json(payload))})
    return manifest


def verify_ack(ack: Path, label: str, manifest: Path) -> None:
    expected = json.loads(manifest.read_text())["sha256"]
    value = json.loads(ack.read_text())
    required = {
        "schema": 1,
        "label": label,
        "handoff_manifest_sha256": expected,
        "verified_manifest_sha256": expected,
    }
    if value != required:
        raise RuntimeError(f"invalid durable handoff acknowledgement: {ack}")


def verify_handoff_manifest(output_dir: Path, manifest: Path) -> dict[str, Any]:
    value = json.loads(manifest.read_text())
    if set(value) != {"schema", "label", "files", "sha256"} or value["schema"] != 1:
        raise RuntimeError(f"invalid handoff manifest schema: {manifest}")
    payload = {key: value[key] for key in ("schema", "label", "files")}
    expected = sha256_bytes(canonical_json(payload))
    if value["sha256"] != expected:
        raise RuntimeError(f"invalid handoff manifest digest: {manifest}")
    seen: set[str] = set()
    root = output_dir.resolve()
    for item in value["files"]:
        relative = str(item["path"])
        if relative in seen:
            raise RuntimeError(f"duplicate path in handoff manifest: {relative}")
        seen.add(relative)
        path = (output_dir / relative).resolve()
        if path == root or root not in path.parents or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid handoff file path: {relative}")
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"handoff file does not match manifest: {relative}")
    return {"label": value["label"], "files": len(value["files"]), "sha256": expected}


def audit_handoffs(output_dir: Path, *, require_acks: bool = True) -> dict[str, Any]:
    manifests = sorted((output_dir / "handoff").glob("*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"no handoff manifests found under: {output_dir}")
    verified = []
    for manifest in manifests:
        item = verify_handoff_manifest(output_dir, manifest)
        if require_acks:
            verify_ack(output_dir / "acks" / f"{item['label']}.json", item["label"], manifest)
        verified.append(item)
    return {"schema": 1, "handoffs_verified": len(verified), "handoffs": verified}


def wait_for_ack(output_dir: Path, cell_name: str, manifest: Path, timeout: int) -> None:
    label = cell_name.replace("-", "_")
    ack = output_dir / "acks" / f"{label}.json"
    if ack.exists():
        raise RuntimeError(f"refusing stale acknowledgement: {ack}")
    expected = json.loads(manifest.read_text())["sha256"]
    print(
        f"HANDOFF_READY label={label} manifest={manifest} sha256={expected}",
        flush=True,
    )
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if ack.is_file():
            verify_ack(ack, label, manifest)
            return
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for durable handoff acknowledgement: {label}")


def controller(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    specifications = []
    if arguments.include_baseline:
        specifications.append(("compiled-baseline", "on", False))
    specifications.extend((("compiled-recorded", "on", True), ("eager-recorded", "off", True)))
    write_json(
        output_dir / "manifest.json",
        {
            "schema": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "command": sys.argv,
            "python": sys.executable,
            "model": MODEL,
            "revision": REVISION,
            "layers": arguments.layers,
            "max_tokens": arguments.max_tokens,
            "save_tensors": arguments.save_tensors,
            "cells": [name for name, _, _ in specifications],
        },
    )
    completed = []
    for name, compile_mode, record in specifications:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--run-dir",
            str(output_dir / name),
            "--compile",
            compile_mode,
            "--max-tokens",
            str(arguments.max_tokens),
            "--expected-vllm",
            arguments.expected_vllm,
        ]
        if record:
            command.append("--record")
        if arguments.save_tensors:
            command.append("--save-tensors")
        if arguments.layers:
            command.extend(["--layers", *map(str, arguments.layers)])
        returncode, timed_out = run_cell(
            command,
            output_dir / f"{name}.stdout.log",
            output_dir / f"{name}.stderr.log",
            arguments.timeout_seconds,
        )
        completed.append(
            {"name": name, "compile": compile_mode, "record": record, "returncode": returncode, "timed_out": timed_out}
        )
        write_json(output_dir / "comparison.json", compare_cells(output_dir, completed))
        if returncode:
            break
        if arguments.require_ack:
            handoff = handoff_manifest(output_dir, name)
            wait_for_ack(output_dir, name, handoff, arguments.ack_timeout_seconds)
            audit_handoffs(output_dir)
    failures = [cell for cell in completed if cell["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} localization cell(s) failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("moe-boundary-localise"))
    parser.add_argument("--compile", choices=("on", "off"), default="on", help=argparse.SUPPRESS)
    parser.add_argument("--record", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-vllm", default="0.28.0")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--require-ack",
        action="store_true",
        help="require a verified durable handoff acknowledgement after every successful cell",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="rehash all retained handoffs and validate their acknowledgements without using a GPU",
    )
    parser.add_argument("--ack-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.worker and args.run_dir is None:
        parser.error("--worker requires --run-dir")
    if args.max_tokens <= 0 or args.timeout_seconds <= 0 or args.ack_timeout_seconds <= 0:
        parser.error("token and timeout values must be positive")
    if any(layer < 0 for layer in args.layers):
        parser.error("layer indices must be non-negative")
    return args


def main() -> None:
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    elif arguments.audit_only:
        print(json.dumps(audit_handoffs(arguments.output_dir.resolve()), indent=2, sort_keys=True))
    else:
        controller(arguments)


if __name__ == "__main__":
    main()
