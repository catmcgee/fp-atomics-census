#!/usr/bin/env python3
"""Localize #56900 at the layer-0 opaque attention boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moe_boundary_localise as base  # noqa: E402


BASELINE_COMPILED_TOKENS = [
    [279], [279], [99339], [576], [5205], [220], [13], [315],
    [151643], [99483], [151643], [279], [308], [1075], [1270], [11],
]


def nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,vbios_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "at": datetime.now(UTC).isoformat(),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as error:
        return {"at": datetime.now(UTC).isoformat(), "command": command, "error": str(error)}


def discover_layer0_attention(config: Any) -> list[tuple[str, Any]]:
    selected = []
    for name, value in config.compilation_config.static_forward_context.items():
        text = str(name)
        implementation = getattr(value, "impl", None)
        if (
            base.layer_index(text) == 0
            and "self_attn" in text
            and implementation is not None
            and callable(getattr(implementation, "forward", None))
        ):
            selected.append((text, value))
    selected.sort(key=lambda item: item[0])
    return selected


class AttentionRecorder:
    def __init__(self, torch: Any, prompt_lengths: list[int]) -> None:
        self.torch = torch
        self.prompt_lengths = prompt_lengths
        self.events: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}

    def capture(self, layer: str, call: int, stage: str, tensor: Any) -> None:
        summary = base.tensor_summary(self.torch, tensor, row_hashes=True)
        if call == 0:
            summary["duplicate_blocks"] = base.duplicate_block_report(
                summary.get("row_sha256", []), self.prompt_lengths
            )
        self.events.append(
            {
                "layer": layer,
                "layer_index": base.layer_index(layer),
                "call": call,
                "stage": stage,
                "tensor": summary,
            }
        )

    def install(self, layers: list[tuple[str, Any]]) -> None:
        for layer_name, attention in layers:
            implementation = attention.impl
            original = implementation.forward

            def forward_wrapper(
                implementation_self: Any,
                *args: Any,
                __layer: str = layer_name,
                __original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                del implementation_self
                if len(args) < 4:
                    raise RuntimeError(
                        f"unexpected attention implementation signature for {__layer}: {len(args)} args"
                    )
                # The bound implementation receives (Attention layer, q, k, v,
                # kv_cache, metadata, ...). This method is called from inside
                # unified_attention_with_output, the existing opaque custom op.
                query, key, value = args[1], args[2], args[3]
                output = kwargs.get("output")
                if output is None:
                    raise RuntimeError(f"attention output buffer was not provided for {__layer}")
                call = self.counts.get(__layer, 0)
                self.counts[__layer] = call + 1
                query_before = query.detach().clone()
                key_before = key.detach().clone()
                value_before = value.detach().clone()
                result = __original(*args, **kwargs)
                self.torch.cuda.synchronize()
                self.capture(__layer, call, "query", query_before)
                self.capture(__layer, call, "key", key_before)
                self.capture(__layer, call, "value", value_before)
                self.capture(__layer, call, "attention_output", output)
                return result

            implementation.forward = MethodType(forward_wrapper, implementation)

    def validate(self, layer_names: list[str]) -> None:
        keys = [(event["call"], event["layer"], event["stage"]) for event in self.events]
        if len(keys) != len(set(keys)):
            raise RuntimeError("duplicate attention trace event keys")
        expected = {"query", "key", "value", "attention_output"}
        calls: dict[tuple[str, int], set[str]] = {}
        for event in self.events:
            calls.setdefault((event["layer"], event["call"]), set()).add(event["stage"])
        missing_layers = sorted(set(layer_names) - {layer for layer, _ in calls})
        incomplete = {
            f"{layer}:call{call}": sorted(expected - stages)
            for (layer, call), stages in calls.items()
            if not expected <= stages
        }
        if missing_layers or incomplete:
            raise RuntimeError(
                f"incomplete attention trace: missing_layers={missing_layers}, "
                f"missing_stages={incomplete}"
            )


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
    base.write_json(run_dir / "environment.json", environment)
    base.write_json(run_dir / "cache_before.json", base.cache_snapshot(cache_root))
    base.write_json(run_dir / "nvidia_smi_before.json", nvidia_smi())
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt

        provenance = base.module_provenance()
        provenance["nvidia_smi_before_model"] = nvidia_smi()
        base.write_json(run_dir / "provenance.json", provenance)
        if not base.version_matches(vllm.__version__, arguments.expected_vllm):
            raise RuntimeError(f"expected vLLM {arguments.expected_vllm}, got {vllm.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        mode = 3 if arguments.compile == "on" else 0
        llm = LLM(
            model=base.MODEL,
            tensor_parallel_size=1,
            seed=0,
            max_model_len=2048,
            enable_prefix_caching=False,
            enforce_eager=False,
            compilation_config={"mode": mode, "cudagraph_mode": "NONE"},
            revision=base.REVISION,
            tokenizer_revision=base.REVISION,
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
        }
        base.write_json(run_dir / "resolved.json", resolved)
        if (
            resolved["mode"] != mode
            or resolved["cudagraph_mode"] != "NONE"
            or resolved["use_v2_model_runner"] is not True
        ):
            raise RuntimeError(f"configuration mismatch: {resolved}")
        tokenizer = llm.get_tokenizer()
        prompt_ids = [tokenizer.encode(prompt) for prompt in base.PROMPTS]
        selected = discover_layer0_attention(config)
        if len(selected) != 1:
            raise RuntimeError(
                f"expected one layer-0 attention implementation, found {[name for name, _ in selected]}"
            )
        recorder = AttentionRecorder(torch, [len(tokens) for tokens in prompt_ids])
        recorder.install(selected)
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=1, logprobs=5, ignore_eos=True),
            use_tqdm=False,
        )
        recorder.validate([name for name, _ in selected])
        base.write_json(
            run_dir / "result.json",
            {
                "model": base.MODEL,
                "revision": base.REVISION,
                "compile": arguments.compile,
                "prompts": base.PROMPTS,
                "prompt_token_ids": prompt_ids,
                "selected_attention": [name for name, _ in selected],
                "output": base.output_summary(outputs),
            },
        )
        base.write_json(
            run_dir / "trace.json",
            {
                "schema": 1,
                "capture_boundary": (
                    "attention implementation forward inside existing opaque "
                    "unified_attention_with_output op"
                ),
                "events": recorder.events,
            },
        )
        base.write_json(run_dir / "nvidia_smi_after.json", nvidia_smi())
        base.write_json(run_dir / "cache_after.json", base.cache_snapshot(cache_root))
    except BaseException as error:
        base.write_json(
            run_dir / "failure.json",
            {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        base.write_json(run_dir / "nvidia_smi_after.json", nvidia_smi())
        base.write_json(run_dir / "cache_after.json", base.cache_snapshot(cache_root))
        raise


def compare_cells(output_dir: Path, cells: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"schema": 1, "cells": cells}
    results = {}
    for cell in cells:
        path = output_dir / cell["name"] / "result.json"
        if cell["returncode"] == 0 and path.is_file():
            results[cell["name"]] = json.loads(path.read_text())
    compiled = results.get("compiled-attention")
    eager = results.get("eager-attention")
    if compiled:
        report["instrumentation_preserves_compiled_tokens"] = (
            compiled["output"]["token_ids"] == BASELINE_COMPILED_TOKENS
        )
    if compiled and eager:
        left, right = compiled["output"]["token_ids"], eager["output"]["token_ids"]
        report["compiled_vs_eager"] = {
            "identical_outputs": sum(a == b for a, b in zip(left, right)),
            "duplicate_pairs_compiled": compiled["output"]["duplicate_pairs"],
            "duplicate_pairs_eager": eager["output"]["duplicate_pairs"],
        }
        compiled_trace = json.loads((output_dir / "compiled-attention" / "trace.json").read_text())
        eager_trace = json.loads((output_dir / "eager-attention" / "trace.json").read_text())
        report["trace_comparison"] = base.compare_traces(compiled_trace, eager_trace)
    return report


def controller(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    base.write_json(
        output_dir / "manifest.json",
        {
            "schema": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "command": sys.argv,
            "python": sys.executable,
            "model": base.MODEL,
            "revision": base.REVISION,
            "max_tokens": 1,
            "cells": ["compiled-attention", "eager-attention"],
            "baseline_compiled_tokens": BASELINE_COMPILED_TOKENS,
        },
    )
    completed = []
    for name, compile_mode in (("compiled-attention", "on"), ("eager-attention", "off")):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--run-dir",
            str(output_dir / name),
            "--compile",
            compile_mode,
            "--expected-vllm",
            arguments.expected_vllm,
        ]
        returncode, timed_out = base.run_cell(
            command,
            output_dir / f"{name}.stdout.log",
            output_dir / f"{name}.stderr.log",
            arguments.timeout_seconds,
        )
        completed.append(
            {
                "name": name,
                "compile": compile_mode,
                "returncode": returncode,
                "timed_out": timed_out,
            }
        )
        base.write_json(output_dir / "comparison.json", compare_cells(output_dir, completed))
        if returncode:
            break
        (output_dir / name / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        if arguments.require_ack:
            manifest = base.handoff_manifest(output_dir, name)
            base.wait_for_ack(output_dir, name, manifest, arguments.ack_timeout_seconds)
            base.audit_handoffs(output_dir)
    if any(cell["returncode"] for cell in completed):
        raise SystemExit("attention boundary cell failed; no later GPU process was started")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compile", choices=("on", "off"), default="on", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("attention-boundary-localise"))
    parser.add_argument("--expected-vllm", default="0.28.0")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--require-ack", action="store_true")
    parser.add_argument("--ack-timeout-seconds", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.worker and arguments.run_dir is None:
        parser.error("--worker requires --run-dir")
    if arguments.timeout_seconds <= 0 or arguments.ack_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    worker(arguments) if arguments.worker else controller(arguments)


if __name__ == "__main__":
    main()
