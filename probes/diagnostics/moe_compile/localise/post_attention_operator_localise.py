#!/usr/bin/env python3
"""Localize #56900 within the layer-0 O-projection/add/RMSNorm block.

The compiled cell wraps the already-loaded Inductor partition (after model
construction) and records the first dynamic-shape O projection and following
fused add/RMSNorm call.  The eager cell records the inputs and outputs of the
layer-0 post-attention RMSNorm.  Both use the same prompts and one-token decode
as the preserved compiled baseline.
"""

from __future__ import annotations

import argparse
import hashlib
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
from attention_boundary_localise import BASELINE_COMPILED_TOKENS, nvidia_smi  # noqa: E402


HIDDEN_SIZE = 2048
KERNEL_NAME = "triton_red_fused_fused_add_rms_norm_0"


def shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in getattr(value, "shape", ()))


def is_o_projection_call(left: Any, right: Any, out: Any) -> bool:
    left_shape, right_shape, out_shape = shape(left), shape(right), shape(out)
    return (
        len(left_shape) == 2
        and left_shape[0] > 0
        and left_shape[1] == HIDDEN_SIZE
        and right_shape == (HIDDEN_SIZE, HIDDEN_SIZE)
        and out_shape == left_shape
    )


def candidate_generated_modules(modules: list[Any], cache_root: Path) -> list[Any]:
    selected, seen = [], set()
    for module in modules:
        path_text = getattr(module, "__file__", None)
        if not path_text or id(module) in seen:
            continue
        path = Path(path_text).resolve()
        try:
            path.relative_to(cache_root.resolve())
        except ValueError:
            continue
        if not hasattr(module, "Runner") or not hasattr(module, KERNEL_NAME):
            continue
        if not hasattr(module, "extern_kernels"):
            continue
        selected.append(module)
        seen.add(id(module))
    return sorted(selected, key=lambda item: str(getattr(item, "__file__", "")))


class PostAttentionRecorder:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.mm_calls = 0
        self.norm_calls = 0
        self.tensors: dict[str, Any] = {}
        self.module_path: str | None = None

    def record_mm(self, left: Any, right: Any, out: Any) -> None:
        if self.mm_calls:
            return
        self.mm_calls += 1
        self.tensors["attention_output"] = left.detach().clone()
        self.tensors["o_projection_matrix"] = right.detach().clone()
        self.tensors["o_projection_output"] = out.detach().clone()

    def record_norm_before(self, projection: Any, residual: Any, weight: Any) -> None:
        if self.norm_calls:
            return
        self.tensors["norm_projection_input"] = projection.detach().clone()
        self.tensors["residual"] = residual.detach().clone()
        self.tensors["norm_weight"] = weight.detach().clone()

    def record_norm_after(self, output: Any) -> None:
        if self.norm_calls:
            return
        self.torch.cuda.synchronize()
        self.tensors["norm_output"] = output.detach().clone()
        self.norm_calls += 1

    def validate(self) -> None:
        expected = {
            "attention_output",
            "o_projection_matrix",
            "o_projection_output",
            "norm_projection_input",
            "residual",
            "norm_weight",
            "norm_output",
        }
        if self.mm_calls != 1 or self.norm_calls != 1 or set(self.tensors) != expected:
            raise RuntimeError(
                f"incomplete post-attention capture: mm={self.mm_calls}, "
                f"norm={self.norm_calls}, tensors={sorted(self.tensors)}"
            )
        if not self.torch.equal(
            self.tensors["o_projection_output"], self.tensors["norm_projection_input"]
        ):
            raise RuntimeError("captured O-projection output is not the fused norm input")


class ExternKernelsProxy:
    def __init__(self, original: Any, recorder: PostAttentionRecorder) -> None:
        self.original = original
        self.recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def mm(self, left: Any, right: Any, *, out: Any) -> Any:
        result = self.original.mm(left, right, out=out)
        if is_o_projection_call(left, right, out) and not self.recorder.mm_calls:
            self.recorder.torch.cuda.synchronize()
            self.recorder.record_mm(left, right, out)
        return result


class KernelProxy:
    def __init__(self, original: Any, recorder: PostAttentionRecorder, module_path: str) -> None:
        self.original = original
        self.recorder = recorder
        self.module_path = module_path

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) < 6:
            raise RuntimeError(f"unexpected {KERNEL_NAME} signature: {len(args)} args")
        projection, residual, weight, output, rows, width = args[:6]
        matches = (
            shape(projection) == shape(residual) == shape(output)
            and shape(projection) == (int(rows), HIDDEN_SIZE)
            and shape(weight) == (HIDDEN_SIZE,)
            and int(width) == HIDDEN_SIZE
        )
        should_record = matches and not self.recorder.norm_calls
        if should_record:
            if not self.recorder.mm_calls:
                raise RuntimeError("fused norm appeared before matching O projection")
            self.recorder.record_norm_before(projection, residual, weight)
            self.recorder.module_path = self.module_path
        result = self.original.run(*args, **kwargs)
        if should_record:
            self.recorder.record_norm_after(output)
        return result


def install_generated_recorders(torch: Any, cache_root: Path) -> tuple[PostAttentionRecorder, list[str]]:
    from torch._inductor.codecache import PyCodeCache

    modules = candidate_generated_modules(list(PyCodeCache.modules), cache_root)
    if not modules:
        raise RuntimeError(f"no loaded generated partition with {KERNEL_NAME}")
    recorder = PostAttentionRecorder(torch)
    paths = []
    for module in modules:
        module_path = str(Path(module.__file__).resolve())
        paths.append(module_path)
        module.extern_kernels = ExternKernelsProxy(module.extern_kernels, recorder)
        original_kernel = getattr(module, KERNEL_NAME)
        setattr(module, KERNEL_NAME, KernelProxy(original_kernel, recorder, module_path))
    return recorder, paths


def get_model(llm: Any) -> Any:
    return (
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
    )


def find_layer0_post_attention_norm(model: Any) -> tuple[str, Any]:
    found = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("model.layers.0.post_attention_layernorm")
        or name.endswith("layers.0.post_attention_layernorm")
    ]
    unique = {id(module): (name, module) for name, module in found}
    if len(unique) != 1:
        raise RuntimeError(f"expected one layer-0 post-attention norm, found {[x[0] for x in found]}")
    return next(iter(unique.values()))


class EagerNormRecorder:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.calls = 0
        self.tensors: dict[str, Any] = {}

    def install(self, layer: Any) -> None:
        original = layer.forward

        def wrapper(layer_self: Any, x: Any, residual: Any = None) -> Any:
            del layer_self
            should_record = self.calls == 0
            if should_record:
                if residual is None:
                    raise RuntimeError("layer-0 post-attention norm had no residual")
                self.tensors["o_projection_output"] = x.detach().clone()
                self.tensors["residual"] = residual.detach().clone()
                self.tensors["norm_weight"] = layer.weight.detach().clone()
            result = original(x, residual)
            if should_record:
                self.torch.cuda.synchronize()
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError("unexpected post-attention RMSNorm result")
                self.tensors["norm_output"] = result[0].detach().clone()
                self.tensors["updated_residual"] = result[1].detach().clone()
                self.calls += 1
            return result

        layer.forward = MethodType(wrapper, layer)

    def validate(self) -> None:
        expected = {
            "o_projection_output", "residual", "norm_weight", "norm_output", "updated_residual"
        }
        if self.calls != 1 or set(self.tensors) != expected:
            raise RuntimeError(f"incomplete eager capture: calls={self.calls}, tensors={sorted(self.tensors)}")


def save_capture(torch: Any, run_dir: Path, tensors: dict[str, Any]) -> dict[str, Any]:
    tensor_dir = run_dir / "tensors"
    tensor_dir.mkdir()
    summaries = {}
    for name, value in sorted(tensors.items()):
        cpu = value.detach().cpu()
        path = tensor_dir / f"{name}.pt"
        torch.save(cpu, path)
        summaries[name] = {
            **base.tensor_summary(torch, value, row_hashes=True),
            "path": path.relative_to(run_dir).as_posix(),
            "file_sha256": base.sha256_file(path),
        }
    return summaries


def source_record(path_text: str | None, cache_root: Path) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = Path(path_text)
    return {
        "path_under_cache": path.resolve().relative_to(cache_root.resolve()).as_posix(),
        "size": path.stat().st_size,
        "sha256": base.sha256_file(path),
    }


def worker(arguments: argparse.Namespace) -> None:
    run_dir = arguments.run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"refusing existing worker directory: {run_dir}")
    cache_root = run_dir / "cache"
    cache_root.mkdir(parents=True)
    environment = {
        "VLLM_PLUGINS": "", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TORCHINDUCTOR_COMPILE_THREADS": "1", "VLLM_LOGGING_LEVEL": "DEBUG",
    }
    os.environ.pop("VLLM_DEBUG_DUMP_PATH", None)
    os.environ.update(environment)
    base.write_json(run_dir / "environment.json", environment)
    base.write_json(run_dir / "nvidia_smi_before.json", nvidia_smi())
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt

        base.write_json(run_dir / "provenance.json", base.module_provenance())
        if not base.version_matches(vllm.__version__, arguments.expected_vllm):
            raise RuntimeError(f"expected vLLM {arguments.expected_vllm}, got {vllm.__version__}")
        mode = 3 if arguments.compile == "on" else 0
        llm = LLM(
            model=base.MODEL, tensor_parallel_size=1, seed=0, max_model_len=2048,
            enable_prefix_caching=False, enforce_eager=False,
            compilation_config={"mode": mode, "cudagraph_mode": "NONE"},
            revision=base.REVISION, tokenizer_revision=base.REVISION,
        )
        config = llm.llm_engine.vllm_config
        resolved = {
            "vllm": vllm.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "compile_requested": arguments.compile, "mode": config.compilation_config.mode.value,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "use_v2_model_runner": config.use_v2_model_runner,
        }
        base.write_json(run_dir / "resolved.json", resolved)
        if resolved["mode"] != mode or resolved["cudagraph_mode"] != "NONE" or resolved["use_v2_model_runner"] is not True:
            raise RuntimeError(f"configuration mismatch: {resolved}")
        tokenizer = llm.get_tokenizer()
        prompt_ids = [tokenizer.encode(prompt) for prompt in base.PROMPTS]
        if arguments.compile == "on":
            recorder, candidates = install_generated_recorders(torch, cache_root / "torchinductor")
            selected_name = None
        else:
            selected_name, norm = find_layer0_post_attention_norm(get_model(llm))
            recorder = EagerNormRecorder(torch)
            recorder.install(norm)
            candidates = []
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=1, logprobs=5, ignore_eos=True),
            use_tqdm=False,
        )
        recorder.validate()
        summaries = save_capture(torch, run_dir, recorder.tensors)
        result = {
            "schema": 1, "model": base.MODEL, "revision": base.REVISION,
            "compile": arguments.compile, "prompts": base.PROMPTS,
            "prompt_token_ids": prompt_ids, "output": base.output_summary(outputs),
            "capture": summaries, "selected_norm": selected_name,
            "generated_module_candidates": [str(Path(x).relative_to(cache_root)) for x in candidates],
        }
        if arguments.compile == "on":
            result["selected_generated_source"] = source_record(recorder.module_path, cache_root)
        base.write_json(run_dir / "result.json", result)
        base.write_json(run_dir / "nvidia_smi_after.json", nvidia_smi())
        base.write_json(run_dir / "cache_after.json", base.cache_snapshot(cache_root))
    except BaseException as error:
        base.write_json(run_dir / "failure.json", {
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()
        })
        base.write_json(run_dir / "nvidia_smi_after.json", nvidia_smi())
        base.write_json(run_dir / "cache_after.json", base.cache_snapshot(cache_root))
        raise


def tensor_compare(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    if shape(left) != shape(right):
        return {"same_shape": False, "left_shape": shape(left), "right_shape": shape(right)}
    left_f, right_f = left.float(), right.float()
    difference = (left_f - right_f).abs()
    rows = difference.reshape(difference.shape[0], -1).amax(dim=1)
    return {
        "same_shape": True, "exact": bool(torch.equal(left, right)),
        "elements_differing": int(torch.count_nonzero(left != right).item()),
        "rows_differing": int(torch.count_nonzero(rows).item()),
        "max_abs": float(difference.max().item()), "mean_abs": float(difference.mean().item()),
    }


def analyze(output_dir: Path) -> dict[str, Any]:
    import torch

    compiled_path, eager_path = output_dir / "compiled-operator", output_dir / "eager-boundary"
    if not (compiled_path / "result.json").is_file() or not (eager_path / "result.json").is_file():
        return {"schema": 1, "complete": False}
    load = lambda root, name: torch.load(root / "tensors" / f"{name}.pt", map_location="cpu", weights_only=True)
    compiled_result = json.loads((compiled_path / "result.json").read_text())
    eager_result = json.loads((eager_path / "result.json").read_text())
    compiled = {name: load(compiled_path, name) for name in (
        "attention_output", "o_projection_matrix", "o_projection_output", "norm_projection_input",
        "residual", "norm_weight", "norm_output"
    )}
    eager = {name: load(eager_path, name) for name in (
        "o_projection_output", "residual", "norm_weight", "norm_output", "updated_residual"
    )}
    summed = compiled["o_projection_output"].float() + compiled["residual"].float()
    reference = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + 1e-6)
    reference = (reference * compiled["norm_weight"].float()).to(compiled["norm_output"].dtype)
    comparisons = {
        "compiled_projection_vs_eager_projection": tensor_compare(torch, compiled["o_projection_output"], eager["o_projection_output"]),
        "compiled_residual_vs_eager_residual": tensor_compare(torch, compiled["residual"], eager["residual"]),
        "compiled_weight_vs_eager_weight": tensor_compare(torch, compiled["norm_weight"].reshape(1, -1), eager["norm_weight"].reshape(1, -1)),
        "compiled_norm_vs_eager_norm": tensor_compare(torch, compiled["norm_output"], eager["norm_output"]),
        "compiled_norm_vs_fp32_formula": tensor_compare(torch, compiled["norm_output"], reference),
        "compiled_sum_vs_eager_updated_residual": tensor_compare(torch, summed.to(eager["updated_residual"].dtype), eager["updated_residual"]),
    }
    order = [
        ("o_projection", comparisons["compiled_projection_vs_eager_projection"]),
        ("residual", comparisons["compiled_residual_vs_eager_residual"]),
        ("post_attention_rms_norm", comparisons["compiled_norm_vs_eager_norm"]),
    ]
    first = next((name for name, value in order if not value.get("exact", False)), None)
    return {
        "schema": 1, "complete": True,
        "instrumentation_preserves_compiled_tokens": compiled_result["output"]["token_ids"] == BASELINE_COMPILED_TOKENS,
        "compiled_vs_eager_output_tokens": compiled_result["output"]["token_ids"] == eager_result["output"]["token_ids"],
        "comparisons": comparisons, "first_differing_post_attention_stage": first,
        "interpretation_limit": "This comparison relies on the previously measured exact layer-0 attention output alignment; it does not infer request order from duplicate blocks.",
    }


def controller(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    base.write_json(output_dir / "manifest.json", {
        "schema": 1, "created_at": datetime.now(UTC).isoformat(), "command": sys.argv,
        "python": sys.executable, "model": base.MODEL, "revision": base.REVISION,
        "max_tokens": 1, "cells": ["compiled-operator", "eager-boundary"],
        "baseline_compiled_tokens": BASELINE_COMPILED_TOKENS,
    })
    completed = []
    for name, compile_mode in (("compiled-operator", "on"), ("eager-boundary", "off")):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--run-dir",
                   str(output_dir / name), "--compile", compile_mode,
                   "--expected-vllm", arguments.expected_vllm]
        returncode, timed_out = base.run_cell(
            command, output_dir / f"{name}.stdout.log", output_dir / f"{name}.stderr.log",
            arguments.timeout_seconds,
        )
        completed.append({"name": name, "compile": compile_mode, "returncode": returncode, "timed_out": timed_out})
        report = {"schema": 1, "cells": completed}
        if all(item["returncode"] == 0 for item in completed) and len(completed) == 2:
            report.update(analyze(output_dir))
        base.write_json(output_dir / "comparison.json", report)
        if returncode:
            break
        (output_dir / name / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        if arguments.require_ack:
            manifest = base.handoff_manifest(output_dir, name)
            base.wait_for_ack(output_dir, name, manifest, arguments.ack_timeout_seconds)
            base.audit_handoffs(output_dir)
    if any(cell["returncode"] for cell in completed):
        raise SystemExit("post-attention cell failed; no later GPU process was started")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compile", choices=("on", "off"), default="on", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("post-attention-operator-localise"))
    parser.add_argument("--expected-vllm", default="0.28.0")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--require-ack", action="store_true")
    parser.add_argument("--ack-timeout-seconds", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.worker and arguments.run_dir is None:
        parser.error("--worker requires --run-dir")
    return arguments


def main() -> None:
    arguments = parse_args()
    worker(arguments) if arguments.worker else controller(arguments)


if __name__ == "__main__":
    main()
