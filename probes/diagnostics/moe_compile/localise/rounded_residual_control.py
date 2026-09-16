#!/usr/bin/env python3
"""Compare matched opaque residual barriers with and without bf16 rounding.

The layer-0 boundary probe found that eager CUDA fused-add/RMSNorm normalizes a
bf16-rounded residual sum, while vLLM's native IR decomposition normalizes the
unrounded fp32 sum. Both arms change only the native fused_add_rms_norm
decomposition and insert the same opaque fp32-output custom op. The rounded arm
performs fp32 -> bf16 -> fp32 inside that op; the identity arm clones fp32.
Their matched fusion barrier distinguishes rounding from materialization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moe_boundary_localise as base  # noqa: E402


CELL_PREFIX = "compiled-opaque"
BARRIER_OP = "torch.ops.fp_atomics_census.residual_barrier.default"


def install_residual_barrier_decomposition(
    torch: Any, *, round_residual: bool
) -> dict[str, Any]:
    from vllm.ir.ops import fused_add_rms_norm

    def residual_barrier_impl(value: Any) -> Any:
        if value.dtype is not torch.float32:
            raise RuntimeError(f"residual_barrier expected float32, got {value.dtype}")
        if round_residual:
            return value.to(torch.bfloat16).to(torch.float32)
        return value.clone()

    # torch is intentionally imported only after worker environment setup, so
    # attach the live runtime types before asking torch.library to infer schema.
    residual_barrier_impl.__annotations__ = {
        "value": torch.Tensor,
        "return": torch.Tensor,
    }
    residual_barrier = torch.library.custom_op(
        "fp_atomics_census::residual_barrier", mutates_args=()
    )(residual_barrier_impl)

    @residual_barrier.register_fake
    def residual_barrier_fake(value: Any) -> Any:
        return torch.empty_like(value, dtype=torch.float32)

    native = fused_add_rms_norm.impls["native"]
    original = native.impl_fn
    priority_before = fused_add_rms_norm.get_priority()

    def rounded_fused_add_rms_norm(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype is not torch.bfloat16 or x_residual.dtype is not torch.bfloat16:
            raise RuntimeError(
                f"rounded-residual control requires bf16 activations, got "
                f"{x.dtype} and {x_residual.dtype}"
            )
        orig_dtype = x.dtype
        summed = x.to(torch.float32) + x_residual.to(torch.float32)
        # Both arms cross this exact opaque fp32 -> fp32 boundary. Only its
        # runtime implementation differs, so fusion/materialization is matched.
        normalized = residual_barrier(summed)
        x_residual = normalized.to(orig_dtype)
        x_var = normalized if variance_size is None else normalized[..., :variance_size]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + epsilon)
        if weight is not None:
            normalized = normalized.to(weight.dtype) * weight
        return normalized.to(orig_dtype), x_residual

    native.impl_fn = rounded_fused_add_rms_norm
    fused_add_rms_norm.set_default(["native"])
    return {
        "op": "vllm.ir.ops.fused_add_rms_norm",
        "provider": "native",
        "barrier_custom_op": BARRIER_OP,
        "barrier_output_dtype": "torch.float32",
        "round_residual": round_residual,
        "original_module": original.__module__,
        "original_qualname": original.__qualname__,
        "replacement_module": rounded_fused_add_rms_norm.__module__,
        "replacement_qualname": rounded_fused_add_rms_norm.__qualname__,
        "priority_before": priority_before,
        "priority_after": fused_add_rms_norm.get_priority(),
    }


def generated_source_audit(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    rows = []
    for path in files:
        text = path.read_text(errors="replace")
        calls = text.count("torch.ops.vllm.moe_forward_shared")
        barrier_calls = text.count(BARRIER_OP)
        if calls or barrier_calls:
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": base.sha256_file(path),
                    "moe_forward_shared_occurrences": calls,
                    "residual_barrier_occurrences": barrier_calls,
                }
            )
    return {
        "python_sources": len(files),
        "moe_forward_shared_occurrences": sum(
            item["moe_forward_shared_occurrences"] for item in rows
        ),
        "residual_barrier_occurrences": sum(
            item["residual_barrier_occurrences"] for item in rows
        ),
        "relevant_sources": rows,
    }


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

        override = install_residual_barrier_decomposition(
            torch, round_residual=bool(arguments.round_residual)
        )

        provenance = base.module_provenance()
        provenance["nvidia_smi_before_model"] = nvidia_smi()
        base.write_json(run_dir / "provenance.json", provenance)
        if not base.version_matches(vllm.__version__, arguments.expected_vllm):
            raise RuntimeError(f"expected vLLM {arguments.expected_vllm}, got {vllm.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        llm = LLM(
            model=base.MODEL,
            tensor_parallel_size=1,
            seed=0,
            max_model_len=2048,
            enable_prefix_caching=False,
            enforce_eager=False,
            compilation_config={"mode": 3, "cudagraph_mode": "NONE"},
            revision=base.REVISION,
            tokenizer_revision=base.REVISION,
        )
        config = llm.llm_engine.vllm_config
        resolved = {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mode": config.compilation_config.mode.value,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "use_v2_model_runner": config.use_v2_model_runner,
            "fused_add_rms_norm_override": override,
        }
        base.write_json(run_dir / "resolved.json", resolved)
        if (
            resolved["mode"] != 3
            or resolved["cudagraph_mode"] != "NONE"
            or resolved["use_v2_model_runner"] is not True
            or override["priority_after"] != ["native"]
        ):
            raise RuntimeError(f"configuration mismatch: {resolved}")
        tokenizer = llm.get_tokenizer()
        prompt_ids = [tokenizer.encode(prompt) for prompt in base.PROMPTS]
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_ids],
            SamplingParams(
                temperature=0.0,
                max_tokens=arguments.max_tokens,
                logprobs=5,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        source_audit = generated_source_audit(cache_root / "torchinductor")
        base.write_json(run_dir / "generated_source_audit.json", source_audit)
        if source_audit["moe_forward_shared_occurrences"] == 0:
            raise RuntimeError("generated sources do not contain moe_forward_shared")
        if source_audit["residual_barrier_occurrences"] == 0:
            raise RuntimeError(
                "generated sources do not call the opaque residual barrier custom op"
            )
        base.write_json(
            run_dir / "result.json",
            {
                "model": base.MODEL,
                "revision": base.REVISION,
                "prompts": base.PROMPTS,
                "prompt_token_ids": prompt_ids,
                "sampling": {
                    "temperature": 0.0,
                    "max_tokens": arguments.max_tokens,
                    "logprobs": 5,
                    "ignore_eos": True,
                },
                "output": base.output_summary(outputs),
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


def controller(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    arm = "rounded" if arguments.round_residual else "identity"
    cell = f"{CELL_PREFIX}-{arm}"
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
            "cell": cell,
            "control": {
                "vllm.ir.ops.fused_add_rms_norm.native": "opaque residual barrier",
                "round_residual": bool(arguments.round_residual),
                "barrier_output_dtype": "torch.float32",
            },
            "model": base.MODEL,
            "revision": base.REVISION,
            "max_tokens": arguments.max_tokens,
        },
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(output_dir / cell),
        "--expected-vllm",
        arguments.expected_vllm,
        "--round-residual",
        str(arguments.round_residual),
        "--max-tokens",
        str(arguments.max_tokens),
    ]
    returncode, timed_out = base.run_cell(
        command,
        output_dir / f"{cell}.stdout.log",
        output_dir / f"{cell}.stderr.log",
        arguments.timeout_seconds,
    )
    result_path = output_dir / cell / "result.json"
    base.write_json(
        output_dir / "comparison.json",
        {
            "schema": 1,
            "cell": cell,
            "round_residual": bool(arguments.round_residual),
            "returncode": returncode,
            "timed_out": timed_out,
            "result": json.loads(result_path.read_text()) if result_path.is_file() else None,
        },
    )
    if returncode:
        raise SystemExit("rounded-residual control failed; no later GPU process was started")
    (output_dir / cell / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if arguments.require_ack:
        manifest = base.handoff_manifest(output_dir, cell)
        base.wait_for_ack(output_dir, cell, manifest, arguments.ack_timeout_seconds)
        base.audit_handoffs(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("moe-rounded-residual"))
    parser.add_argument("--expected-vllm", default="0.28.0")
    parser.add_argument("--round-residual", type=int, choices=(0, 1), required=True)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--require-ack", action="store_true")
    parser.add_argument("--ack-timeout-seconds", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.worker and arguments.run_dir is None:
        parser.error("--worker requires --run-dir")
    if arguments.timeout_seconds <= 0 or arguments.ack_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    if arguments.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    worker(arguments) if arguments.worker else controller(arguments)


if __name__ == "__main__":
    main()
