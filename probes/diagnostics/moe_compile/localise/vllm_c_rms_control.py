#!/usr/bin/env python3
"""Run #56900 with vLLM's native CUDA fused-add/RMSNorm provider.

Compiled CUDA normally lowers ``fused_add_rms_norm`` to vLLM's native torch
decomposition. That decomposition is the first measured compiled/eager
boundary difference. This control keeps compilation enabled but selects the
supported ``vllm_c`` provider, which lowers the IR op to
``torch.ops._C.fused_add_rms_norm``. It tests the full model with the eager
CUDA RMSNorm implementation without adding an artificial graph barrier.
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


CELL = "compiled-vllm-c-rms"
CUDA_OP = "torch.ops._C.fused_add_rms_norm"


def generated_source_audit(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    rows = []
    for path in files:
        text = path.read_text(errors="replace")
        moe_calls = text.count("torch.ops.vllm.moe_forward_shared")
        cuda_rms_calls = text.count(CUDA_OP)
        ir_rms_calls = text.count("torch.ops.vllm_ir.fused_add_rms_norm")
        if moe_calls or cuda_rms_calls or ir_rms_calls:
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": base.sha256_file(path),
                    "moe_forward_shared_occurrences": moe_calls,
                    "cuda_fused_add_rms_norm_occurrences": cuda_rms_calls,
                    "unlowered_ir_fused_add_rms_norm_occurrences": ir_rms_calls,
                }
            )
    return {
        "python_sources": len(files),
        "moe_forward_shared_occurrences": sum(
            item["moe_forward_shared_occurrences"] for item in rows
        ),
        "cuda_fused_add_rms_norm_occurrences": sum(
            item["cuda_fused_add_rms_norm_occurrences"] for item in rows
        ),
        "unlowered_ir_fused_add_rms_norm_occurrences": sum(
            item["unlowered_ir_fused_add_rms_norm_occurrences"] for item in rows
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
        "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
    }
    os.environ.pop("VLLM_DEBUG_DUMP_PATH", None)
    os.environ.update(environment)
    base.write_json(run_dir / "environment.json", environment)
    base.write_json(run_dir / "cache_before.json", base.cache_snapshot(cache_root))
    base.write_json(run_dir / "nvidia_smi_before.json", nvidia_smi())
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt, ir

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
            ir_op_priority={"fused_add_rms_norm": ["vllm_c"]},
            revision=base.REVISION,
            tokenizer_revision=base.REVISION,
        )
        config = llm.llm_engine.vllm_config
        configured_priority = config.kernel_config.ir_op_priority.fused_add_rms_norm
        runtime_priority = ir.ops.fused_add_rms_norm.get_priority()
        resolved = {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mode": config.compilation_config.mode.value,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "use_v2_model_runner": config.use_v2_model_runner,
            "configured_fused_add_rms_norm_priority": configured_priority,
            "runtime_fused_add_rms_norm_priority": runtime_priority,
            "supported_fused_add_rms_norm_providers": (
                ir.ops.fused_add_rms_norm.supported_providers()
            ),
        }
        base.write_json(run_dir / "resolved.json", resolved)
        if (
            resolved["mode"] != 3
            or resolved["cudagraph_mode"] != "NONE"
            or resolved["use_v2_model_runner"] is not True
            or not configured_priority
            or configured_priority[0] != "vllm_c"
            or not runtime_priority
            or runtime_priority[0] != "vllm_c"
            or "vllm_c" not in resolved["supported_fused_add_rms_norm_providers"]
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
        if source_audit["cuda_fused_add_rms_norm_occurrences"] == 0:
            raise RuntimeError(
                "generated sources do not call torch.ops._C.fused_add_rms_norm"
            )
        if source_audit["unlowered_ir_fused_add_rms_norm_occurrences"] != 0:
            raise RuntimeError(f"generated sources retain unlowered IR RMSNorm: {source_audit}")
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
            "cell": CELL,
            "control": {
                "vllm.ir.ops.fused_add_rms_norm.priority": ["vllm_c"],
                "expected_lowered_op": CUDA_OP,
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
        str(output_dir / CELL),
        "--expected-vllm",
        arguments.expected_vllm,
        "--max-tokens",
        str(arguments.max_tokens),
    ]
    returncode, timed_out = base.run_cell(
        command,
        output_dir / f"{CELL}.stdout.log",
        output_dir / f"{CELL}.stderr.log",
        arguments.timeout_seconds,
    )
    result_path = output_dir / CELL / "result.json"
    base.write_json(
        output_dir / "comparison.json",
        {
            "schema": 1,
            "cell": CELL,
            "returncode": returncode,
            "timed_out": timed_out,
            "result": json.loads(result_path.read_text()) if result_path.is_file() else None,
        },
    )
    if returncode:
        raise SystemExit("vllm-c RMSNorm control failed; no later GPU process was started")
    (output_dir / CELL / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if arguments.require_ack:
        manifest = base.handoff_manifest(output_dir, CELL)
        base.wait_for_ack(output_dir, CELL, manifest, arguments.ack_timeout_seconds)
        base.audit_handoffs(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("moe-vllm-c-rms"))
    parser.add_argument("--expected-vllm", default="0.28.0")
    parser.add_argument("--max-tokens", type=int, default=32)
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
