#!/usr/bin/env python3
"""Run #56900 with Inductor's eager-numerics precision casts enabled.

The layer-0 boundary probe found that eager RMSNorm consumes a bf16-rounded
residual sum while the default compiled graph normalizes the unrounded fp32
sum. This full-model control asks Inductor to retain eager's low-precision
operator-boundary casts. It tests whether that numerical difference changes
the first-token phenotype; it does not by itself establish root cause.
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


CELL = "compiled-emulate-precision-casts"


def generated_source_audit(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    rows = []
    for path in files:
        text = path.read_text(errors="replace")
        calls = text.count("torch.ops.vllm.moe_forward_shared")
        fusion_disabled = text.count("'enable_fp_fusion': False")
        if calls or fusion_disabled:
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": base.sha256_file(path),
                    "moe_forward_shared_occurrences": calls,
                    "enable_fp_fusion_false_occurrences": fusion_disabled,
                }
            )
    return {
        "python_sources": len(files),
        "moe_forward_shared_occurrences": sum(
            item["moe_forward_shared_occurrences"] for item in rows
        ),
        "enable_fp_fusion_false_occurrences": sum(
            item["enable_fp_fusion_false_occurrences"] for item in rows
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
        "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
    }
    os.environ.pop("VLLM_DEBUG_DUMP_PATH", None)
    os.environ.update(environment)
    base.write_json(run_dir / "environment.json", environment)
    base.write_json(run_dir / "cache_before.json", base.cache_snapshot(cache_root))
    base.write_json(run_dir / "nvidia_smi_before.json", nvidia_smi())
    try:
        import torch
        import torch._inductor.config as inductor_config

        config_after = {
            "emulate_precision_casts": inductor_config.emulate_precision_casts,
            "inplace_buffers": inductor_config.inplace_buffers,
            "allow_buffer_reuse": inductor_config.allow_buffer_reuse,
        }

        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt

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
            "inductor": config_after,
        }
        base.write_json(run_dir / "resolved.json", resolved)
        if (
            resolved["mode"] != 3
            or resolved["cudagraph_mode"] != "NONE"
            or resolved["use_v2_model_runner"] is not True
            or config_after["emulate_precision_casts"] is not True
        ):
            raise RuntimeError(f"configuration mismatch: {resolved}")
        tokenizer = llm.get_tokenizer()
        prompt_ids = [tokenizer.encode(prompt) for prompt in base.PROMPTS]
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=1, logprobs=5, ignore_eos=True),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        source_audit = generated_source_audit(cache_root / "torchinductor")
        base.write_json(run_dir / "generated_source_audit.json", source_audit)
        if source_audit["moe_forward_shared_occurrences"] == 0:
            raise RuntimeError("generated sources do not contain moe_forward_shared")
        if source_audit["enable_fp_fusion_false_occurrences"] == 0:
            raise RuntimeError(
                "generated sources do not attest disabled floating-point fusion"
            )
        base.write_json(
            run_dir / "result.json",
            {
                "model": base.MODEL,
                "revision": base.REVISION,
                "prompts": base.PROMPTS,
                "prompt_token_ids": prompt_ids,
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
                "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
                "torch._inductor.config.emulate_precision_casts": True,
            },
            "model": base.MODEL,
            "revision": base.REVISION,
            "max_tokens": 1,
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
        raise SystemExit("precision-cast control failed; no later GPU process was started")
    (output_dir / CELL / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if arguments.require_ack:
        manifest = base.handoff_manifest(output_dir, CELL)
        base.wait_for_ack(output_dir, CELL, manifest, arguments.ack_timeout_seconds)
        base.audit_handoffs(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("moe-precision-casts"))
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
