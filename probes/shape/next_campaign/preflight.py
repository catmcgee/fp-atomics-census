#!/usr/bin/env python3
"""Fail-closed compatibility check before replaying a historical C2 record."""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"


def load(path: Path):
    return json.loads(path.read_text())


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def collect_actual(required_gpus: int) -> dict:
    import torch
    import shape_hook

    packages = {
        dist.metadata["Name"]: dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    drivers = command(
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
    ).splitlines()
    vbios = command(
        "nvidia-smi", "--query-gpu=vbios_version", "--format=csv,noheader"
    ).splitlines()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "nccl_runtime_version": list(torch.cuda.nccl.version()),
        "gpu": names[0] if names else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": names,
        "sm": (
            f"{torch.cuda.get_device_properties(0).major}{torch.cuda.get_device_properties(0).minor}"
            if names
            else None
        ),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count if names else None,
        "driver": drivers[0].strip() if drivers else None,
        "drivers": [x.strip() for x in drivers],
        "vbios": [x.strip() for x in vbios],
        "installed_distributions": dict(sorted(packages.items())),
        "shape_hook_file": str(Path(shape_hook.__file__).resolve()),
        "shape_hook_file_sha256": __import__("hashlib").sha256(
            Path(shape_hook.__file__).read_bytes()
        ).hexdigest(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "VLLM_DISABLE_COMPILE_CACHE",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_USE_V2_MODEL_RUNNER",
                "NCCL_NVLS_ENABLE",
            )
        },
        "required_gpus": required_gpus,
    }


def weight_index(
    rows: list[dict], required_models: set[tuple[str, str]], label: str
) -> tuple[dict, list[str]]:
    out = {}
    errors = []
    present = set()
    for row in rows:
        key_model = (row.get("repo"), row.get("revision"))
        if key_model not in required_models:
            continue
        present.add(key_model)
        key = (*key_model, row.get("file"))
        size, digest = row.get("size"), row.get("sha256")
        if key in out:
            errors.append(f"{label} weight manifest has duplicate row {key!r}")
            continue
        if not isinstance(key[-1], str) or not key[-1]:
            errors.append(f"{label} weight manifest has an empty file name for {key_model!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"{label} weight manifest has invalid size for {key!r}: {size!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{label} weight manifest has invalid SHA-256 for {key!r}: {digest!r}")
        out[key] = (size, digest)
    for model in sorted(required_models - present):
        errors.append(f"{label} weight manifest has no rows for required model {model!r}")
    return out, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--reference-env", type=Path, required=True)
    ap.add_argument("--required-gpus", type=int, required=True)
    ap.add_argument("--expected-vbios", required=True)
    ap.add_argument("--reference-weights", type=Path)
    ap.add_argument("--actual-weights", type=Path)
    ap.add_argument("--required-model", action="append", default=[])
    ap.add_argument("--required-env", action="append", default=[])
    ap.add_argument("--actual-json", type=Path, help="test/inspection input instead of probing this host")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    errors: list[str] = []
    try:
        source_sha = command("git", "-C", str(args.source_root), "rev-parse", "HEAD")
        source_status = command("git", "-C", str(args.source_root), "status", "--porcelain")
    except Exception as exc:  # retain a report instead of losing the failure
        source_sha, source_status = None, "unavailable"
        errors.append(f"source inspection failed: {type(exc).__name__}: {exc}")
    if source_sha != EXPECTED_SOURCE:
        errors.append(f"source commit {source_sha} != {EXPECTED_SOURCE}")
    if source_status:
        errors.append("source worktree is dirty")

    reference = load(args.reference_env)
    if reference.get("gpu_count") != args.required_gpus:
        errors.append(
            f"--required-gpus {args.required_gpus} != reference gpu_count {reference.get('gpu_count')!r}"
        )
    try:
        manifest = load(args.source_root / "scan-manifest.json")
        pinned = {row["name"]: row["sha"] for row in manifest["repos"]}
        expected_hook_sha256 = __import__("hashlib").sha256(
            (args.source_root / "probes/shape/shape_hook_pkg/shape_hook/__init__.py").read_bytes()
        ).hexdigest()
    except Exception as exc:
        pinned, expected_hook_sha256 = {}, None
        errors.append(f"source manifest/hook inspection failed: {type(exc).__name__}: {exc}")
    if pinned != reference.get("pinned_shas"):
        errors.append("source scan-manifest pins differ from the historical record")
    try:
        actual = load(args.actual_json) if args.actual_json else collect_actual(args.required_gpus)
    except Exception as exc:  # CUDA/import failures must still produce the durable report
        actual = {"collection_error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"runtime collection failed: {actual['collection_error']}")
    checks = {
        "python": (reference.get("python"), actual.get("python")),
        "torch": (reference.get("torch"), actual.get("torch")),
        "cuda": (reference.get("cuda"), actual.get("cuda")),
        "cudnn": (reference.get("cudnn"), actual.get("cudnn")),
        "nccl_runtime_version": (
            reference.get("nccl_runtime_version"),
            actual.get("nccl_runtime_version"),
        ),
        "gpu": (reference.get("gpu"), actual.get("gpu")),
        "gpu_count": (args.required_gpus, actual.get("gpu_count")),
        "sm": (reference.get("sm"), actual.get("sm")),
        "sm_count": (reference.get("sm_count"), actual.get("sm_count")),
        "driver": (reference.get("driver"), actual.get("driver")),
        "installed_distributions": (
            reference.get("installed_distributions"),
            actual.get("installed_distributions"),
        ),
        "shape_hook_file_sha256": (
            expected_hook_sha256,
            actual.get("shape_hook_file_sha256"),
        ),
    }
    for field, (want, got) in checks.items():
        if want != got:
            if field == "installed_distributions":
                want = want or {}
                got = got or {}
                missing = sorted(set(want) - set(got))
                extra = sorted(set(got) - set(want))
                changed = sorted(k for k in set(want) & set(got) if want[k] != got[k])
                errors.append(
                    f"installed distributions differ: missing={missing}, extra={extra}, "
                    f"changed={[(k, want[k], got[k]) for k in changed]}"
                )
            else:
                errors.append(f"{field} {got!r} != {want!r}")

    names = actual.get("gpu_names") or ([actual.get("gpu")] if actual.get("gpu") else [])
    if len(names) != args.required_gpus or any(n != reference.get("gpu") for n in names):
        errors.append(f"GPU list {names!r} does not contain exactly {args.required_gpus} copies of {reference.get('gpu')!r}")
    drivers = actual.get("drivers") or ([actual.get("driver")] if actual.get("driver") else [])
    if len(drivers) != args.required_gpus or any(d != reference.get("driver") for d in drivers):
        errors.append(f"driver list {drivers!r} does not match every GPU")
    vbios = actual.get("vbios") or []
    if len(vbios) != args.required_gpus or any(v != args.expected_vbios for v in vbios):
        errors.append(f"VBIOS list {vbios!r} != {args.required_gpus} copies of {args.expected_vbios!r}")

    required_env = {}
    for item in args.required_env:
        if "=" not in item:
            errors.append(f"bad --required-env {item!r}; expected KEY=VALUE")
            continue
        key, value = item.split("=", 1)
        required_env[key] = value
    actual_env = actual.get("env") or {}
    for key, value in required_env.items():
        if actual_env.get(key) != value:
            errors.append(f"environment {key}={actual_env.get(key)!r} != {value!r}")

    required_models = set()
    for item in args.required_model:
        if "@" not in item:
            errors.append(f"bad --required-model {item!r}; expected repo@revision")
            continue
        required_models.add(tuple(item.rsplit("@", 1)))
    if not required_models:
        errors.append("at least one --required-model is mandatory")
    if not args.reference_weights or not args.actual_weights:
        errors.append("--reference-weights and --actual-weights are both mandatory")
    else:
        try:
            want, weight_errors = weight_index(load(args.reference_weights), required_models, "reference")
            got, actual_weight_errors = weight_index(load(args.actual_weights), required_models, "actual")
            errors.extend(weight_errors + actual_weight_errors)
        except Exception as exc:
            want, got = {}, {}
            errors.append(f"weight manifest read failed: {type(exc).__name__}: {exc}")
        if want != got:
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            changed = sorted(k for k in set(want) & set(got) if want[k] != got[k])
            errors.append(
                f"model file digests differ: missing={missing}, extra={extra}, "
                f"changed={[(k, want[k], got[k]) for k in changed]}"
            )

    report = {
        "schema": 1,
        "compatible": not errors,
        "expected_source": EXPECTED_SOURCE,
        "actual_source": source_sha,
        "source_dirty": bool(source_status),
        "pinned_shas": pinned,
        "reference_env": str(args.reference_env),
        "required_gpus": args.required_gpus,
        "required_models": sorted([list(x) for x in required_models]),
        "required_env": required_env,
        "actual": actual,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"compatible": report["compatible"], "errors": errors}, indent=2))
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    sys.exit(main())
