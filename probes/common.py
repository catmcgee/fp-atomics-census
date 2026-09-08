"""Runtime measurements with immutable run directories and explicit outcomes.

Each probe evaluates one baseline plus ``repeats`` comparisons. A null result
is conditional on its inputs, stack and comparison count. Sparse engine API
observations are not full-logit or KV commitments.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import uuid
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

try:
    import torch
except ImportError:  # Offline census and analysis do not require PyTorch.
    torch = None

RESULTS = Path(__file__).resolve().parent / "results"
RUN_ID = os.environ.get("PROBE_RUN_ID") or str(uuid.uuid4())
if not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_ID):
    raise ValueError("PROBE_RUN_ID must be a single safe path component")
MANIFEST = Path(__file__).resolve().parent.parent / "scan-manifest.json"


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__ if torch else None,
        "cuda": torch.version.cuda if torch else None,
        "cudnn": torch.backends.cudnn.version() if torch and torch.backends.cudnn.is_available() else None,
        "packages": {p: _version(p) for p in ("vllm", "sglang", "sgl-kernel", "flashinfer-python", "flash-attn", "deep-gemm", "deep-ep", "triton", "nvidia-cublas-cu12", "nvidia-nccl-cu12", "nvidia-cublas-cu13", "nvidia-nccl-cu13")},
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "VLLM_BATCH_INVARIANT", "VLLM_MARLIN_USE_ATOMIC_ADD", "VLLM_ATTENTION_BACKEND")},
        "run_tag": os.environ.get("RUN_TAG"), "run_id": RUN_ID,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(),
        "installed_distributions": dict(sorted((d.metadata["Name"], d.version) for d in md.distributions() if d.metadata.get("Name"))),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled() if torch else None,
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled() if torch else None,
        "seed_policy": "torch seed 0 immediately before each evaluation; input construction seed must also be recorded",
        "selected_kernel_identity": None, "loaded_cubin_hashes": None,
    }
    if MANIFEST.exists():
        info["pinned_shas"] = {r["name"]: r["sha"] for r in json.loads(MANIFEST.read_text())["repos"]}
    if torch and torch.cuda.is_available():
        try:
            info["nccl_runtime_version"] = torch.cuda.nccl.version()
        except (AttributeError, RuntimeError):
            info["nccl_runtime_version"] = None
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "sm": f"{p.major}{p.minor}", "sm_count": p.multi_processor_count, "gpu_count": torch.cuda.device_count()})
        try:
            info["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split("\n")[0].strip()
        except Exception:  # noqa: BLE001
            info["driver"] = None
    return info


@lru_cache(maxsize=1)
def source_identity() -> dict:
    root = MANIFEST.parent
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = list((root / "probes").rglob("*.py"))
    digests = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "probe_source_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
            "files": digests}


def stack_dir(env: dict | None = None) -> Path:
    env = env or environment()
    raw = f"{env.get('gpu', 'cpu')}_{env.get('driver', 'nodriver')}_torch{env['torch']}"
    d = RESULTS / re.sub(r"[^A-Za-z0-9.+-]+", "-", raw).strip("-") / RUN_ID
    d.mkdir(parents=True, exist_ok=True)
    return d


def _as_bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.view(torch.int16)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.int8)
    return t


def tensor_hash(ts: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(json.dumps({"shape": list(t.shape), "dtype": str(t.dtype), "bytes": t.numel() * t.element_size()}, sort_keys=True).encode() + b"\0")
        h.update(_as_bits(t).cpu().numpy().tobytes())
    return h.hexdigest()


def bitwise_equal(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> tuple[bool, list[dict]]:
    diffs = []
    ok = len(a) == len(b)
    if not ok:
        diffs.append({"reason": "tensor count mismatch", "baseline": len(a), "comparison": len(b)})
    for i, (x, y) in enumerate(zip(a, b)):
        if x.dtype != y.dtype or x.shape != y.shape:
            ok = False
            diffs.append({"tensor": i, "reason": "dtype or shape mismatch", "baseline_dtype": str(x.dtype),
                          "comparison_dtype": str(y.dtype), "baseline_shape": list(x.shape), "comparison_shape": list(y.shape)})
            continue
        xb, yb = _as_bits(x).cpu(), _as_bits(y).cpu()
        if xb.shape != yb.shape or not torch.equal(xb, yb):
            ok = False
            n = int((xb != yb).sum()) if xb.shape == yb.shape else -1
            diffs.append({"tensor": i, "mismatched_elements": n, "numel": x.numel(), "shape": list(x.shape), "dtype": str(x.dtype), **ulp_diagnostics(x, y)})
    return ok, diffs


def ulp_diagnostics(x, y) -> dict:
    """ULP distances for finite binary16/bfloat16/binary32 outputs."""
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return {}
    finite = torch.isfinite(x) & torch.isfinite(y)
    bits_x, bits_y = _as_bits(x).to(torch.int64), _as_bits(y).to(torch.int64)
    sign = -(1 << (x.element_size() * 8 - 1))
    ox = torch.where(bits_x < 0, sign - bits_x, bits_x)
    oy = torch.where(bits_y < 0, sign - bits_y, bits_y)
    distances = (ox - oy).abs()[finite]
    return {"nonfinite_pairs": int((~finite).sum()),
            "max_ulp": int(distances.max()) if distances.numel() else None,
            "ulp_counts": {"0": int((distances == 0).sum()), "1": int((distances == 1).sum()),
                           "2": int((distances == 2).sum()), "3_or_more": int((distances >= 3).sum())}}


def write_report(report: dict) -> Path:
    tag = report["env"].get("run_tag")
    name = report["probe"]
    if any(c in name for c in "/\\") or (tag and any(c in tag for c in "/\\")):
        raise ValueError("report name/tag must be a filename component")
    path = stack_dir(report["env"]) / (f"{name}.{tag}.json" if tag else f"{name}.json")
    with path.open("x") as f:  # never replace an earlier observation
        json.dump(report, f, indent=2, allow_nan=False)
    return path


def record_status(name: str, status: str, reason: str, extra: dict | None = None) -> bool:
    if status not in ("ERROR", "SKIPPED", "REJECTED", "INVALID"):
        raise ValueError(status)
    write_report({"schema": 2, "probe": name, "env": environment(), "extra": extra or {},
                  "verdict": status, "reason": reason, "runs": [], "evaluations": 0})
    return False


def captured_tensor_inputs(fn) -> dict:
    """Hash tensors captured directly by a probe closure, without serialising engines."""
    captured = {}
    def visit(value, label):
        if torch.is_tensor(value):
            captured[label] = tensor_hash([value])
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                if torch.is_tensor(child):
                    visit(child, f"{label}/{i}")
        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, (str, int)) and torch.is_tensor(child):
                    visit(child, f"{label}/{key}")
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()) if hasattr(fn, "__code__") else []:
        visit(cell.cell_contents, name)
    return captured


def run_twice(name: str, fn: Callable[[], Sequence[torch.Tensor]], repeats: int = 3, extra: dict | None = None) -> bool:
    """Compare a baseline with repeats; persist failures separately from differences."""
    if repeats < 1:
        raise ValueError("at least one comparison is required")
    env = environment()
    report = {"schema": 2, "probe": name, "env": env, "extra": extra or {}, "runs": [],
              "hash_schema": "sha256-dtype-shape-bytes", "evaluations": 0,
              "independent_processes": 1, "seed": 0}
    report["captured_tensor_input_sha256"] = captured_tensor_inputs(fn)
    report["input_scope"] = "direct tensor closures and explicit extra metadata; engine internals/weights require separate commitments"
    key = {"captured_inputs": report["captured_tensor_input_sha256"], "extra": extra or {}, "source": env["source"]["probe_source_sha256"],
           "hash_schema": report["hash_schema"], "flags": env["env"],
           "deterministic": [env["deterministic_algorithms"], env["deterministic_warn_only"]]}
    report["comparison_key"] = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    verdict = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            for k in range(repeats + 1):
                torch.manual_seed(0)
                tensors = [t.clone() for t in fn()]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                report["evaluations"] += 1
                if not tensors or any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors):
                    report["verdict"] = "INVALID"
                    report["reason"] = "empty output or nonfinite tensor"
                    verdict = False
                    break
                if k == 0:
                    first = tensors
                    report["first_hash"] = tensor_hash(first)
                    continue
                ok, diffs = bitwise_equal(first, tensors)
                report["runs"].append({"run": k, "identical": ok, "diffs": diffs, "hash": tensor_hash(tensors)})
                verdict &= ok
            report.setdefault("verdict", "bitwise-identical" if verdict else "DIFFERS")
        except Exception as e:
            report["verdict"] = "REJECTED" if isinstance(e, RuntimeError) and "deterministic" in str(e).lower() and env["deterministic_algorithms"] and not env["deterministic_warn_only"] else "ERROR"
            report["reason"] = f"{type(e).__name__}: {e}"
            verdict = False
        report["warnings"] = [str(w.message) for w in caught]
    path = write_report(report)
    print(f"PROBE {name} {report['verdict']} -> {path}")
    return verdict


def maybe_print_hash(ts: Sequence[torch.Tensor]) -> None:
    if "--print-hash" in sys.argv:
        print(tensor_hash(ts))
