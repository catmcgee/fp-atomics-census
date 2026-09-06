"""Shared helpers for the runtime probes.

Every probe calls ``run_twice`` with a zero-argument function that returns
one or more tensors. The tensors are compared bit for bit; a mismatch in a
single element is reported as DIFFERS. Reports go to
``probes/results/<stack>/<name>[.<RUN_TAG>].json`` where ``<stack>`` is
derived from the GPU, driver and torch version, so results from different
machines never overwrite each other. Set ``RUN_TAG`` (e.g. ``a`` and ``b``)
to keep two fresh-process invocations apart; ``report.py`` compares their
hashes, which is only meaningful because every probe seeds torch before
building its inputs (PyTorch's default seed is random per process).
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
from pathlib import Path
from typing import Callable, Sequence

import torch

RESULTS = Path(__file__).resolve().parent / "results"
MANIFEST = Path(__file__).resolve().parent.parent / "scan-manifest.json"


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "packages": {p: _version(p) for p in ("vllm", "sglang", "sgl-kernel", "flashinfer-python", "flash-attn", "deep-gemm", "deep-ep", "triton", "nvidia-cublas-cu12", "nvidia-nccl-cu12")},
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "VLLM_BATCH_INVARIANT", "VLLM_MARLIN_USE_ATOMIC_ADD", "VLLM_ATTENTION_BACKEND")},
        "run_tag": os.environ.get("RUN_TAG"),
    }
    if MANIFEST.exists():
        info["pinned_shas"] = {r["name"]: r["sha"][:12] for r in json.loads(MANIFEST.read_text())["repos"]}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "sm": f"{p.major}{p.minor}", "sm_count": p.multi_processor_count, "gpu_count": torch.cuda.device_count()})
        try:
            info["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split("\n")[0].strip()
        except Exception:  # noqa: BLE001
            info["driver"] = None
    return info


def stack_dir(env: dict | None = None) -> Path:
    env = env or environment()
    raw = f"{env.get('gpu', 'cpu')}_{env.get('driver', 'nodriver')}_torch{env['torch']}"
    d = RESULTS / re.sub(r"[^A-Za-z0-9.+-]+", "-", raw).strip("-")
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
        h.update(str(tuple(t.shape)).encode())
        h.update(_as_bits(t).cpu().numpy().tobytes())
    return h.hexdigest()


def bitwise_equal(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> tuple[bool, list[dict]]:
    diffs = []
    ok = True
    for i, (x, y) in enumerate(zip(a, b)):
        xb, yb = _as_bits(x).cpu(), _as_bits(y).cpu()
        if xb.shape != yb.shape or not torch.equal(xb, yb):
            ok = False
            n = int((xb != yb).sum()) if xb.shape == yb.shape else -1
            diffs.append({"tensor": i, "mismatched_elements": n, "numel": x.numel(), "shape": list(x.shape), "dtype": str(x.dtype)})
    return ok, diffs


def run_twice(name: str, fn: Callable[[], Sequence[torch.Tensor]], repeats: int = 3, extra: dict | None = None) -> bool:
    """Run ``fn`` 1 + ``repeats`` times and compare every run with the first bit for bit."""
    env = environment()
    out_dir = stack_dir(env)
    tag = os.environ.get("RUN_TAG")
    torch.manual_seed(0)
    first = [t.clone() for t in fn()]
    torch.cuda.synchronize()
    report = {"probe": name, "env": env, "extra": extra or {}, "runs": [], "first_hash": tensor_hash(first)}
    verdict = True
    for k in range(repeats):
        torch.manual_seed(0)
        again = [t.clone() for t in fn()]
        torch.cuda.synchronize()
        ok, diffs = bitwise_equal(first, again)
        report["runs"].append({"run": k + 1, "identical": ok, "diffs": diffs})
        verdict &= ok
    report["verdict"] = "bitwise-identical" if verdict else "DIFFERS"
    fname = f"{name}.{tag}.json" if tag else f"{name}.json"
    (out_dir / fname).write_text(json.dumps(report, indent=2))
    print(f"PROBE {name} {report['verdict']} hash={report['first_hash'][:16]} -> {out_dir / fname}")
    return verdict


def maybe_print_hash(ts: Sequence[torch.Tensor]) -> None:
    if "--print-hash" in sys.argv:
        print(tensor_hash(ts))
