"""Shared helpers for the runtime probes.

Every probe calls ``run_twice`` with a zero-argument function that returns
one or more tensors. The tensors are compared bit for bit; a mismatch in a
single element is reported as DIFFERS.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import torch


def environment() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "NCCL_ALGO": os.environ.get("NCCL_ALGO"),
        "NCCL_PROTO": os.environ.get("NCCL_PROTO"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["sm"] = f"{p.major}{p.minor}"
        info["sm_count"] = p.multi_processor_count
        try:
            info["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except Exception:  # noqa: BLE001
            info["driver"] = None
    return info


def _as_bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    if t.dtype in (torch.float32,):
        return t.view(torch.int32)
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.view(torch.int16)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def bitwise_equal(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> tuple[bool, list[dict]]:
    diffs = []
    ok = True
    for i, (x, y) in enumerate(zip(a, b)):
        xb, yb = _as_bits(x).cpu(), _as_bits(y).cpu()
        if xb.shape != yb.shape or not torch.equal(xb, yb):
            ok = False
            n = int((xb != yb).sum()) if xb.shape == yb.shape else -1
            diffs.append({"tensor": i, "mismatched_elements": n, "shape": list(x.shape), "dtype": str(x.dtype)})
    return ok, diffs


def run_twice(name: str, fn: Callable[[], Sequence[torch.Tensor]], repeats: int = 3, out_dir: Path | None = None) -> bool:
    out_dir = out_dir or Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    torch.manual_seed(0)
    first = [t.clone() for t in fn()]
    torch.cuda.synchronize()
    verdict = True
    report = {"probe": name, "env": environment(), "runs": []}
    for k in range(repeats):
        torch.manual_seed(0)
        again = [t.clone() for t in fn()]
        torch.cuda.synchronize()
        ok, diffs = bitwise_equal(first, again)
        report["runs"].append({"run": k + 1, "identical": ok, "diffs": diffs})
        verdict &= ok
    report["verdict"] = "bitwise-identical" if verdict else "DIFFERS"
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
    torch.save([t.cpu() for t in first], out_dir / f"{name}.first.pt")
    print(f"PROBE {name} {report['verdict']}")
    return verdict


def fresh_process_hash(script: str, args: list[str]) -> str:
    """Run a probe script in a new interpreter and return its printed hash line."""
    out = subprocess.check_output([sys.executable, script, *args, "--print-hash"], text=True)
    return out.strip().splitlines()[-1]


def tensor_hash(ts: Sequence[torch.Tensor]) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in ts:
        h.update(_as_bits(t).cpu().numpy().tobytes())
    return h.hexdigest()


def maybe_print_hash(ts: Sequence[torch.Tensor]) -> None:
    if "--print-hash" in sys.argv:
        print(tensor_hash(ts))
        time.sleep(0)
