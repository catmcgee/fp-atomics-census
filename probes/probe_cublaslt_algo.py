"""Which cuBLASLt algorithm a matmul shape gets, and whether it is run-to-run stable.

    python probes/probe_cublaslt_algo.py --m 1 --n 4096 --k 4096 --dtype bf16
    python probes/probe_cublaslt_algo.py --sweep 7b        # decode and prefill shapes of a 7B/8B model
    python probes/probe_cublaslt_algo.py --sweep 70b

Two instruments. (1) The matmul is run several times and compared bit for
bit. (2) The same matmul is run in a child process with cuBLASLt logging on
(CUBLASLT_LOG_LEVEL=5) and the log is parsed for the algorithm the
heuristic chose: algo id, tile, splitK and reductionScheme. A splitK above
1 with a reduction scheme other than NONE is the only way cuBLASLt reduces
across CTAs, so that line is the evidence a verifier should archive.

Class C rows: vllm-0180, vllm-0181, sglang-0264 to sglang-0266,
flashinfer-0040, flashinfer-0041, DeepGEMM-0009, flash-attention-0014.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from common import run_twice, stack_dir

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (n, k) pairs of the projections in a Llama/Qwen style block; batch M varies.
SWEEPS = {
    "7b": {"nk": [(4096, 4096), (12288, 4096), (11008, 4096), (4096, 11008), (152064, 4096)], "m": [1, 4, 8, 32, 128, 1024]},
    "70b": {"nk": [(8192, 8192), (10240, 8192), (28672, 8192), (8192, 28672), (128256, 8192)], "m": [1, 4, 8, 32, 128, 1024]},
}

_LOG_KEYS = ("algoId", "tile", "splitK", "reductionScheme", "stages", "cluster", "swizzle", "custom")


def _child_matmul(m: int, n: int, k: int, dtype: str) -> None:
    dt = DTYPES[dtype]
    a = torch.randn(m, k, device="cuda", dtype=dt)
    b = torch.randn(k, n, device="cuda", dtype=dt)
    torch.cuda.synchronize()
    (a @ b).sum().item()
    torch.cuda.synchronize()


def heuristic_from_log(m: int, n: int, k: int, dtype: str) -> dict:
    """Run one matmul in a child process with cuBLASLt logging and parse the algo it chose."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "cublaslt.log"
        env = dict(os.environ, CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=str(log), CUBLASLT_LOG_MASK="0")
        env.pop("CUBLASLT_LOG_MASK")
        cmd = [sys.executable, __file__, "--child", "--m", str(m), "--n", str(n), "--k", str(k), "--dtype", dtype]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        text = log.read_text(errors="replace") if log.exists() else ""
    entries = []
    for line in text.splitlines():
        if "cublasLtMatmul" in line and ("splitK" in line or "algoId" in line or "reductionScheme" in line):
            entry = {key: val for key, val in re.findall(r"(\w+)=([\w.:-]+)", line) if key in _LOG_KEYS}
            if entry:
                entries.append(entry)
    return {"entries": entries[:8], "log_lines": len(text.splitlines()), "child_rc": proc.returncode,
            "child_stderr_tail": proc.stderr[-400:] if proc.returncode else ""}


def probe(m: int, n: int, k: int, dtype: str) -> tuple[bool, dict]:
    dt = DTYPES[dtype]
    a = torch.randn(m, k, device="cuda", dtype=dt)
    b = torch.randn(k, n, device="cuda", dtype=dt)
    h = heuristic_from_log(m, n, k, dtype)
    ok = run_twice(f"cublaslt_m{m}_n{n}_k{k}_{dtype}", lambda: [a @ b], extra={"shape": [m, n, k], "dtype": dtype, "heuristic": h})
    return ok, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        _child_matmul(args.m, args.n, args.k, args.dtype)
        return 0
    shapes = [(m, n, k) for (n, k) in SWEEPS[args.sweep]["nk"] for m in SWEEPS[args.sweep]["m"]] if args.sweep else [(args.m, args.n, args.k)]
    all_ok = True
    rows = []
    for m, n, k in shapes:
        ok, h = probe(m, n, k, args.dtype)
        all_ok &= ok
        first = h["entries"][0] if h["entries"] else {}
        rows.append({"m": m, "n": n, "k": k, "identical": ok, **first})
        print(f"  m={m:5d} n={n:6d} k={k:6d} identical={ok} algo={first}")
    (stack_dir() / f"cublaslt_sweep_{args.sweep or 'single'}_{args.dtype}.json").write_text(json.dumps(rows, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
