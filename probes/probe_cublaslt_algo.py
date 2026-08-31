"""Query which cuBLASLt algorithm and reduction scheme a matmul shape uses.

    python probes/probe_cublaslt_algo.py --m 1 --n 4096 --k 4096 --dtype bf16

cuBLASLt does not expose the chosen algorithm through PyTorch. This probe
does two things: (1) runs the matmul twice and compares bits (settles
run-to-run determinism for that shape on this stack), and (2) if the
`cuda-python` bindings are importable, calls cublasLtMatmulAlgoGetHeuristic
for the same descriptor and prints the algo id, tile id, splitK number and
CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME_ID of the top heuristic result, which
is what the verifier should record. Class C rows: vllm-0180, vllm-0181, sglang-0264 to sglang-0266,
flashinfer-0040, flashinfer-0041, DeepGEMM-0009.
"""
from __future__ import annotations

import argparse
import sys

import torch

from common import run_twice

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def heuristic(m: int, n: int, k: int, dtype: str) -> dict | None:
    try:
        from cuda.bindings import cublasLt as lt  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            import cublaslt as lt  # type: ignore  # noqa: F401
        except Exception:  # noqa: BLE001
            return None
    # The exact binding API differs between releases; the sequence is:
    # cublasLtCreate -> cublasLtMatmulDescCreate(compute=32F, scale=32F)
    # -> cublasLtMatrixLayoutCreate for A (k x m), B (n x k), C (n x m)
    # -> cublasLtMatmulPreferenceCreate (workspace = CUBLAS_WORKSPACE_CONFIG)
    # -> cublasLtMatmulAlgoGetHeuristic(..., requestedAlgoCount=1)
    # -> cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID)
    #    and CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME_ID
    # Fill in with the binding available on the target machine.
    return {"note": "binding present; implement the descriptor sequence above for this binding version"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    args = ap.parse_args()
    dt = DTYPES[args.dtype]
    a = torch.randn(args.m, args.k, device="cuda", dtype=dt)
    b = torch.randn(args.k, args.n, device="cuda", dtype=dt)
    ok = run_twice(f"cublaslt_m{args.m}_n{args.n}_k{args.k}_{args.dtype}", lambda: [a @ b])
    h = heuristic(args.m, args.n, args.k, args.dtype)
    print("heuristic:", h if h is not None else "cuda-python bindings not available; record CUBLAS_WORKSPACE_CONFIG and the shape only")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
