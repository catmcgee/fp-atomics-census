"""Exercise generic PyTorch operators; this does not confirm engine call sites.

    python probes/probe_torch_ops.py

Runs, on CUDA, with contended indices: index_add_ (vllm-0045 mean pooling,
vllm-0046 Moondream3), scatter_add_ (flash-attention-0026), float cumsum
(vllm-0049, vllm-0050, sglang-0031, sglang-0035, flash-attention-0027,
flashinfer-0048) and bincount with weights, which has no inventory row.
Each is run twice with the same inputs and compared bit for bit, first
with torch.use_deterministic_algorithms(False) and then with True.
"""
from __future__ import annotations

import sys

import torch

from common import run_twice


def main() -> int:
    torch.manual_seed(0)
    n_tokens, hidden, n_seg = 65536, 4096, 8
    x = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.float32)
    seg = torch.randint(0, n_seg, (n_tokens,), device="cuda")
    probs = torch.softmax(torch.randn(256, 152064, device="cuda"), dim=-1)
    tokens = torch.randint(0, 152064, (1 << 20,), device="cuda")
    weights = torch.rand(1 << 20, device="cuda")
    results = []
    for det in (False, True):
        torch.use_deterministic_algorithms(det, warn_only=False)
        tag = "det_strict" if det else "default"
        results.append(run_twice(f"index_add_{tag}", lambda: [torch.zeros(n_seg, hidden, device="cuda").index_add_(0, seg, x)]))
        results.append(run_twice(f"scatter_add_{tag}", lambda: [torch.zeros(n_seg, hidden, device="cuda").scatter_add_(0, seg[:, None].expand_as(x), x)]))
        results.append(run_twice(f"cumsum_float_{tag}", lambda: [torch.cumsum(probs, dim=-1)]))
        results.append(run_twice(f"bincount_weights_{tag}", lambda: [torch.bincount(tokens, weights=weights, minlength=152064)]))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
