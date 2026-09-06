"""FlashInfer probes for the cubin surface and the renormalisation kernels.

    python probes/probe_flashinfer.py --which attention|moe|renorm|topk

attention: BatchDecodeWithPagedKVCacheWrapper with the FA2 template (expected
    identical) and, on sm90+/sm100, the trtllm-gen backend (flashinfer-0038, class C).
moe: cutlass_fused_moe with use_fused_finalize=True (flashinfer-0001, expected
    DIFFERS under contention) and False (expected identical); the trtllm MoE
    entry points (class C).
renorm: top_p_renorm_probs with is_deterministic False and True (flashinfer-0014)
    and top_k_renorm_probs with a vocabulary large enough to span several CTAs
    (flashinfer-0015).
topk: flashinfer.topk.top_k with deterministic False and True; compares values and index order,
    not only the set (flashinfer-0017, flashinfer-0018).
"""
from __future__ import annotations

import argparse
import sys

import torch

from common import run_twice


def renorm() -> bool:
    import flashinfer

    probs = torch.softmax(torch.randn(64, 262144, device="cuda"), dim=-1)
    ok = True
    for det in (False, True):
        ok &= run_twice(f"fi_top_p_renorm_det{det}", lambda: [flashinfer.sampling.top_p_renorm_probs(probs, 0.9, is_deterministic=det)])
    ok &= run_twice("fi_top_k_renorm_multicta", lambda: [flashinfer.sampling.top_k_renorm_probs(probs, 2048)])
    return ok


def topk() -> bool:
    import flashinfer

    scores = torch.randn(32, 131072, device="cuda")
    ok = True
    for det in (False, True):
        ok &= run_twice(f"fi_radix_topk_order_det{det}", lambda: list(flashinfer.topk.top_k(scores, 2048, deterministic=det)))
    return ok


def attention(backend: str) -> bool:
    import flashinfer

    num_kv_heads, num_qo_heads, head_dim, page_size = 8, 32, 128, 16
    batch, max_pages = 16, 512
    kv = torch.randn(batch * max_pages, 2, page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    indptr = torch.arange(0, batch * max_pages + 1, max_pages, device="cuda", dtype=torch.int32)
    indices = torch.arange(batch * max_pages, device="cuda", dtype=torch.int32)
    last_len = torch.full((batch,), page_size, device="cuda", dtype=torch.int32)
    q = torch.randn(batch, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    w.plan(indptr, indices, last_len, num_qo_heads, num_kv_heads, head_dim, page_size, q_data_type=torch.bfloat16)
    return run_twice(f"fi_decode_{backend}", lambda: [w.run(q, kv)])


def moe(fused: bool) -> bool:
    import flashinfer

    # Shapes follow the cutlass_fused_moe docstring; weights random. Token
    # rows routed to several experts contend in the fused finalize.
    tokens, hidden, inter, experts, topk_ = 4096, 2048, 1024, 8, 2
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(tokens, experts, device="cuda")
    weights, ids = torch.topk(torch.softmax(logits, -1), topk_, dim=-1)
    return run_twice(f"fi_cutlass_fused_moe_fused_finalize_{fused}", lambda: [flashinfer.fused_moe.cutlass_fused_moe(
        x, ids.to(torch.int32), weights.to(torch.float32), w1, w2, torch.bfloat16, [], use_fused_finalize=fused)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["attention", "moe", "renorm", "topk"], required=True)
    ap.add_argument("--backend", default="fa2")
    args = ap.parse_args()
    if args.which == "renorm":
        ok = renorm()
    elif args.which == "topk":
        ok = topk()
    elif args.which == "attention":
        ok = attention(args.backend)
    else:
        ok = moe(True) & moe(False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
