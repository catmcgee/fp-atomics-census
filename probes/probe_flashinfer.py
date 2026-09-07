"""FlashInfer probes for the cubin surface and the renormalisation kernels.

    python probes/probe_flashinfer.py --which attention|moe|renorm|topk

attention: BatchDecodeWithPagedKVCacheWrapper with the FA2 template (expected
    identical) and, on sm90+/sm100, the trtllm-gen backend (flashinfer-0038, class C).
moe: cutlass_fused_moe with use_fused_finalize=True (flashinfer-0001) and False,
    at top-k 2, 4 and 8; order can only matter from three addends per row.
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


def moe(fused: bool, topk_: int, autotune_first: bool = False) -> bool:
    import flashinfer

    # Shapes follow the cutlass_fused_moe docstring; weights random. A token
    # routed to k experts receives k reduction-adds into its output row. With
    # k = 2 onto a zero-initialised row the sum is commutative and cannot
    # depend on order; k >= 3 is where arrival order can change the rounding.
    # cutlass_fused_moe returns [output, *scratch]; only output is compared.
    tokens, hidden, inter, experts = 4096, 2048, 1024, 8
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(tokens, experts, device="cuda")
    weights, ids = torch.topk(torch.softmax(logits, -1), topk_, dim=-1)

    def call():
        out = flashinfer.fused_moe.cutlass_fused_moe(
            x, ids.to(torch.int32), weights.to(torch.float32), w1, w2, torch.bfloat16, [], use_fused_finalize=fused)
        return [out[0] if isinstance(out, (list, tuple)) else out]

    if autotune_first:
        # Without autotuning FlashInfer runs a default tactic, which need not be a
        # fused-finalize config even when use_fused_finalize is true; tuning first
        # lets the runner pick among the fused configs (moe_kernels.h getConfigs).
        from flashinfer.autotuner import autotune
        with autotune(True):
            call()
    name = f"fi_cutlass_fused_moe{'_autotuned' if autotune_first else ''}_fused_finalize_{fused}_topk{topk_}"
    return run_twice(name, call, extra={"topk": topk_, "tokens": tokens, "experts": experts, "autotuned": autotune_first})


def b12x_moe() -> bool:
    """FlashInfer's SM120/SM121 CuTe DSL fused MoE (flashinfer-0006 to 0008, 0010).

    Weight preparation follows benchmarks/routines/moe.py (backend "b12x"):
    NVFP4 weights with swizzled block scales converted to the MMA layout, bf16
    activations, external top-k routing. Token counts pick the micro, static
    and dynamic backends; top-k 2, 4 and 8 vary the number of reduction-adds
    per output row in the bf16x2 scatter-add finalize.
    """
    import flashinfer
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fused_moe.cute_dsl.b12x_moe import b12x_fused_moe

    device = "cuda"
    hidden, inter, experts, sf_vec = 2048, 1024, 8, 16
    torch.manual_seed(0)
    w1_bf16 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device=device) / 10
    w2_bf16 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device=device) / 10
    gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_q, w1_sf = fp4_quantize(w1_bf16.view(experts * 2 * inter, hidden), global_scale=gs, sf_vec_size=sf_vec, is_sf_swizzled_layout=True)
    w1_weight = w1_q.view(experts, 2 * inter, hidden // 2)
    w1_weight_sf = convert_sf_to_mma_layout(w1_sf, m=2 * inter, k=hidden, num_groups=experts, sf_vec_size=sf_vec)
    w2_q, w2_sf = fp4_quantize(w2_bf16.view(experts * hidden, inter), global_scale=gs, sf_vec_size=sf_vec, is_sf_swizzled_layout=True)
    w2_weight = w2_q.view(experts, hidden, inter // 2)
    w2_weight_sf = convert_sf_to_mma_layout(w2_sf, m=hidden, k=inter, num_groups=experts, sf_vec_size=sf_vec)
    alpha = torch.ones(experts, device=device, dtype=torch.float32)
    ok = True
    for tokens in (16, 512, 4096):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) / 10
        logits = torch.randn(tokens, experts, device=device)
        for topk_ in (2, 4, 8):
            weights, ids = torch.topk(torch.softmax(logits, -1), topk_, dim=-1)
            ids = ids.to(torch.int32); weights = weights.to(torch.float32)

            def run():
                out = b12x_fused_moe(x, w1_weight, w1_weight_sf, w2_weight, w2_weight_sf, ids, weights, experts, topk_,
                                     w1_alpha=alpha, w2_alpha=alpha, fc2_input_scale=gs, quant_mode="nvfp4")
                return [out[0] if isinstance(out, (list, tuple)) else out]
            ok &= run_twice(f"fi_b12x_moe_nvfp4_tokens{tokens}_topk{topk_}", run, extra={"tokens": tokens, "topk": topk_, "experts": experts})
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["attention", "moe", "renorm", "topk", "b12x_moe"], required=True)
    ap.add_argument("--backend", default="fa2")
    ap.add_argument("--autotune", action="store_true", help="moe: run FlashInfer autotuning before comparing")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if args.which == "renorm":
        ok = renorm()
    elif args.which == "topk":
        ok = topk()
    elif args.which == "attention":
        ok = attention(args.backend)
    elif args.which == "b12x_moe":
        ok = b12x_moe()
    else:
        ok = True
        for k in (2, 4, 8):
            ok &= moe(True, k, args.autotune) & moe(False, k, args.autotune)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
