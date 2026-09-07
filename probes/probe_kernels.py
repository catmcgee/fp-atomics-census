"""Direct kernel probes for the engine-specific class-A and low-confidence rows.

    python probes/probe_kernels.py --which vllm_moe_wna16|vllm_lora_shrink|sglang_marlin|sglang_fp8_blockwise|sglang_lora_shrink|deepgemm_bmk_bnk_mn|marlin_atomic

vllm_moe_wna16: vllm._custom_ops.moe_wna16_gemm with tokens/experts <= 6 (vllm-0014).
vllm_lora_shrink: vllm.lora.ops.triton_ops lora_shrink with the default config (vllm-0018).
sglang_marlin: sgl_kernel gptq_marlin_gemm with n < 2048, k >= 2048 (sglang-0008).
sglang_fp8_blockwise: sgl_kernel.fp8_blockwise_scaled_mm with k > 3n (sglang-0011).
sglang_lora_shrink: the trtllm_lora_temp sgemm_lora_a split-K path (sglang-0021).
deepgemm_bmk_bnk_mn: deep_gemm.einsum bmk,bnk->mn (DeepGEMM-0001/0002).
Each builds inputs of the shape that reaches the site, runs twice and compares bits.
"""
from __future__ import annotations

import argparse
import sys

import torch

from common import run_twice


def sglang_fp8_blockwise() -> bool:
    try:
        from sgl_kernel import fp8_blockwise_scaled_mm
    except ImportError:
        # sglang 0.5.19 moved the op into a JIT module and builds it for SM120 only; the
        # SM90 stream-K dispatcher of the pinned sha (sglang-0011) is not in the release wheel.
        from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
        print("note: using sglang.kernels.ops.gemm.fp8_blockwise_gemm; at 0.5.19 this kernel is documented as SM120-only")

    m, n, k = 512, 1024, 8192  # k > 3n selects the stream-K kernel
    a = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn).t()
    sa = torch.rand(m, k // 128, device="cuda", dtype=torch.float32)
    sb = torch.rand(k // 128, n // 128, device="cuda", dtype=torch.float32).t().contiguous().t()
    return run_twice("sglang_fp8_blockwise_streamk", lambda: [fp8_blockwise_scaled_mm(a, b, sa, sb, torch.bfloat16)])


def sglang_marlin() -> bool:
    from sglang.srt.layers.quantization.marlin_utils import should_use_atomic_add_reduce
    m, n, k = 16, 1024, 8192
    print("should_use_atomic_add_reduce:", should_use_atomic_add_reduce(m, n, k, torch.device("cuda"), torch.float16))
    print("Build a GPTQ-Marlin layer of shape (k, n) with the engine's loader and run probe_engine_logits with a model whose")
    print("projections satisfy n < 2048 and k >= 2048 under the chosen tensor-parallel size; the layer-level API needs packed weights.")
    return True


def vllm_moe_wna16() -> bool:
    print("Use probe_engine_logits.py with --quantization moe_wna16 on a 4-bit GPTQ MoE checkpoint (group size 128) and a")
    print("batch of at most 6 * num_experts tokens; the CUDA kernel is selected only in that regime (fused_moe.py:1227-1235).")
    return True


def vllm_lora_shrink() -> bool:
    from vllm.lora.ops.triton_ops import lora_shrink  # noqa: F401
    print("Use probe_engine_logits.py with --extra-arg enable_lora=True and a LoRA adapter; compare VLLM_BATCH_INVARIANT=0 and =1.")
    return True


def sglang_lora_shrink() -> bool:
    print("Use probe_engine_logits.py --engine sglang with --extra-arg lora-paths=<adapter> twice: once with the default")
    print("SGLANG_ENABLE_LORA_SHRINK_SPLIT_K and once with it set to false; compare the two JSON reports.")
    return True


def vllm_moe_wna16_kernel() -> bool:
    """vLLM's moe_wna16 CUDA kernel (vllm-0014) on random 4-bit weights.

    Mirrors tests/kernels/moe/test_moe.py::test_fused_moe_wn16. fused_moe
    selects the CUDA kernel when tokens per expert <= 6 (fused_moe.py
    should_moe_wna16_use_cuda); the m=64, e=8 case is the Triton control.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_moe
    from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
    from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
    from vllm.scalar_type import scalar_types

    ok = True
    e, topk_, n, k, group_size = 8, 2, 512, 1024, 128
    for dtype in (torch.float16, torch.bfloat16):
        for m in (16, 64):
            torch.manual_seed(0)
            a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
            w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
            score = torch.randn((m, e), device="cuda", dtype=dtype)
            w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
            w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
            w1_s = torch.empty((e, 2 * n, k // group_size), device="cuda", dtype=dtype)
            w2_s = torch.empty((e, k, n // group_size), device="cuda", dtype=dtype)
            for i in range(2 * e):
                ex = i % e
                w, wq, ws = (w1, w1_q, w1_s) if i < e else (w2, w2_q, w2_s)
                _, qweight, scales, _ = quantize_weights(w[ex].T, scalar_types.uint4b8, group_size, False, False)
                qweight = qweight.T.contiguous().to(torch.uint8)
                wq[ex] = qweight[:, 1::2] * 16 + qweight[:, ::2]
                ws[ex] = scales.T
            qc = int4_w4a16_moe_quant_config(w1_scale=w1_s, w2_scale=w2_s, block_shape=[0, group_size])
            cfg = VllmConfig()

            def run():
                with set_current_vllm_config(cfg):
                    return [fused_moe(a, w1_q, w2_q, score, topk_, renormalize=False, global_num_experts=e, quant_config=qc)]
            tag = "fp16" if dtype == torch.float16 else "bf16"
            kind = "cuda" if m / e <= 6 else "triton"
            ok &= run_twice(f"moe_wna16_{kind}_m{m}_e{e}_{tag}", run, repeats=5, extra={"m": m, "e": e, "n": n, "k": k, "dtype": tag, "kernel": kind})
    return ok


def marlin_atomic() -> bool:
    """vLLM's Marlin GEMM on random 4-bit weights with use_atomic_add off and on.

    Shapes satisfy n < 2048 and k >= 2048, the regime in which SGLang turns the
    atomic-add reduction on (sglang-0008) and vLLM would with
    VLLM_MARLIN_USE_ATOMIC_ADD=1 (vllm-0010). The kernel splits K across CTAs
    and, with use_atomic_add, adds each slice's partial into C with atomicAdd;
    whether the order matters depends on how many slices land on one tile.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    ok = True
    for (m, n, k) in ((16, 512, 3584), (16, 1024, 8192), (64, 512, 8192), (1, 512, 8192)):
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(0)
            a = torch.randn(m, k, dtype=dtype, device="cuda")
            w = torch.randn(k, n, dtype=dtype, device="cuda")
            w_ref, q_w, s_, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, 128, act_order=False)
            ws = marlin_make_workspace_new(a.device)
            # (use_atomic_add, use_fp32_reduce): SGLang's apply_gptq_marlin_linear passes both True.
            for atomic, fp32 in ((False, True), (True, False), (True, True)):
                def run(atomic=atomic, fp32=fp32):
                    return [ops.marlin_gemm(a, None, q_w, None, s_, None, None, None, g_idx, sort_idx, ws, scalar_types.uint4b8,
                                            m, n, k, is_k_full=True, use_atomic_add=atomic, use_fp32_reduce=fp32)]
                tag = "fp16" if dtype == torch.float16 else "bf16"
                name = f"marlin_gemm_atomic{atomic}{'_fp32reduce' if atomic and fp32 else ''}_m{m}_n{n}_k{k}_{tag}"
                ok &= run_twice(name, run, repeats=5, extra={"shape": [m, n, k], "dtype": tag, "use_atomic_add": atomic, "use_fp32_reduce": fp32})
    return ok


def deepgemm_bmk_bnk_mn() -> bool:
    import deep_gemm
    s, m, n, k = 64, 512, 512, 128
    a = torch.randn(s, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(s, n, k, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    def run():
        deep_gemm.einsum("bmk,bnk->mn", a, b, d)
        return [d.clone()]
    return run_twice("deepgemm_bmk_bnk_mn", run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["vllm_moe_wna16", "vllm_lora_shrink", "sglang_marlin", "sglang_fp8_blockwise", "sglang_lora_shrink", "deepgemm_bmk_bnk_mn", "marlin_atomic", "vllm_moe_wna16_kernel"])
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    fn = {"vllm_moe_wna16": vllm_moe_wna16, "vllm_lora_shrink": vllm_lora_shrink, "sglang_marlin": sglang_marlin,
          "sglang_fp8_blockwise": sglang_fp8_blockwise, "sglang_lora_shrink": sglang_lora_shrink, "deepgemm_bmk_bnk_mn": deepgemm_bmk_bnk_mn,
          "marlin_atomic": marlin_atomic, "vllm_moe_wna16_kernel": vllm_moe_wna16_kernel}[args.which]
    return 0 if fn() else 1


if __name__ == "__main__":
    sys.exit(main())
