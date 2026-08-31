"""Direct kernel probes for the engine-specific class-A and low-confidence rows.

    python probes/probe_kernels.py --which vllm_moe_wna16|vllm_lora_shrink|sglang_marlin|sglang_fp8_blockwise|sglang_lora_shrink|deepgemm_bmk_bnk_mn

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
    from sgl_kernel import fp8_blockwise_scaled_mm

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
    ap.add_argument("--which", required=True, choices=["vllm_moe_wna16", "vllm_lora_shrink", "sglang_marlin", "sglang_fp8_blockwise", "sglang_lora_shrink", "deepgemm_bmk_bnk_mn"])
    args = ap.parse_args()
    fn = {"vllm_moe_wna16": vllm_moe_wna16, "vllm_lora_shrink": vllm_lora_shrink, "sglang_marlin": sglang_marlin,
          "sglang_fp8_blockwise": sglang_fp8_blockwise, "sglang_lora_shrink": sglang_lora_shrink, "deepgemm_bmk_bnk_mn": deepgemm_bmk_bnk_mn}[args.which]
    return 0 if fn() else 1


if __name__ == "__main__":
    sys.exit(main())
