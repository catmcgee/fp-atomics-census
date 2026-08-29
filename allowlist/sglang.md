# sglang: deterministic configuration allowlist

Repository `sgl-project/sglang` at `b41552334d46b9d83a6d834be361cb20b31d2137`.
Derived from `inventory/sglang.jsonl`. Forward pass, NVIDIA GPUs, default
server arguments unless a flag is named.

## Linear layers

| weight format | kernel at this sha | class-A sites | status | flags / conditions |
|---|---|---|---|---|
| bf16/fp16 unquantised | `F.linear` (cuBLASLt) | none in SGLang source | class C | `--enable-deterministic-inference` patches matmul through `batch_invariant_ops`; otherwise record `CUBLAS_WORKSPACE_CONFIG` |
| fp8 per-tensor / per-token | `torch._scaled_mm` (cuBLASLt) or sgl-kernel `fp8_scaled_mm` | none in SGLang source | class C | |
| fp8 blockwise, DeepGEMM backend (auto default on Hopper/Blackwell when DeepGEMM is installed, `fp8_utils.py:508-520`) | DeepGEMM | none (DeepGEMM allowlist) | allowed | `SGLANG_ENABLE_JIT_DEEPGEMM=1` (default) |
| fp8 blockwise, CUTLASS backend (auto when DeepGEMM is disabled or not installed on Hopper; `--fp8-gemm-runner-backend cutlass`) | sgl-kernel `fp8_blockwise_scaled_mm` | sglang-0009 (A): stream-K with `ReductionMode::Nondeterministic` whenever `k > 3n` | **forbidden** | no flag changes the mode; use DeepGEMM or Triton (`--fp8-gemm-runner-backend triton`) |
| fp8 blockwise, Triton backend | Triton | none | allowed | |
| fp8 blockwise, FlashInfer TRT-LLM backend (auto on Blackwell without DeepGEMM) | cubins | class C | probe | |
| GPTQ / AWQ / fp8 / fp4 through Marlin (default for convertible GPTQ checkpoints) | Marlin | sglang-0008 (A3) with the atomic-add write-out effectively **on** for `n < 2048` and `k >= 2048` on CUDA (`marlin_utils.py:463-485`) | **forbidden** for such shapes | there is no environment variable to disable it at this sha; shapes with `n >= 2048` or `k < 2048` use the serialised reduce (sglang-0010, A2) and are allowed |
| GPTQ with `--quantization gptq` forced, or non-convertible checkpoints | Exllama (`gptq_kernel.cu`) | sglang-0001..0005 (A) | **forbidden** | let the `gptq_marlin` override apply |

## Mixture of experts

| MoE path | class-A sites | status | flags / conditions |
|---|---|---|---|
| Triton fused MoE (default runner) | none; alignment tickets sglang-0012 B-indirect with order-invariant consumers | allowed | keep `--enable-fused-moe-sum-all-reduce` off (default): with it, sglang-0017 (A3) atomically sums experts |
| DeepGEMM MoE runner | none (DeepGEMM allowlist); EP scatter tickets B-indirect | allowed | |
| CUTLASS MoE (sgl-kernel) | data-prep tickets sglang-0013 B-indirect | allowed pending the CUTLASS grouped-GEMM check | |
| `--moe-runner-backend flashinfer_cutlass` | flashinfer-0001 (A3, `use_fused_finalize` default True; SGLang does not pass the flag) | **forbidden** as shipped | |
| `flashinfer_trtllm` (auto on sm100 for fp8 Llama-4 style models and modelopt_fp4 Gemma-4, `arg_groups/overrides.py:535-575`) | cubins | class C | probe |
| Marlin MoE | uses the same `should_use_atomic_add_reduce` helper through `marlin_utils_fp8/fp4` for the MoE Marlin paths | check per layer shape; the dense finding applies where `use_atomic_add` is computed | see sglang-0008 |
| DeepEP / Mooncake / NIXL dispatch | DeepEP allowlist | allowed | |

## Attention

| backend | class-A sites | status | notes |
|---|---|---|---|
| `fa3` (default on Hopper) | none | allowed | deterministic mode fixes `num_splits=1` for batch invariance |
| `flashinfer` | none in the FA2-template kernels | allowed | deterministic mode fixes split sizes |
| `triton` | none found | allowed | |
| `trtllm_mha` / `trtllm_mla` (default on sm100 for several model families) | cubins (sglang-0271) | class C | probe |
| DSA / NSA sparse attention with `--dsa-topk-backend sgl-kernel` or `flashinfer` | sglang-0016, flashinfer-0016 B-indirect, downstream unknown | **gate** | `SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1` fixes the FlashInfer collect order; the sparse attention consumer still needs the probe |
| vision attention (cuDNN through FlashInfer) | cubins | class C | |

## LoRA

| kernel | class-A sites | status | flags |
|---|---|---|---|
| MoE LoRA shrink (`_moe_lora_shrink_splitk_kernel`) | sglang-0018 (A) when the base grid is below 128 programs | **forbidden** for small batches | no flag |
| trtllm_lora_temp LoRA A shrink | sglang-0020 (A3) | **forbidden** by default | `SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=0` |
| LoRA B expand-add kernels | sglang-0021..0024 (A3, `STORE_WRITEBACK` default not read) | unknown | |
| Fused MoE LoRA kernel | sglang-0019 (A3, split factors not read) | unknown | |

## Sampling

| path | class-A sites | status | flags |
|---|---|---|---|
| FlashInfer sampling backend (default on CUDA) with top-k/top-p only | none (flashinfer-0018) | allowed | |
| FlashInfer backend with min-p (`top_k_renorm_prob` then `top_p_renorm_prob`) | flashinfer-0015 (A, multi-CTA `sum_topk` atomic for large vocabularies) and flashinfer-0014 (A3, `is_deterministic` default False) | **forbidden** as called | SGLang passes neither a deterministic flag nor a single-CTA guarantee |
| PyTorch sampling backend (forced by `--enable-deterministic-inference` or per-request seeds) | sglang-0026 (A, float `cumsum`, low confidence) | **gate** | runtime probe; `torch.use_deterministic_algorithms(True)` is not set by the server |
| Penalties | A1 stores only | allowed | |

## Parallelism

| mechanism | class-A sites | status | flags |
|---|---|---|---|
| custom all-reduce (1-stage or 2-stage) | none; fixed order (sglang-0014) | allowed | `--enable-deterministic-inference` selects the fixed-order variant explicitly |
| PyNccl / NCCL | class C | probe | record `NCCL_ALGO`, `NCCL_PROTO` |
| FlashInfer all-reduce fusion | disabled by deterministic mode; fixed order on this reading (flashinfer-0022) | allowed on this reading | |

## Summary for a verifier

Allowed with no flags: bf16 dense with fa3/flashinfer/triton attention, fp8
blockwise through DeepGEMM or Triton, Triton and DeepGEMM MoE, DeepEP,
custom all-reduce, FlashInfer sampling without min-p. Class C remains:
cuBLASLt for unquantised and per-tensor fp8 linears, NCCL, TRT-LLM cubins
on Blackwell.

Forbid or gate: Marlin layers with `n < 2048` and `k >= 2048` (the atomic
write-out is on by default at this sha), the CUTLASS fp8 blockwise backend
with `k > 3n`, Exllama, the FlashInfer CUTLASS MoE runner, fused-sum MoE,
LoRA shrink kernels for small batches, min-p sampling through FlashInfer,
the PyTorch sampler, and sparse attention top-k.

`--enable-deterministic-inference` addresses batch invariance (fixed
split sizes, batch-invariant matmul, fixed-order all-reduce, PyTorch
sampler). It does not touch the Marlin atomic write-out, the CUTLASS
stream-K mode, the LoRA split-K kernels or the FlashInfer renormalisation
atomics, and it routes sampling to a float `cumsum` that PyTorch documents
as non-deterministic.
