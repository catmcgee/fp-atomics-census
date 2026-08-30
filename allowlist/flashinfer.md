# flashinfer: deterministic configuration allowlist

Repository `flashinfer-ai/flashinfer` at `c0d920d3270ce28e0d7db8cc2163a76245e63712`.
Derived from `inventory/flashinfer.jsonl`. FlashInfer is a kernel library;
the table is organised by API family, and the engine allowlists say which
engine configuration reaches each family.

## Attention

| API family | source | class-A sites | status |
|---|---|---|---|
| `BatchPrefillWithPagedKVCacheWrapper`, `BatchDecodeWithPagedKVCacheWrapper`, single-request prefill/decode (FA2-template kernels in `include/flashinfer/attention/`) | open | none; split-KV partials merged by `cascade.cuh` kernels in fixed order | allowed |
| POD attention (`pod.cuh`, `batch_pod.cuh`) | open | none; scheduler counters flashinfer-0028 are class B | allowed |
| Sparse MLA for sm120 (`sparse_mla_sm120/`) | open | none; float `atomicMax` only (flashinfer-0029) | allowed |
| fmha_v2 (TensorRT-LLM sm80-sm90 kernels compiled from source under `csrc/fmha_v2`) | open | multi-CTA long-sequence path is A2 (flashinfer-0024): partial O accumulated under a turnstile lock in CTA order | allowed |
| XQA decode (`csrc/xqa`) | open | none; last-CTA merge in fixed index order (flashinfer-0027) | allowed |
| TRT-LLM-gen FMHA (`trtllm_batch_context_with_kv_cache`, `trtllm_batch_decode_with_kv_cache`) | cubin | unknown (flashinfer-0031) | class C, probe |
| cuDNN SDPA (`cudnn_batch_prefill_with_kv_cache`) | cubin | unknown (flashinfer-0032) | class C, probe |
| CuTe DSL FMHA (`DSL_FMHA` artifact) | cubin | unknown | class C, probe |

## GEMM

| API family | class-A sites | status |
|---|---|---|
| `bmm_fp8`, `mm_bf16` (cuBLASLt) | class C (flashinfer-0033, 0034) | probe; record workspace size and `CUBLAS_WORKSPACE_CONFIG` |
| FP4 GEMM sm120 (`fp4_gemm_template_sm120.h`, CUTLASS stream-K with the default `ReductionMode::Deterministic`) | none in FlashInfer; CUTLASS fixup is A2 (cutlass-0001) | allowed |
| Dense block-scaled GEMM sm120 (CuTe DSL) | flashinfer-0010 (A3, `split_k_atomic_bf16` default False) | allowed with the default |
| Masked grouped GEMM Blackwell (CuTe DSL) | flashinfer-0011 (A, single-writer status not established) | **gate** until the probe or a second read settles the writer count |
| DeepGEMM through FlashInfer (`flashinfer/deep_gemm.py`) | cubins (DEEPGEMM artifact) | class C |
| TRT-LLM-gen GEMM / batched GEMM | cubins | class C |

## Mixture of experts

| API family | class-A sites | status | flags |
|---|---|---|---|
| `cutlass_fused_moe` (SM90/SM100 CUTLASS backend) | flashinfer-0001 (A3): the fused finalize epilogue accumulates a token's experts with `red.global.add` | **forbidden** with the default | pass `use_fused_finalize=False` (deterministic finalize kernel); vLLM and SGLang do not expose this at their pinned shas |
| `trtllm_fp8_block_scale_moe`, `trtllm_fp4_block_scale_moe` (TRT-LLM-gen) | routing tickets flashinfer-0020 B-indirect; GEMMs and finalize are cubins | class C, probe | |
| CuTe DSL finalize-fusion grouped GEMM (`blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4`) | flashinfer-0004 (A) | **forbidden** | no flag |
| CuTe DSL sm12x MoE kernels | flashinfer-0006..0008 (A) | **forbidden** | no flag |
| MoE all-to-all (`trtllm_alltoall`, `moeAlltoAll`) | counters only (flashinfer-0021, 0022); slot tickets B-indirect with order-invariant consumers | allowed | |
| bgmv MoE LoRA | flashinfer-0030 (A) | no public API at this sha | |

## Sampling and top-k

| API | class-A sites | status | flags |
|---|---|---|---|
| `sampling_from_probs`, `top_k_sampling_from_probs`, `top_p_sampling_from_probs`, `top_k_top_p_sampling_from_probs`, `min_p_sampling_from_probs` | none (flashinfer-0018: `atomicMin` tie-break only) | allowed | `deterministic` defaults to True (Belloch scan); with False cub's block scan is used, which is also fixed for a given configuration |
| `top_p_renorm_probs` with `vocab_size >= NUM_BUCKETS` | flashinfer-0014 (A3) | **forbidden** with the default | `is_deterministic=True` (default False) |
| `top_k_renorm_probs` | flashinfer-0015 (A) when a row spans several CTAs | **forbidden** for large vocabularies | no flag; rows that fit one CTA are clean |
| `radix_topk`, `top_k_page_table_transform` | flashinfer-0016 B-indirect | gate on the consumer | `deterministic=True` or a `tie_break` mode fixes the output order |
| `fast_topk_cuda_v4` (cluster exact top-k) | flashinfer-0017 B-indirect | gate on the consumer | |

## Communication

| API | class-A sites | status |
|---|---|---|
| `trtllm_allreduce_fusion` (one-shot Lamport and two-shot) | none; sums ranks in fixed order (flashinfer-0022) | allowed on this reading |
| MNNVL all-reduce | none; fixed order (flashinfer-0023) | allowed on this reading |
| MoE all-reduce fusion | counter only; reduction loop not read | unknown |
| `mixed_comm` (NVLS multicast `multimem.ld_reduce`, NVSHMEM) | class C (flashinfer-0025) | probe |
| GEMM + all-reduce (CuTe DSL two-shot; TRT-LLM epilogue) | flashinfer-0003 (A, dead code), flashinfer-0012 locks only | not reachable at this sha |
| all-gather + matmul overlap | locks only (flashinfer-0036) | reduction order not read |

## Normalisation

| API | class-A sites | status |
|---|---|---|
| RMSNorm, fused add RMSNorm, layer-norm + SiLU | none; amax via float `atomicMax` (flashinfer-0029) | allowed |

## Summary for a verifier

The open-source attention kernels in FlashInfer are free of order-dependent
atomics, including split-KV, POD and the fmha_v2 multi-CTA path. The
problems are in MoE finalize (CUTLASS fused epilogue on by default, CuTe DSL
finalize fusion, sm12x kernels), in the renormalisation kernels used for
min-p sampling, and in the pre-compiled TRT-LLM, cuDNN and DeepGEMM cubins,
which cannot be read. A verifier accepting FlashInfer should require
`use_fused_finalize=False`, `is_deterministic=True` for top-p
renormalisation, single-CTA top-k renormalisation, and should treat every
cubin-backed call as needing the runtime probe.
