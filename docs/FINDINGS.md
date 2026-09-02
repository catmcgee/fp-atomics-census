# Findings

## The premise, and where it holds

Cankaya (2026) argues that once an inference configuration is recorded,
the only run-to-run non-determinism left in the engines comes from
floating-point atomic accumulation, and that such atomics are rare
enough to be configured away. He found one kernel, the exllama GPTQ path, and pointed at
FlashAttention-3's backward pass. This census read
every atomic site in eight repositories at their 6 July 2026 commits
to test that claim.

The table below counts sites by class (see `docs/TAXONOMY.md`).

| Engine | Sites | A | A1 | A2 | A3 | B | B-indirect | C |
|---|---|---|---|---|---|---|---|---|
| vLLM | 187 | 21 | 20 | 2 | 9 | 118 | 10 | 7 |
| SGLang | 271 | 12 | 28 | 1 | 8 | 209 | 4 | 9 |
| FlashInfer | 85 | 17 | 5 | 2 | 3 | 46 | 5 | 7 |
| FlashAttention | 27 | 12 | 0 | 0 | 10 | 4 | 0 | 1 |
| Marlin | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| DeepGEMM | 9 | 2 | 4 | 0 | 0 | 1 | 1 | 1 |
| DeepEP | 12 | 0 | 1 | 0 | 0 | 8 | 2 | 1 |
| CUTLASS | 6 | 0 | 0 | 2 | 2 | 1 | 1 | 0 |
| Total | 599 | 64 | 58 | 8 | 32 | 388 | 23 | 26 |

Two thirds of the sites are integer atomics that are exact regardless
of order: counters, tickets, locks, flags, histogram bins.
The premise is right that float atomics are the minority. It is right
about the dense forward path too: attention forward in every backend
that ships source (FA2, FA3, FA4, FlashInfer's own templates, the
Triton backends) has no order-dependent atomic; unquantised linear
layers go to cuBLASLt; the Triton fused MoE, the Marlin MoE, DeepGEMM's
grouped GEMMs and DeepEP's dispatch and combine have none on their
default paths; CUTLASS stream-K defaults to its deterministic reduction.
For a bf16 dense model or a common FP8 or GPTQ dense model served by
vLLM with the stock configuration, the census found no order-dependent
atomic in engine source.

The premise fails as a statement about the ecosystem. There are 26
plain class-A sites on default paths, and six more sites where a
configuration flag defaults to the atomic side. They cluster in five
places: adapters, MoE finalization, quantised kernels with a split-K
or atomic-add fast path, sampling renormalisation, and PyTorch library
operators used by model code.

## Every default-path class-A site

Eleven of the 26 are the backward pass of FlashAttention (FA2 sm80, FA3
hopper, FA4 sm80, sm100 and the MLA sm100 kernels, and the Triton
backward). They accumulate dQ, or dK and dV, with float `atomicAdd`
across query or key blocks. They do not run at inference. They are listed for
completeness.

The forward-path sites are:

- **vLLM `moe_wna16`** (vllm-0014, `csrc/libtorch_stable/moe/moe_wna16.cu:213`).
  The CUDA kernel for 4-bit GPTQ and AWQ MoE accumulates each token's
  expert output with `atomicAdd` on `half` or `bfloat16`, and the kernel
  is selected for small batches (`fused_moe.py:1227-1235`). High
  confidence.
- **vLLM ROCm RDNA3 GPTQ** (vllm-0032 to vllm-0037, `csrc/rocm/q_gemm_rdna3*.cu`).
  Six compare-and-swap loops adding packed `half2` and `bfloat162`,
  the exllama pattern Cankaya found, rewritten for gfx11. Medium
  confidence only because the ROCm build was not exercised.
- **vLLM mean pooling** (vllm-0045, `pooler/seqwise/methods.py:97`) and
  **Moondream3 expert combine** (vllm-0046, `models/moondream3.py:555`)
  use `index_add_`, which PyTorch documents as non-deterministic on
  CUDA. Medium confidence: the documented behaviour, not a read of the
  ATen kernel.
- **vLLM diffusion Gemma** (vllm-0050, `models/diffusion_gemma.py:555`)
  and **SGLang Gemma3n audio** (sglang-0035, `models/gemma3n_audio.py:89`)
  take a float `cumsum` on CUDA. Low confidence: PyTorch's
  documentation has listed `cumsum` as non-deterministic in some
  versions and the scan implementation is version-dependent. The probe
  in `probes/probe_torch_ops.py` settles it for a given build.
- **SGLang FP8 blockwise stream-K** (sglang-0011,
  `fp8_blockwise_gemm_sm90_dispatch.cuh:166`). When `k > 3n` the
  dispatcher selects a CUTLASS stream-K kernel built with
  `ReductionMode::Nondeterministic`. High confidence. This is the
  default FP8 path on Hopper for the projection shapes that satisfy the
  condition.
- **SGLang MoE LoRA shrink** (sglang-0019, `lora/triton_ops/virtual_experts.py:240`).
  The split-K shrink kernel finishes with `tl.atomic_add` on float.
  High confidence, reached whenever a LoRA adapter is active on an MoE
  model.
- **FlashInfer `copy_red_global`** (flashinfer-0002,
  `cutlass_extensions/arch/copy_red_global.hpp:84`). The
  `red.global.add.noftz.f16x2` primitive behind the fused MoE finalize
  epilogue. It is only reached through the gate below.
- **FlashInfer multi-CTA top-k renormalisation** (flashinfer-0016,
  `include/flashinfer/topk.cuh:1794`). The sum of kept probabilities is
  a float `atomicAdd` across CTAs and there is no deterministic switch.
  High confidence, reached for vocabularies too large for one CTA.

## Gates that default to the atomic side

- **vLLM LoRA shrink** (vllm-0018, vllm-0019). The Triton config sets
  `split_k` to 64 or 8 unless `VLLM_BATCH_INVARIANT=1`, and the
  split-K partials are combined with `tl.atomic_add`. LoRA serving in
  vLLM is order-dependent by default.
- **SGLang Marlin** (sglang-0008). `should_use_atomic_add_reduce`
  contains an `if not True:` stub (`marlin_utils.py:476`), so atomic-add
  reduction is on for every `n < 2048, k >= 2048` shape on CUDA. vLLM's
  copy of the same kernel is gated by `VLLM_MARLIN_USE_ATOMIC_ADD`,
  default off.
- **SGLang LoRA shrink split-K** (sglang-0021).
  `SGLANG_ENABLE_LORA_SHRINK_SPLIT_K` defaults to true.
- **FlashInfer fused MoE finalize** (flashinfer-0001). `use_fused_finalize`
  defaults to true and the docstring says the result is non-deterministic.
  Both vLLM and SGLang reach it through their FlashInfer CUTLASS MoE
  backends.
- **FlashInfer top-p renormalisation** (flashinfer-0015). `is_deterministic`
  defaults to false.
- **FlashAttention backward** (ten A3 rows). `deterministic` defaults to
  false; training only.

Gates on the safe side by default: vLLM Marlin, vLLM fused MoE LoRA,
CUTLASS 3.x stream-K, Machete and vLLM's CUTLASS 2.x `scaled_mm`.

## B-indirect sites

Twenty-three integer atomics decide a slot or an order that float data
later follows. Ten are order-invariant on reading: the MoE alignment
and token-sort kernels in vLLM and SGLang (vllm-0016, vllm-0026,
sglang-0013, sglang-0014, sglang-0028) either sort after the atomic
pass or feed GEMMs whose rows are independent; DeepEP's dispatch
counters (DeepEP-0001, DeepEP-0006) and FlashInfer's all-to-all
(flashinfer-0021) place whole tokens, and the combine sums experts in
a fixed order; CUTLASS's stream-K visitor store (cutlass-0006) is a
turnstile. One is not: the CuTe DSL dynamic MoE kernel on sm12x
(flashinfer-0010) finishes with a bf16x2 scatter-add, so each token's
expert sum already follows arrival order.

Twelve are unknown, and they share one shape: top-k selection kernels
that write selected indices to slots obtained by atomic ticket
(vllm-0038 to vllm-0044, sglang-0017, flashinfer-0017, flashinfer-0018,
flashinfer-0020, DeepGEMM-0007). The set of indices is fixed; their
order is arrival order. Whether that matters depends on the consumer.
For sampling it does not, unless ties are broken by position. For
DeepSeek sparse attention it might: the sparse attention kernel
gathers keys in list order, and a softmax reduced in a different key
order rounds differently. Those kernels (FlashMLA sparse, the NSA
backend, the TRT-LLM routing cubins) were not read, so the rows stay
`unknown` and `probes/probe_flashinfer.py --which topk` is the
instrument.

## The opaque surface

Twenty-six sites call into binaries: cuBLASLt (10, every unquantised
linear layer in vLLM and SGLang, SGLang's FP8 `scaled_mm` fallback,
FlashInfer `bmm_fp8`, DeepGEMM's fallback), cuBLAS (3), cuDNN (3,
vision-encoder attention), NCCL (4), the TensorRT-LLM cubins that
FlashInfer downloads (3: attention and MoE in FlashInfer, the SGLang
and vLLM backends that select them), FlashMLA sparse (vllm-0185) and
NVLink multicast (flashinfer-0026). cuBLASLt's heuristic may pick a
split-K algorithm with a non-`NONE` reduction scheme for skinny
shapes; nothing in engine source pins it beyond
`CUBLAS_WORKSPACE_CONFIG`. These are the sites that no reading can
settle and that a verifier has to probe on the exact SKU, driver and
library versions (`docs/RUNTIME_PROBES.md`).

## What a verifier needs

1. A configuration record that includes the environment flags listed
   in the allowlists, not only the model and sampling parameters.
2. A forbid list: LoRA (both engines) unless `VLLM_BATCH_INVARIANT=1`
   and the SGLang split-K flag is off; `moe_wna16`; FlashInfer's CUTLASS
   MoE backend with fused finalize; SGLang's Marlin path for shapes
   with `n < 2048, k >= 2048` until the stub is fixed; SGLang FP8
   blockwise GEMM for shapes with `k > 3n`; the top-p and multi-CTA
   top-k renormalisers; `index_add_`-based pooling; the ROCm RDNA3
   kernels.
3. A version pin and a probe result for cuBLASLt, cuDNN, NCCL and the
   cubins, because the allowlists cannot cover them.
4. Awareness that run-to-run determinism is not batch invariance.
   Batch invariance is what lets a verifier re-run a single request out
   of a batch; the engines' batch-invariant modes cost performance and
   are off by default.

## Nulls and what could not be told

Marlin, DeepEP and CUTLASS have no plain class-A site. DeepGEMM has
two, both on the `bmk,bnk->mn` einsum, not on the engine paths.
FlashAttention has none in forward. Directories that were not scanned
are listed in `docs/METHOD.md`; the main gaps are generated code
(`torch.compile`, JIT variants) and the ROCm and XPU trees of SGLang.
Nineteen rows carry low confidence, most of them PyTorch operators
whose determinism depends on the build, and the B-indirect unknowns
above. Every default-path class-A row was read twice.
