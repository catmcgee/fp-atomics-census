# Background reading

What each source claims about determinism in LLM inference, and what those
claims add to the list of code paths this census must check. Read before the
scan so that the seed list is grounded in what has already been observed rather
than in what the scanner happens to match.

## Cankaya 2026, "Bit-Exact AI Inference Verification Without Performance Tradeoffs"

arXiv:2606.00279. https://arxiv.org/abs/2606.00279

Section 2 separates two sources of apparent non-determinism. Element-wise
operations differ across hardware and libraries because special function units
and math libraries (libdevice against libm, version to version) approximate
transcendentals differently. Reductions differ because floating-point addition
is not associative and the summation tree is chosen partly in silicon (the
tensor core MMA reduction tree is fixed per generation) and partly in software.
Software-level reduction order is either static (dispatch heuristics keyed on
shape, dtype and library version, so fully reproducible given those inputs) or
atomic (partial results combined with `atomicAdd` on floating-point outputs,
order decided by the warp scheduler). The paper states that atomics were "the
sole source of genuine non-determinism we identified in our experiments, and
was limited to specific INT de-quantization kernels".

Section 3 reports experiments on unmodified vLLM (0.11.2 with PyTorch 2.9 and
CUDA 12.8) and HF Transformers (4.57.3 with flash-attn 2.8.3) with no
determinism flags set. Repeated runs were bitwise identical when the hardware
SKU, the software stack including attention backend, the quantisation format and
kernel variant, and the tensor-parallel rank were held fixed. Different physical
cards of the same SKU matched. Outputs were not invariant to batch size, with
sequence-length dependent "equivalence classes" of batch sizes that share a
kernel choice. Pipeline-parallel rank left outputs unchanged. The one genuine
run-to-run divergence was Qwen3-8B GPTQ with the Marlin kernel disabled, which
sends vLLM to the exllama kernel in `csrc/quantization/gptq/q_gemm.cu`
(`atomicAdd(out, result01)` at lines 319-320 of the version tested, "with other
occasions of atomics in that same kernel"). AWQ and GPTQ with Marlin were
deterministic. MoE models (Qwen3-30B-A3B, GLM-4.6, DeepSeek-V2-coder-lite)
under tensor parallelism at 120k-token contexts were bit-identical across runs.
Kimi-K2-Thinking was the sole non-deterministic model, traced to its INT4
de-quantisation.

Section 5 lists limitations relevant here. On Hopper, for sequence lengths under
about 250 tokens, cuBLAS can dispatch to the proprietary nvjet kernel family,
which was deterministic but cannot be emulated from source. The backward pass is
out of scope for the paper, but footnote 13 records `atomicAdd` on dQ at lines
964, 983 and 992 of `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` and on dK and dV
at lines 480 and 511 of `hopper/epilogue_bwd.hpp` in FlashAttention-3, gated by
a `Deterministic` template parameter that serialises accumulation with
`Barrier::arrive_inc` and `wait_eq`. It also notes that fused normalisation
backward kernels use atomics for weight gradients.

Section 6 lists what a prover must record for a verifier to reproduce outputs:
hardware SKU, exact weights in the deployed quantisation format, parallelism
topology (separately for prefill and decode), software versions including any
custom kernels, and the batch size at each forward pass (with per-entry
prefill/decode status and sequence lengths when the two are mixed).

What this adds to the seed list: the exllama kernel (confirm the atomics are
still there at the pinned sha and whether the path is still the default when
Marlin is unavailable); the FlashAttention-3 backward line numbers (re-locate
at the pinned sha and check what `Deterministic` defaults to at every call
site); the INT4 MoE de-quantisation path that Kimi-K2-Thinking would take in
vLLM; the nvjet family as a class-C entry under cuBLAS; fused normalisation
backward kernels (training only, but record them).

## DiFR, "Inference Verification Despite Nondeterminism"

arXiv:2511.20621. https://arxiv.org/abs/2511.20621

Section 3.1 states that bitwise reproducibility "is rarely achievable in
practice", attributing this to non-associativity combined with reduction orders
that depend on effective batch size and kernel strategy, and to variation in
CUDA version, GPU architecture and kernel implementation. For MoE models it adds
that routing and capacity constraints make a token's expert assignment depend on
the rest of the batch. It acknowledges the Thinking Machines result that
batch-invariant kernels remove this on fixed hardware, in which case "the
verifier can re-run the model once and check exact equality of logits or
tokens", but argues that statistical verification remains necessary across
heterogeneous deployments. The proposed tests are Token-DiFR (compare
token-level sampling outcomes reconstructed from a shared PRNG seed) and
Activation-DiFR (compare compressed internal activations), both tolerance based.
Section 7.4 notes speculative decoding was not evaluated.

What this adds: nothing new to scan, but it frames the stakes. A tolerance
scheme is what a verifier falls back to if bit-exactness cannot be demanded.
The census is what decides whether it can be.

## Thinking Machines, "Defeating Nondeterminism in LLM Inference"

Horace He and others, 10 September 2025.
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Distinguishes run-to-run non-determinism (same inputs, different outputs, caused
by concurrent atomic adds whose order depends on which core finishes first)
from lack of batch invariance (deterministic given the whole batch, but each
element's result changes with batch size because the kernel picks a different
tile or split-K configuration). The post's central empirical claim is that "in
the typical forward pass of an LLM, there is usually not a single atomic add
present", for two reasons: there is enough parallelism along the batch
dimension that the reduction dimension need not be split across cores, and
where it is split, libraries use a separate clean-up reduction or a semaphore
that fixes the accumulation order. The named exceptions are `scatter_add` in
PyTorch and FlashAttention backward. It notes that the common Triton
FlashAttention backward avoids atomics by recomputation at the cost of about
40% more FLOPs. The proposed fix for batch invariance is to replace RMSNorm,
matmul and attention with kernels whose reduction order does not depend on
batch size (fixed tile configuration, no split-K, fixed split-KV size in
attention), released as `batch_invariant_ops`, which overrides aten operators
through `torch.Library`.

What this adds: `scatter_add` and the wider family of PyTorch index/scatter
ops belong in the scanner's torch pattern set; the "semaphore strategy" is the
signature of sub-case A2 and should be recognised (Marlin's global reduce,
CUTLASS `Semaphore`, FlashAttention `Barrier`).

## vLLM batch invariance and reproducibility docs

`docs/features/batch_invariance.md` and `docs/usage/reproducibility.md` at the
pinned sha (`5769a73`).
https://github.com/vllm-project/vllm/blob/5769a7382cb111288a9c2927342ca908943b9793/docs/features/batch_invariance.md

vLLM "does not guarantee the reproducibility of the results by default". The
beta batch-invariance mode is enabled with `VLLM_BATCH_INVARIANT=1` (default
`0`, `vllm/envs.py:579`). `init_batch_invariance`
(`vllm/model_executor/layers/batch_invariant.py:974-983`) first sets
environment variables (`override_envs_for_invariance`, lines 953-971:
symmetric-memory all-reduce off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, a fixed
NCCL configuration with `allreduce:tree`, one channel and the `Simple`
protocol, AOT compile off), then installs `torch.Library` overrides
(`enable_batch_invariant_mode`, lines 897-940). On Ampere the overrides
replace `aten::mm`, `addmm`, `matmul` and `linear` with a persistent Triton
matmul (lines 910-913); on Hopper and Blackwell the code instead sets
`CUBLAS_WORKSPACE_CONFIG=:16:8` and `CUBLASLT_WORKSPACE_SIZE=1` on the stated
grounds that "the only source of batch variance is split-k, which we disable
via the cuBLAS workspace config" (lines 914-919). `_log_softmax`, `softmax`,
`mean.dim` and `bmm` are overridden on every CUDA device (lines 926-940), and
TF32 is disabled (lines 981-983). Elsewhere the flag disables the custom
all-reduce (`vllm/config/parallel.py:962-963`), NCCL symmetric-memory
all-reduce (`vllm/distributed/device_communicators/all_reduce_utils.py:115,144`,
`cuda_communicator.py:243-247`, `symm_mem.py:113`), cascade attention
(`vllm/config/vllm.py:1451-1458`), TensorRT-LLM attention kernels from
FlashInfer (`vllm/utils/flashinfer.py:382`), prefix caching for the FlashInfer
and Triton-MLA backends (`vllm/model_executor/layers/attention/attention.py:372-386`),
FlashAttention version 4 (`vllm/v1/attention/backends/fa_utils.py:226`) and
Marlin for AWQ, with the comment "Marlin kernels are not batch invariant"
(`vllm/model_executor/layers/quantization/auto_awq.py:309-311`; the AWQ path
then dequantises and calls `torch.matmul`, line 950). It pins the fused-MoE
Triton configuration (`vllm/model_executor/layers/fused_moe/fused_moe.py:1067,1247`),
requires MoE kernels to declare batch-invariance support
(`modular_kernel.py:578`), prefers the direct FP8 path (`quantization/fp8.py:454`)
and forces NVFP4 linear layers onto the CUTLASS kernel
(`vllm/model_executor/kernels/linear/__init__.py:889-905`). Offline,
`VLLM_ENABLE_V1_MULTIPROCESSING=0` makes scheduling deterministic instead.

What this adds: every file that branches on `VLLM_BATCH_INVARIANT` is a
place where the default path may differ from the deterministic path and
must be read. At this sha they are, beyond those already cited:
`vllm/lora/ops/triton_ops/utils.py`,
`vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`,
`vllm/model_executor/layers/fused_moe/router/{fused_topk_bias_router,grouped_topk_router}.py`,
`vllm/model_executor/layers/quantization/{humming,online/fp8}.py`,
`vllm/model_executor/layers/quantization/utils/humming_utils.py`,
`vllm/model_executor/kernels/linear/scaled_mm/{flashinfer,marlin}.py`,
`vllm/model_executor/layers/{layernorm,linear,vocab_parallel_embedding}.py`,
`vllm/model_executor/layers/attention/mla_attention.py`,
`vllm/v1/attention/ops/{int4_per_token_head,triton_unified_attention,triton_unified_attention_diffkv}.py`,
`vllm/v1/attention/backends/{flash_attn,flashinfer,flex_attention}.py`,
`vllm/v1/attention/backends/mla/{flashattn_mla,flashmla,triton_mla,prefill/flash_attn}.py`
and `vllm/config/attention.py`. The Marlin comment is about batch
invariance, not run-to-run determinism; Marlin's serialised reduce is class
A2 in this census and the comment must not be read as evidence of atomics.

## SGLang deterministic inference

`docs/advanced_features/deterministic_inference.md` at the pinned sha
(`b415523`), `python/sglang/srt/environ.py`, and the LMSYS post of 22
September 2025.
https://lmsys.org/blog/2025-09-22-sglang-deterministic/

`--enable-deterministic-inference` (default off,
`python/sglang/srt/server_args.py:2490-2493`; mirrored by
`SGLANG_ENABLE_DETERMINISTIC_INFERENCE`, `environ.py:661`) is supported only
with the FlashInfer, FlashAttention-3 and Triton attention backends
(`arg_groups/overrides.py:1406-1440` rejects anything else and picks a
default per architecture). FlashInfer is planned with fixed split sizes,
FA3 runs with one split, and Triton uses a fixed decode split size and an
aligned chunked-prefill truncation point
(`SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096`,
`SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048`,
`SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=4096`,
`SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=256`, `environ.py:668-671`). The flag
forces the PyTorch sampling backend (`overrides.py:1361-1367`), disables the
FlashInfer all-reduce fusion (`overrides.py:1393-1403`,
`server_args.py:3744-3745`), selects a fixed-order all-reduce
(`distributed/device_communicators/custom_all_reduce.py:380,421`,
`distributed/parallel_state.py:1010`; `SGLANG_USE_1STAGE_ALLREDUCE`,
`environ.py:666`, is the AMD equivalent), pins the fused-MoE Triton
configuration (`layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py:72,173`)
and changes the MoE router (`layers/moe/router.py:347-368`). The FlashInfer
DSA top-k kernel has its own determinism knobs
(`SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC`,
`SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK`, `environ.py:628-629`). The blog post
states that TP1 and TP2 were deterministic while larger TP "require
modifications to reduce kernels", that MoE support was future work at the
time (the doc at this sha gives a Qwen3-30B-A3B example), and that the
companion training setup used FlashAttention-2 rather than 3 "to enable
deterministic backward passes", `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`NCCL_ALGO=Ring` and `torch.use_deterministic_algorithms(True)`.

What this adds: every SGLang file that branches on the flag is on the read
list: `layers/attention/{flashinfer,flashattention,triton,dsa}_backend.py`,
`layers/sampler.py`, `sampling/sampling_batch_info.py`,
`managers/scheduler.py`, `model_executor/model_runner.py`,
`models/deepseek_v2.py`, `models/deepseek_common/attention_backend_handler.py`,
`arg_groups/speculative_hook.py` and the files cited above. The TensorRT-LLM
MoE finalize kernel that SGLang reaches through FlashInfer
(`layers/moe/moe_runner/flashinfer_trtllm.py:76-79`,
`sglang.jit_kernel.moe_finalize_fuse_shared`) is a named candidate for an
atomic combine and belongs to the FlashInfer and SGLang scans.

## FlashAttention deterministic mode

README at the pinned sha (`1f7ce2f`) and `flash_attn/flash_attn_interface.py`.
https://github.com/Dao-AILab/flash-attention/blob/1f7ce2f7cb503473559f3d44d575ae05b1ed8557/README.md

The public API takes `deterministic: bool` with the documented meaning "Whether
to use the deterministic implementation of the backward pass, which is slightly
slower and uses more memory. The forward pass is always deterministic." The
deterministic backward was added in release 2.4. FlashAttention-3 (the `hopper/`
tree) carries the same flag through to a `Deterministic` template parameter, as
noted by Cankaya. The Triton AMD backend documents a default kernel
configuration "optimized for determinism" with autotuning opt-in.

What this adds: check the forward claim by reading, not by trusting the README;
find the default value of `deterministic` at each Python entry point and at each
engine's call into flash-attention; find every `atomicAdd` in `csrc/flash_attn`
and `hopper/` and map each to the flag.

## Thinking Machines `batch_invariant_ops`

https://github.com/thinking-machines-lab/batch_invariant_ops

A `torch.Library` based override of aten matmul, softmax and mean with
batch-invariant Triton kernels. Both vLLM and SGLang derive their deterministic
modes from it. Not scanned here (it is a dependency, not an engine), but its
approach explains why the engines' deterministic modes work by replacing kernels
rather than by adding flags to existing ones.

## Consolidated additions to the seed list

Beyond the seed list in the task description, the reading adds:

1. FlashAttention-3 backward: `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` (dQ)
   and `hopper/epilogue_bwd.hpp` (dK, dV), with the `Deterministic` template
   parameter and its default at every call site.
2. FlashAttention `csrc/layer_norm` and any other fused-normalisation backward
   kernels with weight-gradient atomics (training only, recorded for
   completeness).
3. vLLM INT4 MoE de-quantisation paths (what Kimi-K2-Thinking would use).
4. Every vLLM file that branches on `VLLM_BATCH_INVARIANT` (list above).
5. Every SGLang file that branches on `enable_deterministic_inference` (list
   above).
6. The MoE finalize (combine) kernels reached through FlashInfer's
   TensorRT-LLM and CUTLASS backends, and SGLang's `moe_finalize_fuse_shared`
   JIT kernel.
7. FlashInfer all-reduce and all-reduce fusion kernels ("does not provide a
   fixed reduction order" per vLLM).
8. FlashInfer DSA top-k kernels and their determinism/tie-break knobs.
9. cuBLAS nvjet kernel family on Hopper as a class-C entry.
10. The PyTorch operator family that `torch.use_deterministic_algorithms(True)`
    rejects, as scanner patterns over engine Python code.
