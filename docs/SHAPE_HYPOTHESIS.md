# Batch shape, not content: hypothesis and predictions

Written before any of the experiments in `probes/shape/` were run. The
predictions below are predictions; the results section of the README
records what happened.

## Hypothesis

After every order-dependent kernel is configured away or declined, the
remaining run-to-run variation in a stock deployment depends on the batch
**shape**, not on the batch **content**. Shape means the integers the
engine hands to its kernels at a forward pass:

- the padded batch size, which for a decode step is the CUDA-graph
  capture size the step was rounded up to;
- each sequence's query length and KV length at the step;
- each request's computed-token count, which absorbs prefix-cache hits
  and chunked prefill;
- the scheduled-token total;
- the parallelism configuration (TP, PP, EP, DP).

Two forward passes with the same shape vector and the same tokens for one
request should give that request the same bits, whatever the other
requests contain.

## Mechanisms that make this plausible, from source

The census pins vLLM at `5769a7382cb1`, FlashInfer at `c0d920d3270c` and
SGLang at `b41552334d46`; the wheels the probes run are vLLM 0.28.0,
FlashInfer 0.6.16 and SGLang 0.5.19. Line numbers are at the pinned shas.

- **cuBLASLt selects the kernel from (M, N, K).** The trace lines the
  cuBLASLt probe captured (`probes/results/*/cublaslt_sweep_7b_bf16.json`)
  show the heuristic changing algorithm, split-K count and reduction
  scheme with M for fixed N and K, for example split-K 2 with the
  workspace reduction at M of 1, 4 and 8 for N = K = 4096 and no split at
  M of 32 and above. M is the number of tokens in the pass. Every
  unquantised linear layer goes through this call
  (`vllm/model_executor/layers/quantization/utils/w8a8_utils.py`,
  inventory rows vllm-0180 and vllm-0181).
- **vLLM pads decode batches to captured CUDA-graph sizes.** The capture
  sizes are `cudagraph_capture_sizes` in
  `vllm/config/compilation.py:631`, rounded at `:1488-1514`; the
  dispatcher picks a graph from the token count and whether the batch is
  uniform decode in `vllm/v1/cudagraph_dispatcher.py:235-248`, reading the
  capture sizes at `:75`; the model runner pads to `num_tokens_padded` in
  `vllm/v1/worker/gpu_model_runner.py:2242-2262` and dispatches at `:2887`.
  A batch of 13 decode requests and a batch of 16 therefore run the same
  padded shape.
- **FlashInfer plans split-KV per request from the batch's KV lengths.**
  `include/flashinfer/attention/scheduler.cuh:150-191` decides `split_kv`
  and `max_num_pages_per_batch` from the batch size times the number of
  KV heads against the grid size, and from each request's page count in
  `kv_indptr`. vLLM's backend feeds those from the batch at
  `vllm/v1/attention/backends/flashinfer.py:253-271`. A request's
  attention reduction order therefore depends on its own KV length and
  on how many other requests share the pass, not on their tokens.
- **Marlin and LoRA split-K are shape conditions.** SGLang turns Marlin's
  atomic-add reduction on when `n < 2048` and `k >= 2048`
  (`python/sglang/srt/layers/quantization/marlin_utils.py:470-476`), and
  vLLM's LoRA shrink picks `split_k = 64` below a batch of 128 and 8 above
  (`vllm/lora/ops/triton_ops/utils.py:219-223`). Both are decided by
  integers, though the atomic paths themselves are then order-dependent
  and are excluded from this hypothesis by construction.
- **Prefix caching changes a request's computed-token count, not its
  neighbours'.** A new request arrives with `num_computed_tokens` set to
  its cache hit (`vllm/v1/core/sched/output.py:31-38`), and the model
  runner reads it per request
  (`vllm/v1/worker/gpu_model_runner.py:1422`). It is part of the shape
  vector.
- **What Cankaya lists for replay is a shape vector.** Section 6 of
  arXiv:2606.00279 names hardware SKU, exact weights, parallelism
  topology, software versions, and "batch size at each forward pass",
  with per-entry prefill or decode status and sequence lengths when the
  two are mixed. Nothing in that list is another request's content.

## Predicted exceptions, all of them content couplings

- **X1, MoE routing.** Per-expert token counts depend on which experts
  the neighbours' tokens select
  (`vllm/model_executor/layers/fused_moe/router/fused_moe_router.py:45`,
  then the alignment kernel of inventory row vllm-0016). Grouped GEMM
  shapes and the finalize order therefore depend on neighbours' content.
- **X2, per-tensor dynamic activation quantisation.** The scale is the
  maximum over the whole activation tensor of the pass:
  `csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:79-113`
  (`segmented_max_reduction_strided`, `atomicMaxFloat` on the shared
  scale). Per-token quantisation computes one scale per row
  (`common.cu:138`) and blockwise quantisation one per group
  (`vllm/model_executor/layers/quantization/utils/fp8_utils.py:566`);
  neither reads other rows. The choice is `use_per_token_if_dynamic` in
  `vllm/_custom_ops.py:1802-1827` and `activation_scheme` in
  `vllm/model_executor/layers/quantization/fp8.py:101-112`, with
  `block_quant` at `:298-301`.
- **X3, shared-prefix effects not captured by computed-token counts.**
  Cascade attention and similar prefix-sharing paths
  (`vllm/v1/attention/backends/flashinfer.py:591`, `use_cascade`) group
  requests by shared content, so two batches with equal counts can still
  run different kernels if their prefixes differ.

## Predictions

- **P1.** Two runs with identical shape vectors produce identical hashes,
  always. Every hash difference in the stock runs is explained by a shape
  difference. Mixtral's two outputs correspond to two shape vectors.
- **P2.** A logged mid-stream step, rebuilt and re-run in a fresh process
  with the same shape vector and the same tokens, reproduces the hash.
- **P3.** Dummy-neighbour replay: replace every neighbour's tokens with
  random tokens of identical lengths and identical computed-token counts,
  keep the target request unchanged. For dense models with per-token or
  no quantisation, the target request's hash is identical to the
  original.
- **P4.** P3 fails for MoE models, and the failure co-occurs with changed
  per-expert counts. P3 fails under per-tensor dynamic FP8. P3 holds
  under per-token or blockwise FP8.
- **P5.** Cross-SKU hashes (H100, B200, RTX PRO 6000) differ for the same
  configuration. The README must never imply cross-SKU identity.

## Experiments

E1 instrumentation, E2 bucket attribution, E3 logged replay, E4
dummy-neighbour replay and E5 cross-SKU are specified in
`probes/shape/RUNBOOK.md`, with one command per experiment and a results
directory argument. Verdicts are IDENTICAL or DIFFERS with the run count;
identical in twelve runs is evidence, not proof.
