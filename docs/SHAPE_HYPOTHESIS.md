# Batch shape and content: hypothesis and predictions

These are the research predictions retained from the experiment plan, with scope corrections. They are not established properties. The README distinguishes measured outcomes from untested predictions; repository commit dates alone do not establish preregistration.

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
- the parallelism configuration (TP, PP, EP, DP);
- batch row order, target placement and the actual attention/GEMM plan.

The current key contains only recorded planner state. It is an approximation, not a sufficient execution-state specification.

Two forward passes with the same shape vector and the same tokens for one
request should give that request the same bits, whatever the other
requests contain.

## Mechanisms that make this plausible, from source

The census pins vLLM at `98dff2a81d74`, FlashInfer at `8bc3b5780277` (the
commit its v0.6.18.post1 tag points to) and SGLang at `b41552334d46`; the
wheels the probes below run are vLLM 0.28.0, FlashInfer 0.6.16 and SGLang
0.5.19, so the FlashInfer wheel is older than the pinned release. Line
numbers are at the pinned shas.

- **cuBLASLt algorithm choice can depend on (M, N, K).** The trace lines the
  cuBLASLt probe captured (`probes/results/*/cublaslt_sweep_7b_bf16.json`)
  from separate diagnostic processes show the heuristic changing algorithm, split-K count and reduction
  scheme with M for fixed N and K, for example split-K 2 with the
  workspace reduction at M of 1, 4 and 8 for N = K = 4096 and no split at
  M of 32 and above. M is the number of tokens in the pass. Ordinary dense linear layers can use these GEMM paths
  (`vllm/model_executor/layers/quantization/utils/w8a8_utils.py`,
  inventory row vllm-0180; vllm-0181 is the different FP8 scaled_mm path).
- **vLLM pads decode batches to captured CUDA-graph sizes.** The capture
  sizes are `cudagraph_capture_sizes` in
  `vllm/config/compilation.py:648`, rounded at `:1544-1570`; the
  dispatcher picks a graph from the token count and whether the batch is
  uniform decode in `vllm/v1/cudagraph_dispatcher.py:235-248`, reading the
  capture sizes at `:75`; the model runner pads to `num_tokens_padded` in
  `vllm/v1/worker/gpu_model_runner.py:2323-2344` and dispatches at `:2977`.
  A batch of 13 decode requests and a batch of 16 may use the same padded
  token count under a capture configuration containing that bucket. Equal
  padded counts alone do not imply equal attention plans or target placement.
- **FlashInfer plans split-KV per request from the batch's KV lengths.**
  `include/flashinfer/attention/scheduler.cuh:150-191` decides `split_kv`
  and `max_num_pages_per_batch` from the batch size times the number of
  KV heads against the grid size, and from each request's page count in
  `kv_indptr`. vLLM's backend feeds those from the batch at
  `vllm/v1/attention/backends/flashinfer.py:316-334`. A request's
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
  its cache hit (`vllm/v1/core/sched/output.py:36-43`), and the model
  runner reads it per request
  (`vllm/v1/worker/gpu_model_runner.py:1482`). It is part of the shape
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
  (`vllm/model_executor/layers/quantization/utils/fp8_utils.py:534`);
  neither reads other rows. The choice is `use_per_token_if_dynamic` in
  `vllm/_custom_ops.py:1905-1930` and `activation_scheme` in
  `vllm/model_executor/layers/quantization/fp8.py:97-108`, with
  `block_quant` at `:268-271`.
- **X3, shared-prefix effects not captured by computed-token counts.**
  Cascade attention and similar prefix-sharing paths
  (`vllm/v1/attention/backends/flashinfer.py:661`, `use_cascade`) group
  requests by shared content, so two batches with equal counts can still
  run different kernels if their prefixes differ.

## Predictions

- **P1.** Repeated observations with identical recorded target history, target inputs and
  environment should produce identical hashes in the tested scope. This is tested only for repeated history groups. Singleton groups have no
  comparison power, and a matching key does not establish complete planner state.
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

E1 instrumentation, E2 bucket attribution, E3 scripted repeatability, E4
dummy-neighbour comparisons, E5 recorded-stack comparison and E6 recorded-schedule
reconstruction are specified in
`probes/shape/RUNBOOK.md`, with one command per experiment and a results
directory argument. Comparisons can also be INVALID, NOT COMPARABLE or NOT TESTED. E3 does not test P2 because it neither reconstructs a recorded schedule nor teacher-forces a continuation. E6 is the P2 experiment: it records admissions and per-pass token ids, replays the recorded schedule in a fresh process with teacher forcing and compares every pass. Its records and same-stack replays so far come from one H100 software stack with vLLM 0.28.0 (in the third campaign one installed package, openai, differs), in three campaigns: Qwen2.5-7B-Instruct in a plain arm and a chunked-prefill arm; then Qwen3-8B-FP8 (blockwise FP8, compiled), Llama 3.1 8B Instruct (compiled) and Qwen1.5-MoE-A2.7B-Chat (uncompiled) in a plain arm each; then Qwen2.5-7B-Instruct at TP=2 in both arms, Qwen1.5-MoE-A2.7B-Chat uncompiled with CUDA graphs, Qwen3-8B-FP8 in a chunked-prefill arm, and Qwen2.5-7B-Instruct with online FP8 at per-token and at forced per-tensor dynamic activation scales. All eleven of those replays were IDENTICAL in every row of every pass, each run once in a new process on the same pod with the compiled artefacts its record had left on disk. Replayed again with every compile and kernel cache cleared, on a different pod with different GPUs of the same model and driver, and for the two Qwen2.5-7B TP=1 arms in a different datacentre and so certainly on a different machine, six of the eleven stayed IDENTICAL and five did not: the four compiled FP8 arms and the two uncompiled arms held, and every compiled bf16 arm differs from pass 0, slot 0, in 493 to 555 of its rows, with every comparison requirement met and every token held by teacher forcing. P2 therefore has supporting evidence, for these recorded schedules on this stack with vLLM 0.28.0, that is conditional on the compiled artefacts: it holds warm for all eleven, and cold for the FP8 and uncompiled arms only. It has none for any other schedule, stack or parallel configuration. Two further bounds come from the same campaign. Two cold replays of one record on one machine differed in 5 of 512 decode rows, so bit-identity here is an observation about the runs made rather than a demonstrated property. At TP=2 the comparison reads rank 0, and one cold replay has 4 of 555 decode rows where the two ranks' hidden-state hashes differ while their full-logit hashes and argmax agree, so a TP above 1 verdict describes rank 0. Why the cold split falls where it does is not established: a benchmark-chosen reduction configuration differing between the record's compile and a fresh one fits every arm, and so does FP8 activation quantisation absorbing a small upstream difference, and the records do not separate them. The forced per-tensor arm shows that a whole-batch replay reproduced bits that depend on every row of the pass through one activation scale. X2 predicts that such bits depend on the neighbours' content, and E6 re-supplies that content, so the result is consistent with X2 and is not a test of it; its per-tensor operation is inferred from the records, not recorded. A boundary rebuild of one decode pass from a single prefill of the recorded prefixes was DIFFERS with the argmax unchanged; the RUNBOOK explains why that mode cannot attribute a difference. E5 does not isolate GPU from driver or input divergence. A teacher-forced E6 replay of the H100 plain record on an RTX PRO 6000 met every comparison requirement and differed in every row; it is consistent with P5 but does not isolate the GPU, because the FlashAttention version, the driver, the container image and the compiled artefacts also differed.

The compiled MoE E4 arms are excluded from P4 evidence: a hook-free check on the same stack showed that torch.compile alone, not CUDA graphs, the hook or the compile cache, turns Qwen1.5-MoE-A2.7B-Chat's output degenerate on vLLM 0.28.0, so their DIFFERS verdicts describe a mis-compiled model, while the uncompiled MoE arms, which are not degenerate, were IDENTICAL under neighbour replacement on that one stack.
Identical in a few repeats is evidence only for those observations.
