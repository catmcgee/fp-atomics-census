# Runtime probes

Static reading settles most rows. It cannot settle three kinds of question:
whether a closed binary reduces in a fixed order, whether a library
operator that PyTorch documents as non-deterministic is non-deterministic
in the shapes an engine uses, and whether an order-dependent site that
the reading says is contended is contended often enough to show up. The
scripts in `probes/` are the experiments that would answer those
questions. None of them was run for this census. They are written against
the pinned shas and the public APIs of each engine, and each needs at
least one NVIDIA GPU; the collective probes need two.

## Protocol

Every probe follows the same rule (`probes/common.py`):

1. Fix every seed and every configuration knob, and record them.
2. Run the computation at least three times in one process, then once
   more in a fresh process.
3. Compare outputs bit for bit (`torch.equal` on the integer view of the
   float tensor). A tolerance would hide exactly the effect being looked
   for.
4. Write GPU SKU, driver, CUDA, PyTorch and library versions, the pinned
   sha and the verdict to a JSON file next to the script.

A probe that prints `bitwise-identical` on one machine says nothing about
another SKU, driver or library version. The verifier's question is
whether a named stack is deterministic, so a probe result is only useful
with the environment block attached. A `DIFFERS` verdict is conclusive for
that stack; an identical verdict after a handful of runs is evidence, not
proof, that no contended path was hit.

## Probes and the rows they settle

| Script | Rows | What it decides |
|---|---|---|
| `probe_engine_logits.py` | every default-path class-A and A3 row reachable from a model configuration | Runs vLLM or SGLang in-process on a fixed batch with greedy sampling and compares the top-5 logprobs of every position across runs. Run once with the configuration that reaches the row (the inventory's `path.entry_points`) and once with the allowlist configuration. This is the instrument for vllm-0014 (`moe_wna16`), vllm-0018/0019 (LoRA shrink with `VLLM_BATCH_INVARIANT` 0 and 1), vllm-0010 and sglang-0008 (Marlin with atomic add), sglang-0021 (LoRA shrink split-K), flashinfer-0001 (fused MoE finalize through either engine), and the whole class-C surface through a plain bf16 model. |
| `probe_cublaslt_algo.py` | vllm-0180, vllm-0181, sglang-0264 to sglang-0266, flashinfer-0040, flashinfer-0041, DeepGEMM-0009, flash-attention-0014 | Runs one matmul shape twice and compares bits; where the `cuda-python` bindings are present, asks `cublasLtMatmulAlgoGetHeuristic` for the top algorithm and prints its id, split-K count and `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME_ID`. A split-K algorithm with a reduction scheme other than `NONE` is the only way cuBLASLt reduces across CTAs; the probe records whether the heuristic chose one for the shape, workspace and `CUBLAS_WORKSPACE_CONFIG` in use. |
| `probe_torch_ops.py` | vllm-0045, vllm-0046, flash-attention-0026 (`index_add_` and `scatter_add_`), vllm-0049, vllm-0050, sglang-0031, sglang-0035, flash-attention-0027, flashinfer-0048 (float `cumsum`) | Runs each operator on contended indices with `torch.use_deterministic_algorithms` off and on. PyTorch documents `index_add_` and `scatter_add_` on CUDA as non-deterministic and provides a deterministic fallback; the probe checks that the documented behaviour holds for the sizes used by the engines and that the fallback is bit-stable. |
| `probe_flashinfer.py --which renorm` | flashinfer-0015, flashinfer-0016 | `top_p_renorm_probs` with `is_deterministic` False and True, and `top_k_renorm_probs` with a vocabulary large enough for the multi-CTA kernel. The multi-CTA renorm sums kept probabilities with a float `atomicAdd` and has no deterministic switch. |
| `probe_flashinfer.py --which topk` | flashinfer-0016, flashinfer-0017, flashinfer-0018, vllm-0038 to vllm-0044, sglang-0017 | `radix_topk` with `deterministic` False and True; compares the index order, not only the set. The B-indirect rows whose `downstream.order_invariant` is `unknown` are settled by whether the consumer of the indices (sampling, MoE routing) is order-sensitive; this probe shows whether the order changes at all. |
| `probe_flashinfer.py --which attention --backend trtllm` | flashinfer-0038, sglang-0267, vllm-0183 (TRT-LLM cubins) | Paged decode attention through the cubin backend, compared with the FA2 template on the same inputs. The cubins are closed; the probe is the only evidence available. |
| `probe_flashinfer.py --which moe` | flashinfer-0001, flashinfer-0002 | `cutlass_fused_moe` with `use_fused_finalize` True and False on a batch where several tokens share experts. The fused finalize uses `red.global.add` on bf16/f16 and is expected to differ under contention; the unfused path is expected to be identical. |
| `probe_kernels.py --which sglang_fp8_blockwise` | sglang-0011 | `fp8_blockwise_scaled_mm` with k > 3n, which selects the stream-K kernel built with `ReductionMode::Nondeterministic`. |
| `probe_kernels.py --which sglang_marlin` | sglang-0008 | Prints what `should_use_atomic_add_reduce` returns for a shape with n < 2048 and k >= 2048, and says how to reach the kernel through the engine probe. |
| `probe_kernels.py --which deepgemm_bmk_bnk_mn` | DeepGEMM-0001, DeepGEMM-0002 | The `bmk,bnk->mn` einsum, whose batch dimension is reduced with a float `atomicAdd` across CTAs. |
| `probe_collectives.py --which nccl` | vllm-0182, sglang-0269, flashinfer-0043, DeepEP-0012 | Two-rank all-reduce, compared per rank across runs, once with the defaults and once with `NCCL_ALGO=Tree NCCL_PROTO=Simple`. NCCL's reduction order is fixed for a given algorithm, protocol and channel count; the probe records whether the defaults pick the same one every time. |
| `probe_collectives.py --which flashinfer` | flashinfer-0023, flashinfer-0026 | `trtllm_allreduce_fusion` in plain all-reduce mode (expected identical, the lamport path is a fixed-order sum) and, where the driver exposes NVLS, the `multimem.ld_reduce` path (class C: the switch reduces in hardware and the order is not documented). |

## Probes that could not be written

- **cuDNN attention** (vllm-0184, sglang-0268, flashinfer-0039). cuDNN's frontend
  chooses an engine per shape; the attention engines on Hopper and
  Blackwell are closed. The engine logits probe with the cuDNN backend
  selected is the only instrument; there is no API to ask which engine
  ran.
- **`multimem.ld_reduce`** (flashinfer-0026). Needs NVLink
  SHARP hardware and a driver that exposes multicast objects; the
  collective probe has the hook but no verdict can be predicted.
- **TRT-LLM MoE cubins** (flashinfer-0038, and the consumer of the
  routing permutation in flashinfer-0020). The GEMM and finalize kernels
  are downloaded at runtime; the MoE probe can call them, but a reading
  of their reduction strategy is not possible.
- **`torch.compile` output**. vLLM compiles the model by default. The
  generated Triton kernels are not in any repository and depend on the
  Inductor version. The engine logits probe covers them with the
  compiled configuration; a separate probe of Inductor's reduction
  lowering (`config.triton.persistent_reductions`, split reductions with
  atomics) would need to be written against a specific PyTorch version.

## What a verifier should run

For a named stack, run the engine logits probe with the allowlist
configuration for that engine, and separately with each default-path
class-A row's configuration, and archive the JSON. Run the cuBLASLt probe
over the model's linear-layer shapes at the batch sizes the verifier
will sample. Run the NCCL probe at the deployment's tensor-parallel size.
The verifier's claim that the stack is bit-exact is only as strong as
the union of these results on the exact SKU and driver.
