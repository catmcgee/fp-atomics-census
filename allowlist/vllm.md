# vllm: deterministic configuration allowlist

Repository `vllm-project/vllm` at `5769a7382cb111288a9c2927342ca908943b9793`.
Derived from `inventory/vllm.jsonl`. Forward pass, NVIDIA GPUs, default
engine flags unless a flag is named. A configuration is allowed when every
class-A site it can reach is A1, A2 or A3 with the gate on the
deterministic side. Class-C entries are not "allowed" or "forbidden": they
are the surface a verifier must probe at runtime (docs/RUNTIME_PROBES.md).

## Linear layers

| weight format | kernel at this sha | class-A sites | status | flags needed |
|---|---|---|---|---|
| bf16/fp16 unquantised | `torch.nn.functional.linear` (cuBLASLt) | none in vLLM source | class C | record `CUBLAS_WORKSPACE_CONFIG`; `VLLM_BATCH_INVARIANT=1` pins `:16:8` on sm90+ and replaces mm/linear with Triton on sm80 |
| fp8 per-tensor / per-token, sm90+ | CUTLASS c3x `PersistentScheduler` | none | allowed | none (dynamic scale `atomicMaxFloat` is class B) |
| fp8 / int8, sm75-sm89 | CUTLASS c2x `ThreadblockSwizzleStreamK` + `kGemmSplitKParallel` | vllm-0186 (A3 by design) | allowed | CUTLASS default `kMixed` reduction strategy and the parallel split-K reduce kernel are fixed order (cutlass-0003, cutlass-0005) |
| fp8 via `torch._scaled_mm` | cuBLASLt | none in vLLM source | class C | as unquantised |
| fp8 blockwise via DeepGEMM | DeepGEMM `fp8_gemm_nt` | none (no split-K, DeepGEMM-0003..0006 are A1) | allowed | none |
| GPTQ, AWQ, compressed-tensors wNa16 on sm80-89 (and sm90 when Machete cannot implement) | Marlin | vllm-0010 (A3) | allowed | `VLLM_MARLIN_USE_ATOMIC_ADD` must stay `0` (default); combine is then vllm-0011 (A2) |
| wNa16 on sm90+ | Machete (CUTLASS 3.x stream-K) | vllm-0187 (A3) | allowed | CUTLASS default `ReductionMode::Deterministic` (cutlass-0001); Machete does not override it |
| wNa16 when Marlin, Machete, AllSpark and Conch cannot implement | Exllama (`q_gemm.cu`) | vllm-0001..0004 (A) | **forbidden** | do not set `VLLM_DISABLED_KERNELS` to remove Marlin/Machete; on GPUs below sm80 there is no allowed wNa16 dense kernel |
| GPTQ w8a16 with AllSpark | AllSpark | vllm-0028 (A2) | allowed | none |
| AWQ with `AWQLinearMethod` (batch-invariant mode or no Marlin support) | `awq_gemm` split-K | none (partials summed by `torch.sum`, `gemm_kernels.cu:483,531`) | allowed | none |
| NVFP4 dense | CUTLASS / FlashInfer | see FlashInfer allowlist for the FlashInfer backends | pending | `VLLM_BATCH_INVARIANT=1` forces the CUTLASS kernel (`kernels/linear/__init__.py:889`) |

## Mixture of experts

| MoE path | class-A sites | status | flags / conditions |
|---|---|---|---|
| Triton `fused_moe_kernel` (bf16, fp8, int8, wNa16 with the Triton path) | none; alignment tickets vllm-0016 are B-indirect with order-invariant consumers | allowed | none |
| Marlin MoE | vllm-0022 (A3, `use_atomic_add=False` constant) | allowed | none |
| CUTLASS fp8 / nvfp4 MoE (`cutlass_moe`) | none in vLLM; data-prep tickets vllm-0026 B-indirect | allowed pending the CUTLASS grouped-GEMM check noted in vllm-0026 | none |
| `moe_wna16` CUDA kernel (4-bit, group 32/64/128, tokens per expert <= 6; reached by `--quantization moe_wna16`, and by AutoGPTQ, AutoAWQ or compressed-tensors MoE layers that Marlin MoE cannot handle) | vllm-0014 (A) | **forbidden** | avoid by using a Marlin-compatible MoE layout, or by forcing the Triton path; there is no flag that selects the Triton kernel for small batches at this sha |
| FlashInfer CUTLASS MoE backend (`FlashInferExperts`) | flashinfer-0001 (A3 with `use_fused_finalize` default True; vLLM does not pass the flag) | **forbidden** as shipped | would need a `use_fused_finalize=False` pass-through, which does not exist at this sha |
| FlashInfer TRT-LLM MoE backends (`trtllm_fp8/fp4_block_scale_moe`) | cubins (flashinfer-0031) | class C | runtime probe |
| DeepEP dispatch/combine (`--all2all-backend deepep_*`) | none; slot tickets DeepEP-0001, DeepEP-0006 are B-indirect with order-invariant combine | allowed | none |
| Moondream3 MoE | vllm-0046 (A, `index_add_`) | **forbidden** for that model | `torch.use_deterministic_algorithms(True)` would replace the operator; vLLM does not set it |

## Attention

| backend | class-A sites | status | notes |
|---|---|---|---|
| FlashAttention 2/3/4 forward (`FLASH_ATTN`) | none (flash-attention allowlist) | allowed | `VLLM_BATCH_INVARIANT=1` refuses FA4 for batch-invariance, not run-to-run, reasons |
| Triton unified attention (`TRITON_ATTN`) | none found in `vllm/v1/attention/ops` | allowed | |
| FlashInfer FA2-template prefill/decode | none (flashinfer allowlist) | allowed | split-KV partials merged by `cascade.cuh` kernels |
| FlashInfer TRT-LLM attention (sm100+ default, sm90 decode) | cubins | class C | `VLLM_USE_TRTLLM_ATTENTION=0` avoids them; `VLLM_BATCH_INVARIANT=1` also disables them |
| cuDNN prefill (vision encoders) | cubins | class C | |
| DeepSeek sparse MLA (`FLASHMLA_SPARSE` with the indexer) | vllm-0038..0044 B-indirect with unknown downstream; FlashMLA is external | **gate** | needs the runtime probe; the top-k kernels' output order is arrival order |
| FlexAttention | torch.compile output | out of static scope | inductor may emit `tl.atomic_add` for scatter patterns |
| Cascade attention | none | allowed | disabled under `VLLM_BATCH_INVARIANT` for batch-invariance reasons only |

## LoRA

| kernel | class-A sites | status | flags |
|---|---|---|---|
| LoRA shrink (`do_shrink_kernel`, fp8 variant) | vllm-0018, vllm-0019 (A3) | **forbidden** by default | `VLLM_BATCH_INVARIANT=1` sets `split_k=1`; or a user LoRA kernel config folder with `split_k: 1` |
| LoRA expand | none | allowed | |
| Fused MoE LoRA | vllm-0020, vllm-0021 (A3, default `split_k=1`) | allowed | keep default configs |

## Sampling and pooling

| path | class-A sites | status | flags |
|---|---|---|---|
| Greedy or FlashInfer sampler (`VLLM_USE_FLASHINFER_SAMPLER=1`, default) | FlashInfer sampling: no float atomics (flashinfer-0018); vLLM passes `deterministic=True` | allowed | keep the default |
| PyTorch top-p path (small batches, per-request generators, fp64 Gumbel, processed logprobs) | vllm-0049 (A, float `cumsum`; low confidence) | **gate** | runtime probe; or `torch.use_deterministic_algorithms(True)` |
| Triton top-k/top-p (`batch >= 8`) | none | allowed | |
| Mean pooling (`MeanPool`) | vllm-0045 (A, `index_add_`) | **forbidden** for pooling models | `torch.use_deterministic_algorithms(True)` |

## Parallelism

| mechanism | class-A sites | status | flags |
|---|---|---|---|
| custom all-reduce (`csrc/custom_all_reduce.cuh`) | none; `packed_reduce` loops ranks in fixed order (line 289-297) | allowed | default when available |
| NCCL (PyNccl) | class C | probe | record `NCCL_ALGO`, `NCCL_PROTO`, channel counts; `VLLM_BATCH_INVARIANT=1` pins `allreduce:tree`, one channel, `Simple` |
| NCCL symmetric memory all-reduce | class C | probe | disabled by `VLLM_BATCH_INVARIANT` |
| FlashInfer all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER=1`, default 0) and allreduce+RMSNorm fusion (on by default for TP>1 on sm90/sm100 with FlashInfer, `config/vllm.py:128-146`) | none in FlashInfer source: flashinfer-0022 shows fixed rank order | allowed on this reading | vLLM's own comment disagrees; keep the fusion off until the probe confirms |
| MiniMax fused all-reduce+RMSNorm | vllm-0029 (B counter); reduction loop not read | unknown | |
| Pipeline parallel | no reduction | allowed | |

## Compilation

`torch.compile` (`CompilationMode.VLLM_COMPILE`) is on by default because
`optimization_level` defaults to `O2` (`config/vllm.py:361`, `1169-1173`).
Inductor-generated kernels are outside this census; inductor can emit
`tl.atomic_add` for scatter patterns. A verifier should either require
`-O0` (`CompilationMode.NONE`) or capture the generated Triton code and
scan it with `scan/py.py`.

## Summary for a verifier

Allowed with no flags: bf16 dense with FA2/FA3/FA4/Triton/FlashInfer-FA2
attention, fp8 through CUTLASS c3x or DeepGEMM, Marlin, Machete and
AllSpark wNa16, Triton and Marlin MoE, DeepEP, custom all-reduce, greedy or
FlashInfer sampling. Class C remains: cuBLASLt for unquantised linears and
`_scaled_mm`, NCCL, TRT-LLM attention cubins on Blackwell.

Forbid or gate: Exllama (never default), `moe_wna16` CUDA MoE kernel,
FlashInfer CUTLASS MoE backend, LoRA shrink without `VLLM_BATCH_INVARIANT=1`,
mean pooling, Moondream3, the PyTorch top-p path, DeepSeek sparse attention
top-k, torch.compile output, and `VLLM_MARLIN_USE_ATOMIC_ADD=1`.
