# fp-atomics-census

A census of floating-point atomic operations in the GPU kernel sources of
the main open-source LLM inference engines, and what it implies for
bit-exact verification of AI inference.

Eight repositories are pinned in `scan-manifest.json` and scanned at those
commits: vLLM, SGLang, FlashInfer, FlashAttention, Marlin, DeepGEMM, DeepEP
and the split-K and stream-K reduction paths of CUTLASS. Every atomic site
the scanner finds is classified by reading the surrounding code. The result
is a machine-readable inventory, a per-engine list of options a verifier
has to pin, and probe scripts for what reading cannot settle.

The census itself is static. The probes in `probes/` have been run on one
stack, reported below; nothing here certifies a configuration as bit-exact. What it establishes is where
order-dependent floating-point accumulation exists in source at the
pinned commits, which options select it, and which binaries remain
opaque.

## Why

Verifying that a datacenter ran the workload it claims to have run, for
AI governance, treaty verification or auditing an API provider, is
simplest when inference is bit-exactly reproducible: the verifier re-runs
a sampled computation and compares hashes. Cankaya (2026) argues that the
engines are already deterministic once the configuration is recorded and
that the only genuine run-to-run non-determinism comes from floating-point
atomic accumulation, of which he found one kernel (the exllama GPTQ path
in vLLM) plus FlashAttention-3's backward pass. If that holds across the
ecosystem, verifiers can demand bit-exact matches instead of tolerance
schemes such as DiFR, which leave an adversary slack. If it does not, the
census says which configurations a verification regime has to forbid or
gate.

## What was found

Counts come from `make summary`, which reads the inventory; no number
below is typed by hand.

| Engine | Sites | A | A1 | A2 | A3 | B | B-indirect | C |
|---|---|---|---|---|---|---|---|---|
| vllm | 164 | 21 | 20 | 2 | 9 | 93 | 12 | 7 |
| sglang | 263 | 12 | 27 | 1 | 8 | 202 | 4 | 9 |
| flashinfer | 91 | 19 | 5 | 2 | 3 | 48 | 6 | 8 |
| flash-attention | 27 | 12 | 0 | 0 | 10 | 4 | 0 | 1 |
| marlin | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| DeepGEMM | 10 | 2 | 4 | 0 | 0 | 1 | 2 | 1 |
| DeepEP | 13 | 0 | 0 | 0 | 0 | 9 | 3 | 1 |
| cutlass | 6 | 0 | 0 | 2 | 2 | 1 | 1 | 0 |
| Total | 576 | 66 | 56 | 8 | 32 | 359 | 28 | 27 |

The premise holds for the dense forward path. Attention forward in every
backend that ships source has no order-dependent atomic; unquantised
linear layers go to cuBLASLt; the Triton fused MoE, the Marlin MoE,
DeepGEMM's grouped GEMMs and DeepEP's dispatch and combine have none on
their default paths; CUTLASS stream-K defaults to its deterministic
reduction. For a bf16 dense model, or a common FP8 or GPTQ dense model,
served by vLLM with the stock configuration, the census found no
order-dependent atomic in engine source.

The premise fails as a statement about the ecosystem. Thirteen class-A
sites sit on default inference paths, and five gates default to the
atomic side. They cluster in adapters, MoE finalisation, quantised
kernels with a split-K or atomic-add fast path, sampling renormalisation
and PyTorch library operators used by model code.

Default-path class-A sites at inference (row ids in `inventory/`):

- vllm-0014: the `moe_wna16` CUDA kernel for 4-bit GPTQ and AWQ MoE
  accumulates each token's expert output with a `half` or `bfloat16`
  `atomicAdd`; selected for small batches.
- vllm-0032 to vllm-0037: six compare-and-swap loops on packed `half2` and
  `bfloat162` in the ROCm gfx1100 GPTQ kernels, the exllama pattern.
  Medium confidence; no ROCm build was exercised.
- vllm-0045 and vllm-0046: `index_add_` in mean pooling and in the
  Moondream3 expert combine. PyTorch documents the CUDA kernel as
  non-deterministic.
- vllm-0050 and sglang-0035: float `cumsum` on CUDA in DiffusionGemma and
  the Gemma3n audio encoder. Low confidence; the scan implementation is
  build-dependent.
- sglang-0011: the FP8 blockwise GEMM selects a CUTLASS stream-K kernel
  built with `ReductionMode::Nondeterministic` when `k > 3n`. Reached on
  Hopper when DeepGEMM is not installed.
- flashinfer-0016: the multi-CTA top-k renormalisation sums kept
  probabilities with a float `atomicAdd` and has no deterministic switch.

Gates whose default selects the atomic side:

- vllm-0018, vllm-0019: LoRA shrink uses split-K with `tl.atomic_add`
  unless `VLLM_BATCH_INVARIANT=1`.
- sglang-0008: `should_use_atomic_add_reduce` contains an `if not True:`
  stub, so Marlin's atomic-add reduction is on for every projection with
  `n < 2048, k >= 2048` as the kernel sees it, after fusion and
  tensor-parallel sharding: the down projection of small models and the
  output projection of a 7B model at TP=2, but no projection of a 7B
  model at TP=1.
- flashinfer-0001: `cutlass_fused_moe` defaults to `use_fused_finalize=True`,
  a `red.global.add` on `f16x2`. vLLM and SGLang reach it only when their
  MoE backend is set to `flashinfer_cutlass`, which is opt-in in both.
- flashinfer-0015: `top_p_renorm_probs` defaults to `is_deterministic=False`.

SGLang's LoRA kernels with float atomics (sglang-0019, sglang-0021 and the
rest of its experimental LoRA backend) sit behind `--lora-use-virtual-experts`
and `SGLANG_EXPERIMENTAL_LORA_OPTI`, both off by default, so a stock SGLang
LoRA deployment has no class-A site in source. vLLM's does.

Twelve class-A sites are training only: FlashAttention's dQ, dK and dV
accumulation across five backward kernels, plus two autograd backward
functions in vLLM and FlashAttention's padding helpers. Ten more A3 rows
are the `deterministic=False` default of those backward kernels. All are
recorded with `default_path: false` because they do not run at inference.

Twenty-eight integer atomics decide a slot or order that float data later
follows. Ten are order-invariant on reading, two are not, and sixteen are
unknown: top-k selection kernels whose output order is arrival order and
whose consumers (sparse attention, TensorRT-LLM routing cubins) were not
read.

Twenty-seven sites call into binaries: cuBLASLt (10), cuBLAS (3), cuDNN (3),
NCCL (4), TensorRT-LLM cubins (3), NVLink multicast (2), FlashMLA sparse
and SGLang's FlashInfer CUTLASS MoE runner.
Nothing in engine source pins cuBLASLt's algorithm choice beyond
`CUBLAS_WORKSPACE_CONFIG`. These are settled only by a runtime probe on
the exact SKU, driver and library versions.

## Static exclusion lists

For each engine, the options a verifier has to pin to stay clear of every
default-path order-dependent site found in source. These are exclusion
lists derived from reading, not certificates: the opaque binaries and any
generated kernels are outside them.

| Engine | Pin | Reason |
|---|---|---|
| vLLM | no LoRA, or `VLLM_BATCH_INVARIANT=1` | vllm-0018, vllm-0019 |
| vLLM | no `moe_wna16` quantisation method | vllm-0014 |
| vLLM | MoE backend not `flashinfer_cutlass`, or patch `use_fused_finalize=False` | flashinfer-0001 |
| vLLM | `VLLM_MARLIN_USE_ATOMIC_ADD` unset (default) | vllm-0010 |
| vLLM | no mean-pooling models, no Moondream3, no DiffusionGemma | vllm-0045, vllm-0046, vllm-0050 |
| vLLM | no gfx1100 ROCm build | vllm-0032 to vllm-0037 |
| SGLang | no GPTQ or AWQ model through Marlin whose fused, sharded projections have `n < 2048, k >= 2048` (small models, TP of 2 or more) until the stub at `marlin_utils.py:476` is fixed; the deterministic-inference mode does not cover it | sglang-0008 |
| SGLang | DeepGEMM installed, or no FP8 blockwise shapes with `k > 3n` | sglang-0011 |
| SGLang | LoRA without `--lora-use-virtual-experts` and without `SGLANG_EXPERIMENTAL_LORA_OPTI` | sglang-0019, sglang-0021 |
| SGLang | `--moe-runner-backend` not `flashinfer_cutlass` | flashinfer-0001 |
| SGLang | no Gemma3n audio | sglang-0035 |
| FlashInfer | `top_p_renorm_probs(..., is_deterministic=True)`; avoid `top_k_renorm_probs` on large vocabularies | flashinfer-0015, flashinfer-0016 |
| FlashInfer | `cutlass_fused_moe(..., use_fused_finalize=False)` | flashinfer-0001 |
| FlashAttention | forward only; for training pass `deterministic=True` | flash-attention-0001 to 0024 |
| DeepGEMM | do not use the `bmk,bnk->mn` einsum | DeepGEMM-0001, DeepGEMM-0002 |
| Marlin, DeepEP, CUTLASS | nothing to pin in source; CUTLASS callers must not set `ReductionMode::Nondeterministic` | cutlass-0001 |
| all | pin and probe cuBLASLt, cuDNN, NCCL and any downloaded cubins | class C rows |

`VLLM_BATCH_INVARIANT` and SGLang's deterministic mode address a
different property, batch invariance, which is what lets a verifier
re-run one request out of a batch. Both are off by default and cost
performance.

## Runtime results on one stack

The probes were run on 6 and 7 September 2026 on RunPod: one H100 SXM, a
two-H100 SXM pod, and one B200, all with driver 580.1xx, CUDA 13.0, torch
2.13.0, vLLM 0.28.0, SGLang 0.5.19 and FlashInfer 0.6.16. The JSON reports
are in `probes/results/`, one directory per stack; the table below is
condensed from `python probes/report.py`, and every inventory row a probe
bears on carries the verdict in its `runtime_evidence` field. Each probe
ran at least four times in each of two processes; "differs" means at
least one repeat was not bitwise identical to the first.

| Probe | Verdict | Reading |
|---|---|---|
| `index_add_`, `scatter_add_` on contended indices | differs; identical with deterministic algorithms on | vllm-0045, vllm-0046, flash-attention-0026 confirmed |
| float `cumsum` | identical in both modes | the cumsum rows are stable on this torch build |
| `bincount` with weights | differs, even in deterministic mode | no inventory row; PyTorch has no deterministic kernel for it |
| cuBLASLt, 60 projection shapes, plain matmul and bias linear, bf16 and fp16 | identical on all 240 | the heuristic chose split-K on 11 to 18 shapes per sweep, always with the workspace reduction scheme, never the in-place atomic one |
| FlashInfer `top_p_renorm_probs` | differs by default; identical with `is_deterministic=True` | flashinfer-0015 confirmed |
| FlashInfer multi-CTA `top_k_renorm_probs` | differs | flashinfer-0016 confirmed |
| FlashInfer radix top-k | output order differs by default; identical with `deterministic=True` | the order the B-indirect rows depend on does change at runtime |
| FlashInfer fused MoE finalize, autotuned | identical at top-k 2; differs at top-k 4; identical at top-k 8 in 3 repeats | flashinfer-0001 confirmed; at top-k 2 two addends onto zero commute, so no order dependence is possible |
| FlashInfer unfused finalize, FA2 decode | identical | |
| vLLM Qwen3-8B and Qwen2.5-7B bf16, stock, 12 repeats per process | 1 to 2 of 12 repeats differ per process, whole requests at a time, sometimes changing sampled tokens | batch composition, see below |
| vLLM the same with `VLLM_BATCH_INVARIANT=1`, or with `max_num_seqs=1` on the stock kernels | identical, 24 of 24 | |
| vLLM GPTQ through Marlin (atomic add off) and through Machete, fp16 and bf16 | identical | vllm-0010 and vllm-0187 as read |
| vLLM LoRA with the default split-K | differs, 2 to 24 sampled tokens per run | vllm-0018 confirmed; identical with `VLLM_BATCH_INVARIANT=1` |
| vLLM GPTQ MoE through Marlin MoE, 12 repeats | identical | vllm-0022 null; the `moe_wna16` kernel (vllm-0014) could not be selected in 0.28.0, the method crashes at load, and the batch-invariant MoE kernels refuse this model's block shape |
| SGLang Qwen3-8B bf16 with radix cache and overlap scheduler off | 3 of 6 repeats differ, exactly one request | batch composition; identical with `--enable-deterministic-inference` |
| Marlin kernel on random 4-bit weights, `use_atomic_add=True`, with and without fp32 reduce | differs on every shape, fp16 and bf16 | sglang-0008, vllm-0010: the kernel is order-dependent whenever the flag is on |
| SGLang GPTQ 7B through Marlin at TP=1 | identical, 12 of 12 | every fused projection has `n >= 2048`, so the stub never engages |
| SGLang GPTQ 1.5B (down_proj `n=1536, k=8960`), and 7B at TP=2 | differs, first hashes differ between processes, sampled tokens change | sglang-0008 confirmed; `--enable-deterministic-inference` does not remove it |
| vLLM GPTQ 1.5B, one sequence per batch: Machete, Marlin, Marlin with `VLLM_MARLIN_USE_ATOMIC_ADD=1` | identical, identical, differs | vllm-0010: with composition fixed only the atomic-add path differs |
| NCCL two-rank all-reduce, default and tree, NVLink multicast off | identical | vllm-0182, sglang-0269; the container cannot bind multicast memory, so NVLS stays unprobed |
| vLLM TP=2, custom all-reduce and NCCL, one sequence per batch | identical, same hash for both | with the stock scheduler both differed by whole requests, as at TP=1 |
| vLLM Mixtral GPTQ through Marlin MoE, one sequence per batch | identical | vllm-0022 null; `moe_wna16` cannot be selected explicitly in 0.28.0 |
| DeepGEMM `bmk,bnk->mn` einsum on sm90 | differs | DeepGEMM-0001 confirmed |
| FlashInfer TensorRT-LLM decode cubin on B200 | identical, 8 runs | flashinfer-0038: first runtime evidence for a class-C cubin |
| FlashInfer fused MoE finalize on B200, autotuned | identical at top-k 2 and 8, differs at top-k 4 | same pattern as on the H100 |
| vLLM on B200: Qwen3-8B-FP8 through the FlashInfer FP8 path; Qwen3-8B bf16 with the batch-invariant mode; bf16 MoE on the Triton backend with one sequence per batch | identical | the Blackwell defaults are clean once composition is pinned |
| vLLM on B200, stock scheduler: Qwen3-8B bf16 (FlashInfer attention backend); bf16 MoE | 1 of 12 repeats per process differed by 2 to 5 logprob values; 1 of 6 differed in generation length | batch composition again; the B200 FlashInfer CUTLASS MoE backend refuses unquantised weights, so flashinfer-0001 was not reached through vLLM |
| SGLang on B200: bf16 (default attention is `trtllm_mha`), FP8 blockwise default | identical in 23 of 24 and 12 of 12 | sglang-0267: the TensorRT-LLM attention cubins gave no difference in 24 engine runs |
| SGLang FP8 blockwise | identical | sglang-0011 is unreachable: 0.5.19 builds its CUTLASS FP8 GEMM for SM120 only and routes Hopper to DeepGEMM or Triton |
| FlashInfer all-reduce fusion, NVLink multicast, cuDNN | not run | the fusion probe fails in workspace setup; multicast is blocked by the container; no cuDNN path exercised |

One reading is recorded as a null with the probe names: vLLM's Marlin MoE
was identical in every run, including with batch composition fixed. A
null after twelve runs is not proof of order-invariance, but the
verifier's forbid list can carry it as "not observed" rather than
"forbidden". The SGLang Marlin null of the first day turned out to be a
shape effect: the stub's condition is evaluated on the fused, sharded
weight, and a 7B model at TP=1 never meets it.

The residual differences in the dense path are not kernel randomness. With
the batch-invariant mode on, or with one sequence per batch on the stock
kernels, every configuration was identical in every repeat. On the stock
kernels with the default scheduler, requests reach the engine over a
socket, the first scheduling step sometimes holds a different set of
prompts, and the kernels are not batch-invariant, so a request's logprobs
and occasionally its tokens depend on who else was in the batch. For a
verifier that re-runs sampled requests, this is the binding constraint on
this stack: the atomics the census found are avoidable by configuration,
and what remains is batch invariance, which both engines offer only as an
opt-in mode with a performance cost.

## Layout

```
scan-manifest.json   pinned repo shas and in-scope directories
scan/                candidate scanner (regex plus tree-sitter for C++, ast for Python)
candidates/          raw scanner output, one jsonl and one coverage json per engine
triage/              inventory JSON Schema, validator, summary and CSV export
inventory/           triaged sites, one jsonl per engine; inventory.csv at the root
probes/              GPU scripts for the rows that reading cannot settle
tests/               scanner unit tests on synthetic fixtures
```

## How to run

```
make venv                     # Python 3.11 venv with jsonschema, tree-sitter, pytest
make clone                    # fetch every repo at its pinned sha into repos/
make test                     # scanner unit tests on tests/fixtures
make census REPO=vllm         # re-run the scanner on one repo (SHA=... to override)
make census-all               # re-run on all eight
make validate                 # schema, sha, snippet, reference and provenance checks
make summary                  # the counts quoted above, from the inventory
make csv                      # regenerate inventory.csv
python -m triage.coverage_table candidates/*.coverage.json
```

`make census` refuses to run on a checkout whose HEAD differs from the
manifest. Re-running it regenerates `candidates/`; the inventory is a
human product and is not regenerated.

## How to read the inventory

Each line of `inventory/<engine>.jsonl` is one site, validated against
`triage/inventory.schema.json`. The classes:

- **A**: order-dependent floating-point accumulation with no mitigation
  found: `atomicAdd` on float types, compare-and-swap add loops, PTX
  `red`/`atom` add on float, `multimem.red`, TMA `cp.reduce.async.bulk`
  add, `tl.atomic_add` on float, and PyTorch operators documented as
  non-deterministic on CUDA.
- **A1**: the same primitive where the reading shows one writer per
  address, so no contention.
- **A2**: serialised by a lock or turnstile so the order is fixed.
- **A3**: behind a flag; `gate` records the flag, where it is read, its
  default at the sha, and `default_is_atomic`.
- **B**: exact regardless of order: integer atomics, float max and min,
  locks and counters.
- **B-indirect**: an exact atomic that decides a slot or order for later
  float data; `downstream.order_invariant` says whether that can change a
  float reduction.
- **C**: a call into a binary (cuBLASLt, cuDNN, NCCL, cubins) whose
  reduction strategy cannot be read; `opaque_target` names it.

Fields that matter most:

- `default_path`: true if a stock configuration selects the site on at
  least one supported hardware profile without changing a default;
  `default_path_condition` names that profile and shape. It means
  reachable by default, not selected on every deployment.
- `path.direction`: rows in backward kernels are training only.
- `evidence`: `file:line` references that justify the class; the
  validator checks that each resolves in the pinned checkout.
- `provenance` and `candidate_ids`: `scanner` rows cite the candidate
  they were triaged from; `manual` rows were found by reading.
- `definition_only`: a helper whose reachable call sites have their own
  rows; excluded from the counts.
- `confidence` and `notes`: `low` means the class is a best reading, and
  the notes say what is missing.

`candidates/` keeps everything the scanner matched, including matches in
comments and strings with an `excluded_reason`, so the triage can be
audited.

## Probes

`probes/` holds scripts that run a computation several times and compare
outputs bit for bit, recording GPU, driver and library versions with the
verdict. `probe_engine_logits.py` runs vLLM or SGLang on a fixed batch;
`probe_cublaslt_algo.py` runs a matmul under cuBLASLt logging and records
the algorithm, split-K count and reduction scheme the heuristic chose;
`probe_torch_ops.py` covers `index_add_`, `scatter_add_` and `cumsum`;
`probe_flashinfer.py`, `probe_kernels.py` and `probe_collectives.py`
target the FlashInfer, SGLang, DeepGEMM and NCCL rows named in their
docstrings.

```
pip install -r probes/requirements.txt
bash probes/run_all.sh          # every probe twice in fresh processes, then the report
bash probes/run_all.sh quick    # engine-free probes only
python probes/report.py         # tabulate probes/results/<stack>/*.json by inventory row
```

Reports land in `probes/results/<gpu>_<driver>_<torch>/` and are committed
as evidence for that stack only. The engine probes run against installed
wheels, not the pinned source shas, so their mapping to inventory rows is by
kernel identity; the report records both. A `DIFFERS` verdict is conclusive
for the stack it ran on; an identical verdict after a few runs is evidence,
not proof.

## Limits

- The scanner finds sites; every A1, A2 and A3 decision was made by
  reading, and a wrong reading is possible. Nineteen rows carry low
  confidence.
- Binaries are opaque. cuBLASLt serves every unquantised linear layer.
- Runtime-generated kernels are out of scope: `torch.compile` output,
  which vLLM enables by default, Triton autotune choices and JIT
  variants are seen only through their templates.
- No environment is pinned. The probes record versions when run; the
  census itself makes no claim about any PyTorch, CUDA or driver
  version.
- The ROCm and XPU trees of SGLang and the ROCm kernels of vLLM were
  read but not built or run.
- The pinned shas are from 6 July 2026. Line numbers are valid only there.

## Citations

- N. Cankaya, "Bit-Exact AI Inference Verification Without Performance
  Tradeoffs", arXiv:2606.00279, 2026. https://arxiv.org/abs/2606.00279
- "DiFR: Inference Verification Despite Nondeterminism", arXiv:2511.20621,
  2025. https://arxiv.org/abs/2511.20621
- H. He and Thinking Machines Lab, "Defeating Nondeterminism in LLM
  Inference", September 2025.
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- SGLang team, "Towards Deterministic Inference in SGLang and Reproducible
  RL Training", September 2025.
  https://lmsys.org/blog/2025-09-22-sglang-deterministic/
- PyTorch, `torch.use_deterministic_algorithms` reference.
  https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html

## Licence

MIT. See LICENSE.
