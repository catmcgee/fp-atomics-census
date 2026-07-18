# Taxonomy

This document defines what the census counts and how every counted site is
classified. It is applied strictly: a row in `inventory/` must carry exactly
one class from this list, the evidence that justifies it, and a `file:line`
at the pinned sha in `scan-manifest.json`.

## What is a site

An atomic site is any read-modify-write on global or shared GPU memory whose
result depends on the order in which the participating threads, warps or
thread blocks arrive. The scanner finds candidate sites by pattern; a human
turns candidates into inventory rows by reading the surrounding code. The two
are kept in separate directories (`candidates/` and `inventory/`) so that the
triage can be audited against the raw scan.

Order-dependent stores that are not read-modify-writes (for example a
`scatter_` with duplicate indices, where the last writer wins) are recorded
under the same rules: they are order-dependent, so they are counted, and they
are classified by the same sub-cases below. The `kind` field distinguishes
them.

Host-side atomics (`std::atomic`, `std::mutex`) and atomics that only touch
registers are not sites. They stay in `candidates/` with `excluded_reason`
set, and never enter `inventory/`.

## Class A: order-dependent floating-point accumulation

Any of the following, when the operand type is `float`, `double`, `half`,
`bfloat16`, a packed variant (`half2`, `bfloat162`, `float2`, `float4`) or a
Triton or PyTorch float dtype:

- `atomicAdd`, `atomicSub`, `atomicAdd_block`, `atomicAdd_system` and the
  `cuda::atomic_ref<T>::fetch_add` family;
- `atomicCAS` loops that implement a floating-point add (the loop reinterprets
  a float as an integer, adds, and retries);
- inline PTX `red.*.add.{f32,f16,f16x2,bf16,bf16x2,f64}`,
  `atom.*.add.{f32,...}`, and `multimem.red.*.add.*`;
- TMA bulk reductions into global memory (`cp.reduce.async.bulk.*.add`,
  CUTLASS `SM90_TMA_REDUCE_ADD*`, FlashAttention's `SM90_BULK_REDUCE_ADD`).
  The hardware performs the add in the memory system; when more than one
  thread block reduces into the same tile, the order is the arrival order;
- Triton `tl.atomic_add` on a pointer to a float dtype;
- PyTorch CUDA operators documented as non-deterministic because they
  accumulate with atomics: `index_add_`, `scatter_add_`, `scatter_reduce`
  with `sum`, `mean` or `prod`, `index_put_` with `accumulate=True` where the
  backend uses atomics, `bincount` with a float `weights` tensor, `histc`,
  `cumsum` on a CUDA float tensor, `put_` with `accumulate=True`,
  `nn.Embedding` backward and everything else that
  `torch.use_deterministic_algorithms(True)` either replaces or rejects
  (https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html).

A class-A site is genuinely non-deterministic unless one of three sub-cases
holds. The sub-case must be established by reading the code, and the evidence
(a `file:line` of the guard, lock or index construction) goes in the row.

- **A1, no contention.** Every address is written by exactly one thread over
  the lifetime of the kernel launch, so the atomic is a plain store. Typical
  evidence: the index set is provably duplicate free (a token-to-slot map
  built by a prior kernel with unique slots, or the atomic address is a
  function of the thread's own block and thread id only). "Usually unique" is
  not enough; if duplicates are possible on any input, it is not A1.
- **A2, serialised.** Partial sums are combined in a fixed order. Typical
  evidence: a semaphore, lock, ticket counter or named barrier that makes the
  k-th contributor wait for the (k-1)-th (Marlin's `barrier_acquire` and
  `barrier_release` around its global reduce, CUTLASS `Semaphore`,
  FlashAttention's `Barrier::wait_eq` and `arrive_inc` under the
  `Deterministic` template parameter). A2 needs both a float accumulation and
  the ordering primitive around it in the same path.
- **A3, gated.** The atomic path is only reached when a determinism flag is
  off, or a separate workspace-plus-reduce-kernel path exists and is selected
  by configuration. CUTLASS split-K with a `splitKreduce` kernel (partials
  written to a workspace and summed in fixed order by a second kernel) is A3;
  in-place split-K that atomically adds into the output is not. For A3 the
  row records the flag, where it is read, and its default at the pinned sha.

A class-A row without a sub-case is a genuine non-determinism source. That is
the set the FINDINGS document and the allowlists are built from.

## Class B: exact atomics

Atomics whose result does not depend on arrival order:

- integer `atomicAdd`, `atomicSub`, `atomicInc`, `atomicDec`, `atomicOr`,
  `atomicAnd`, `atomicXor`, and their PTX `red`/`atom` forms on `s32`, `u32`,
  `s64`, `u64`, `b32`, `b64`;
- `atomicMax` and `atomicMin` on any type, including floats reinterpreted as
  integers (`atomicMax((int*)p, __float_as_int(v))`); max and min are exact
  and commutative so the result is order-invariant. The same holds for
  Triton `tl.atomic_max` and `tl.atomic_min`;
- `atomicExch` and `atomicCAS` used as locks, tickets or flags;
- integer `bincount` (no weights), integer `cumsum`, integer `scatter_add_`.

Class B is deterministic in value. But if the integer atomic decides the
**ordering or slot assignment** of data that later feeds a floating-point
reduction, the site is recorded as **B-indirect**. Examples: a ticket counter
that assigns each token a slot in an expert's block (MoE alignment); a packet
counter that fixes where a received chunk lands in a combine buffer
(expert-parallel all-to-all); a split-KV counter that decides which block
performs the final merge. For every B-indirect row the `downstream` field
states whether the downstream reduction is order-invariant under permutation
of the slots (for example, a per-token weighted sum that reads its own row
regardless of slot) or whether a different slot order changes the summation
order. If that cannot be determined, the row says so and carries
`confidence: low`.

## Class C: opaque

Reductions performed inside closed binaries that the scan cannot read:

- cuBLAS and cuBLASLt kernel selection, including the `nvjet` family on Hopper
  and Blackwell;
- cuDNN (attention and convolution);
- pre-compiled TensorRT-LLM cubins shipped or downloaded by FlashInfer;
- NCCL collectives (`allreduce`, `reduce_scatter`, `all_to_all`);
- hardware multicast reductions (`multimem.ld_reduce`) whose in-switch
  reduction order is not documented.

Class C rows do not speculate about what the binary does. They record where
the engine calls into it, which configuration knobs change the reduction
strategy (`CUBLAS_WORKSPACE_CONFIG`, cuBLASLt algorithm `reductionScheme`,
`NCCL_ALGO`, `NCCL_PROTO`), and are listed as "unknown, needs runtime probing".
The probe that would settle each one is in `docs/RUNTIME_PROBES.md`.

## Per-site fields

Every inventory row carries, at minimum:

| field | meaning |
|---|---|
| `id` | stable identifier, `<engine>-<nnnn>` |
| `engine` | repository name from the manifest |
| `file` | path relative to the repository root |
| `line_start`, `line_end` | 1-based inclusive line range at the pinned sha |
| `function` | enclosing kernel or function name |
| `class` | `A`, `A1`, `A2`, `A3`, `B`, `B-indirect`, `C` |
| `kind` | the primitive: `atomicAdd`, `atomicCAS-loop`, `ptx-red`, `tma-reduce`, `tl.atomic_add`, `torch-op`, `store-race`, `library-call`, ... |
| `dtype` | operand dtype, or `unknown` |
| `memory_space` | `global`, `shared`, `distributed-shared`, `unknown` |
| `path` | which code path reaches it: quant format, attention backend, MoE or dense, prefill or decode, forward or backward, TP/EP/PP |
| `default_path` | `true` if reachable in a default configuration of the engine, `false` if opt-in, `unknown` |
| `gate` | for A3: the flag, where it is read, and its default |
| `evidence` | `file:line` references that justify the class and sub-case |
| `snippet` | 3 to 6 lines of code at the site |
| `confidence` | `high`, `medium`, `low` |
| `notes` | why the confidence is what it is, and anything a second reader needs |
| `downstream` | B-indirect only: whether slot order can change a float reduction order |

The full schema is `triage/inventory.schema.json` and is enforced by
`make validate`.

## Rules of evidence

- No row without a `file:line` and a snippet.
- No sub-case without a `file:line` for the guard, lock or index construction.
- If contention or gating cannot be determined with confidence, the row keeps
  class `A` (not A1/A2/A3), sets `confidence: low`, and explains in `notes`.
  Uncertainty is never resolved by assumption in the optimistic direction.
- Where a class depends on a runtime flag, the row names the flag and its
  default at the pinned sha.
- "No atomics found" in a directory means the scanner walked it and found no
  candidates. "Not scanned" means it was outside `scan_paths` or excluded.
  The coverage table in `docs/METHOD.md` keeps the two apart.
- A candidate that might be a false positive stays in `candidates/` with a
  note rather than being deleted.
