# cutlass: deterministic configuration allowlist

Repository `NVIDIA/cutlass` at `4cbaaae4c70dbb2b3e07844010a381b97217ed76`.
Scope limited to split-K and stream-K reduction modes, as used by the
engines' CUTLASS kernels. Derived from `inventory/cutlass.jsonl`.

| reduction mode | mechanism | class | status | who uses it |
|---|---|---|---|---|
| Serial split-K (`GemmUniversalMode::kGemm` with `split_k_slices > 1`, 2.x) | semaphore-ordered accumulation | A2 | allowed | none of the scanned engines at their pinned shas |
| Parallel split-K (`kGemmSplitKParallel`) | partials to workspace, `ReduceSplitK` kernel sums partitions in index order (cutlass-0005) | A2 (A3 by design) | allowed | vLLM `scaled_mm_c2x` (sm75-sm89 int8/fp8) |
| 2.x stream-K (`ThreadblockSwizzleStreamK`) | `kReductionStrategy = kMixed` by default: turnstile or separate reduction (cutlass-0003) | A3, default deterministic | allowed with the default | vLLM `scaled_mm_c2x` |
| 2.x stream-K with `kAtomic` | arrival-order accumulation | A | forbidden | compile-time constant; no engine sets it |
| 3.x stream-K (`PersistentTileSchedulerSm90StreamK`, also sm100 params) with `ReductionMode::Deterministic` (default) | turnstile by K index or separate reduction units (cutlass-0001, 0002) | A2 | allowed | vLLM Machete (does not override the mode), FlashInfer sm120 FP4 GEMM |
| 3.x stream-K with `ReductionMode::Nondeterministic` | `wait_lt` then accumulate in arrival order | A | **forbidden** | SGLang `fp8_blockwise_gemm_sm90_dispatch.cuh:173` (sglang-0009) |
| 3.x `PersistentScheduler` (no stream-K, no split-K) | no cross-CTA reduction | none | allowed | vLLM `scaled_mm_c3x`, SGLang blockwise GEMM for `k <= 3n` |
| EVT row/column reduction fusion (`Sm90RowReduction`, `Sm90ColReduction`) with `FinalReduction` | last tile elected by an integer counter reduces workspace slots in index order (cutlass-0006, B-indirect, order invariant) | B-indirect | allowed | not found in the engines' kernels; the `IsAtomic` variant would be class A |
| Grouped GEMM (`sm90_tile_scheduler_group`) | no split-K | none | allowed | vLLM and SGLang CUTLASS MoE |

What a verifier should require from a CUTLASS-based kernel: no
`ReductionMode::Nondeterministic`, no `kAtomic` swizzle, and either
`PersistentScheduler` or a stream-K/split-K mode listed as A2 above. The
reduction mode is a runtime argument for 3.x stream-K, so it must be
read from the launching code, not from CUTLASS itself.
