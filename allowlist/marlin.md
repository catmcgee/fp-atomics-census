# marlin: deterministic configuration allowlist

Repository `IST-DASLab/marlin` at `1f25790bdd49fba53106164a24666dade68d7c90`.
Derived from `inventory/marlin.jsonl`.

## Configurations

| kernel | quant | split-K across blocks | class | flags needed |
|---|---|---|---|---|
| `Marlin<threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, stages, group_blocks>` | 4-bit weights, fp16 activations, per-channel or grouped scales | yes, whenever a column slice spans more than one block (`slice_count > 1`) | A2 | none |

The reference Marlin kernel contains one floating-point read-modify-write
on global memory: the split-K global reduce
(`marlin/marlin_cuda_kernel.cu:674-678`). It is serialised by a per-column
lock. Block `slice_idx` waits until the lock equals `slice_idx`
(`barrier_acquire`, lines 162-172), reads the fp16 partial from `C`, adds
it into its fp32 fragment, writes the fp16 result back unless it is the
last slice, and increments the lock (`barrier_release`, lines 174-186).
`slice_idx` and `slice_count` are functions of the static striped tile
assignment (lines 235-269), not of runtime arrival, so the accumulation
order is fixed for a given problem shape and SM count.

The whole reference kernel is therefore on the allowlist with no flags.

## Caveats a verifier should record

- The reduce order depends on the number of blocks launched, which the
  host code derives from the SM count of the device
  (`marlin/marlin_cuda_kernel.cu`, `marlin_cuda` in `marlin_cuda.cpp`). Two
  GPUs of different SM counts will produce different (but each
  reproducible) results. Record the SKU, as Cankaya 2026 already requires.
- Partials are rounded to fp16 between slices. This is a precision choice,
  not a determinism issue, but it means the result is not the fp32
  reference sum.
- Engine ports of Marlin (vLLM `csrc/quantization/gptq_marlin`, SGLang
  `sgl-kernel`) are triaged in their own inventories; this file covers the
  reference repository only.
