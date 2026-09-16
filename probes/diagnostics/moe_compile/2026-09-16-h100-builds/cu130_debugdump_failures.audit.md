# MoE compile diagnostic audit

Evidence state: **INCOMPLETE**.

## Controls

- Selected GPU UUIDs: `['GPU-fcd67bc9-4348-93cc-aba6-22a2076f2fdc']`
- Drivers: `['580.126.09']`
- GPU models: `['NVIDIA H100 80GB HBM3']`
- Torch CUDA builds: `['13.0']`
- Distinct named cache roots: `True`

## Cells

| cell | state | usable | repeat tokens | duplicate pairs r0/r1 | short cycles r0/r1 | first-token margin range r0 |
|---|---:|---:|---:|---:|---:|---:|
| `cu130_debugdump_failed/compile-on_graphs-off_runner-v2` | failed | False | n/a | n/a | n/a | n/a |
| `cu130_debugdump_failed/compile-off_graphs-off_runner-v2` | failed | False | n/a | n/a | n/a | n/a |

## One-axis comparisons

| kind | left | right | controlled | identical tokens | identical logprobs | dist diffs |
|---|---|---|---:|---:|---:|---:|
| compiled_vs_eager | `cu130_debugdump_failed/compile-on_graphs-off_runner-v2` | `cu130_debugdump_failed/compile-off_graphs-off_runner-v2` | False | 0/0 | 0/0 | 0 |
|  | confounds | left cell is failed, left selected-backend evidence is incomplete, model differs or is missing, revision differs or is missing, right cell is failed, right selected-backend evidence is incomplete, sampling differs or is missing |  |  |  |  |

## Method bounds

- Cold-root proof covers only VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, and TRITON_CACHE_DIR; it does not prove every driver or library cache was cold.
- Two generate calls establish in-process repeatability only. One cell does not establish fresh-process repeatability of that configuration.
- Matching selection logs identify vLLM's named attention and MoE implementations, not the exact generated GPU kernels.
- Debug-dump instrumentation is recorded as an execution control. A no-dump run retains DEBUG logs and named cache trees, but not depyf/FX debug dumps.
- A CUDA-build comparison is a comparison of the recorded installed stacks. Distribution differences are reported rather than silently attributed to one wheel.
- Recorded versions, module paths, RECORD hashes, and distribution differences bound the current run; they do not by themselves prove exact identity with the historical stack.
- Eager vLLM is an internal reference, not an external accuracy oracle.
- V1 versus V2 changes the runner stack, and is not a one-kernel comparison.

## Unavailable evidence

- Failed cells: `['cu130_debugdump_failed/compile-on_graphs-off_runner-v2', 'cu130_debugdump_failed/compile-off_graphs-off_runner-v2']`
- Missing cells: `[]`
- Invalid cells: `[]`
