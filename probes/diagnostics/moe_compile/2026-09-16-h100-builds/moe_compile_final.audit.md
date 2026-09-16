# MoE compile diagnostic audit

Evidence state: **CONTROLLED**.

## Controls

- Selected GPU UUIDs: `['GPU-fcd67bc9-4348-93cc-aba6-22a2076f2fdc']`
- Drivers: `['580.126.09']`
- GPU models: `['NVIDIA H100 80GB HBM3']`
- Torch CUDA builds: `['12.9', '13.0']`
- Distinct named cache roots: `True`

## Cells

| cell | state | usable | repeat tokens | duplicate pairs r0/r1 | short cycles r0/r1 | first-token margin range r0 |
|---|---:|---:|---:|---:|---:|---:|
| `cu129_v2/compile-on_graphs-off_runner-v2` | success | True | True | 0/0 | 9/9 | 0..2.25 |
| `cu129_v2/compile-off_graphs-off_runner-v2` | success | True | True | 8/8 | 0/0 | 0.0625..5.25000001 |
| `cu130_v2/compile-on_graphs-off_runner-v2` | success | True | True | 0/0 | 9/9 | 0..2.25 |
| `cu130_v2/compile-off_graphs-off_runner-v2` | success | True | True | 8/8 | 0/0 | 0.0625..5.25000001 |
| `cu130_v1/compile-on_graphs-off_runner-v1` | success | True | True | 0/0 | 5/5 | 0..2.4375 |
| `cu130_v1/compile-off_graphs-off_runner-v1` | success | True | True | 8/8 | 0/0 | 0.0625..5.24999989 |
| `cu130_v2_repeat/compile-on_graphs-off_runner-v2` | success | True | True | 0/0 | 9/9 | 0..2.25 |
| `cu129_v2_repeat/compile-on_graphs-off_runner-v2` | success | True | True | 0/0 | 9/9 | 0..2.25 |

## One-axis comparisons

| kind | left | right | controlled | identical tokens | identical logprobs | dist diffs |
|---|---|---|---:|---:|---:|---:|
| compiled_vs_eager | `cu129_v2/compile-on_graphs-off_runner-v2` | `cu129_v2/compile-off_graphs-off_runner-v2` | True | 0/16 | 0/16 | 0 |
| cuda_build | `cu129_v2/compile-on_graphs-off_runner-v2` | `cu130_v2/compile-on_graphs-off_runner-v2` | True | 16/16 | 16/16 | 44 |
| cuda_build | `cu129_v2/compile-on_graphs-off_runner-v2` | `cu130_v2_repeat/compile-on_graphs-off_runner-v2` | True | 16/16 | 16/16 | 44 |
| fresh_process_repeat | `cu129_v2/compile-on_graphs-off_runner-v2` | `cu129_v2_repeat/compile-on_graphs-off_runner-v2` | True | 16/16 | 16/16 | 0 |
| cuda_build | `cu129_v2/compile-off_graphs-off_runner-v2` | `cu130_v2/compile-off_graphs-off_runner-v2` | True | 16/16 | 16/16 | 44 |
| compiled_vs_eager | `cu129_v2/compile-off_graphs-off_runner-v2` | `cu129_v2_repeat/compile-on_graphs-off_runner-v2` | True | 0/16 | 0/16 | 0 |
| compiled_vs_eager | `cu130_v2/compile-on_graphs-off_runner-v2` | `cu130_v2/compile-off_graphs-off_runner-v2` | True | 0/16 | 0/16 | 0 |
| v1_vs_v2 | `cu130_v2/compile-on_graphs-off_runner-v2` | `cu130_v1/compile-on_graphs-off_runner-v1` | True | 0/16 | 0/16 | 0 |
| fresh_process_repeat | `cu130_v2/compile-on_graphs-off_runner-v2` | `cu130_v2_repeat/compile-on_graphs-off_runner-v2` | True | 16/16 | 16/16 | 0 |
| cuda_build | `cu130_v2/compile-on_graphs-off_runner-v2` | `cu129_v2_repeat/compile-on_graphs-off_runner-v2` | True | 16/16 | 16/16 | 44 |
| v1_vs_v2 | `cu130_v2/compile-off_graphs-off_runner-v2` | `cu130_v1/compile-off_graphs-off_runner-v1` | True | 16/16 | 0/16 | 0 |
| compiled_vs_eager | `cu130_v2/compile-off_graphs-off_runner-v2` | `cu130_v2_repeat/compile-on_graphs-off_runner-v2` | True | 0/16 | 0/16 | 0 |
| compiled_vs_eager | `cu130_v1/compile-on_graphs-off_runner-v1` | `cu130_v1/compile-off_graphs-off_runner-v1` | True | 0/16 | 0/16 | 0 |
| v1_vs_v2 | `cu130_v1/compile-on_graphs-off_runner-v1` | `cu130_v2_repeat/compile-on_graphs-off_runner-v2` | True | 0/16 | 0/16 | 0 |
| cuda_build | `cu130_v2_repeat/compile-on_graphs-off_runner-v2` | `cu129_v2_repeat/compile-on_graphs-off_runner-v2` | True | 16/16 | 16/16 | 44 |

## Method bounds

- Cold-root proof covers only VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, and TRITON_CACHE_DIR; it does not prove every driver or library cache was cold.
- Two generate calls establish in-process repeatability only. One cell does not establish fresh-process repeatability of that configuration.
- Matching selection logs identify vLLM's named attention and MoE implementations, not the exact generated GPU kernels.
- Debug-dump instrumentation is recorded as an execution control. A no-dump run retains DEBUG logs and named cache trees, but not depyf/FX debug dumps.
- A CUDA-build comparison is a comparison of the recorded installed stacks. Distribution differences are reported rather than silently attributed to one wheel.
- Recorded versions, module paths, RECORD hashes, and distribution differences bound the current run; they do not by themselves prove exact identity with the historical stack.
- Eager vLLM is an internal reference, not an external accuracy oracle.
- V1 versus V2 changes the runner stack, and is not a one-kernel comparison.
