# Batch-shape experiments

The revised harness targets vLLM 0.28.0. Its CPU analysis and integrity checks are tested locally; the changed GPU controls and hook require a new GPU campaign. Do not mix new observations with the legacy September 2026 arms.

Install the hook in the chosen GPU environment:

```sh
pip install -e probes/shape/shape_hook_pkg
```

Each arm directory is created once. Reusing it fails. Choose a new campaign root for each invocation, supply an immutable model `--revision`, and retain `run.json`, `env.json`, raw hook logs and raw outputs. The manifest includes the input prompts and their digest. Some binary/model content hashes remain explicitly unknown; collect them before making an exact-provenance claim.

## Compilation and graph controls

The controls are independent requests. For vLLM 0.28.0, the harness supplies explicit `CompilationMode` and `CUDAGraphMode` values:

| Arguments | Compilation | Graph mode |
|---|---|---|
| `--cudagraph 1` | VLLM_COMPILE | FULL_AND_PIECEWISE |
| `--cudagraph 0` | VLLM_COMPILE | NONE |
| `--no-compile --cudagraph 1` | NONE | FULL |
| `--no-compile --cudagraph 0` | NONE | NONE |

Full graph capture without compilation is model/backend dependent. Configuration resolution is recorded and checked at the driver and worker. A rejected or downgraded configuration is a failed arm, not a replacement for the requested arm. Recorded dispatch must also be reviewed to determine which modes actually ran; an enabled graph setting does not mean every pass uses a graph. Python hooks can be bypassed by compilation or graph replay, so absent mechanism observations are not zeros.

Legacy `graphs0` arms used `enforce_eager=True`, which disabled both features. Legacy `eager_graphs1` arms also resolved to no graphs. They cannot supply a four-arm factorial. New names include `v2` and distinguish compilation explicitly.

## E2: conditional repeatability with the stock scheduler

```sh
python probes/shape/run_e2.py --model Qwen/Qwen2.5-7B-Instruct --revision MODEL_COMMIT --cudagraph 1 --prefix-caching 1 --repeats 12 --out results_new/e2
```

Vary compilation, graph and prefix settings in separate arms. New prompts include longer inputs; the legacy default accidentally contained only the four short prompts repeated four times. Each generated token must join to a non-null target observation; incomplete or duplicate joins invalidate the arm. Summaries distinguish repeated from singleton history groups.

## E3: scripted trajectory repeatability

```sh
RUN_TAG=a python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --revision MODEL_COMMIT --out results_new/e3
RUN_TAG=b python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --revision MODEL_COMMIT --out results_new/e3
python probes/shape/run_e3.py --compare results_new/e3/Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix1
```

The script adds eight requests, executes five scheduler steps, then adds eight more. `--mixed` uses long second-wave prompts and chunked prefill. Selected observations use offsets from the first actual forward pass. Complete trace lengths, normalised request sets, shapes and observations must agree for an identical verdict.

This does **not** test P2. A separate reconstruction experiment must record exact arrival/admission decisions, per-step token IDs and plans, rebuild prompt/KV state to the chosen boundary in a fresh process, teacher-force the recorded continuation and compare full observations. Neither repeated arrival scripts nor free-running output equality substitutes for those controls.

## E4: neighbour replacement

```sh
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --revision MODEL_COMMIT --target 0 --repeats 12 --out results_new/e4
python probes/shape/run_e4.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --revision MODEL_COMMIT --no-compile --cudagraph 0 --repeats 6 --out results_new/e4
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --revision MODEL_COMMIT --quantization fp8 --fp8-per-tensor --no-compile --cudagraph 0 --repeats 6 --out results_new/e4
```

Original and random-neighbour batches alternate. Prefix caching is disabled. Target slot is recorded and recovered in offline analysis. Use separate campaign roots for different target slots and lengths; a broader campaign should cover multiple positions, distinct targets, mixed prefill, shared prefixes and cache reuse. The current E4 driver does not implement those latter scheduling/cache scenarios. Its null results apply only to the configuration actually run.

No common-prefix truncation is allowed. Summaries are derived files; raw `runs.json` and hook logs are never rewritten during analysis. Per-expert counts inferred from router logits are labelled estimates separately from counts obtained from actual selected expert IDs. Changed counts/scales are mechanism diagnostics, not causal isolation.

## E5: recorded-stack comparison

```sh
python probes/shape/run_e5.py results_h100/e3/ARM/a results_other/e3/ARM/a
```

This compares aligned complete traces and reports a two-stack result. It does not isolate GPU architecture when drivers/configurations differ, or after free-running token divergence. Use identical teacher-forced tokens, weights, tokenizer and controlled software versions for a separate GPU comparison.

## Offline regeneration

```sh
python probes/shape/run_e2.py --analyse SAVED_E2_ARM
python probes/shape/run_e4.py --analyse SAVED_E4_ARM
python probes/shape/run_e3.py --compare SAVED_E3_ARM
make shape-tables
make check-derived
```

Input and analyser SHA-256 digests identify each generated summary. Legacy empty scheduler calls are excluded; new hooks record them explicitly as non-forward events. Missing data is reported as invalid or untested, never as a null result. `make check-derived` performs reanalysis in a temporary copy to detect stale files without overwriting evidence.
