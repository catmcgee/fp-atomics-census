# Diagnostic reproducer for vLLM issue #56900

`moe_compile_reproducer.py` is a fresh-process diagnostic for the compiled
Qwen1.5-MoE failure reported in [vLLM issue #56900](https://github.com/vllm-project/vllm/issues/56900).
It reproduces the original raw 16-prompt batch, model revision
`ec052fda178e241c7c443468d2fa1db6618996be`, TP=1, seed 0 and greedy sampling
(`temperature=0`, `max_tokens=32`, `logprobs=5`, `ignore_eos=True`). It does not
apply a chat template.

The original failure was vLLM 0.28.0 with torch 2.13.0+cu130 on an H100. A later
report did not reproduce it with the cu129 wheel on an H20, but changed the GPU
and driver too. This harness is for the controlled comparison that has not yet
been run: two environments on the same physical GPU and driver, changing the
wheel build only.

## What one run records

Every requested cell runs in a new Python process, with a newly created,
previously empty directory for all three compilation-cache roots:

```text
VLLM_CACHE_ROOT
TORCHINDUCTOR_CACHE_DIR
TRITON_CACHE_DIR
```

The script also sets `VLLM_DISABLE_COMPILE_CACHE=1`, `VLLM_PLUGINS=""`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_LOGGING_LEVEL=DEBUG` and a unique
`VLLM_DEBUG_DUMP_PATH`. It does not redirect `HF_HOME`, so each environment must
already have access to the exact pinned weights and tokenizer revision.
It refuses a version other than vLLM 0.28.0 unless `--expected-vllm` explicitly
names the intended version. The default accepts an official wheel suffix such
as `0.28.0+cu129`; an expected value with a suffix requires that exact suffix.

Each worker saves:

- actual imported torch, Triton and vLLM versions and module paths;
- installed `triton` and `tokenspeed-triton` distribution metadata, RECORD
  hashes, and the actual `triton` module-file hash;
- `collect_env.py` output, GPU UUID/driver, build configuration, the selected
  runner and resolved compilation, attention and kernel configs;
- cache snapshots before and after model construction and generation;
- vLLM FX/pattern debug dumps and live worker stdout/stderr logs;
- exact prompt and generated token ids, requested top-five logprobs and text for
  both identical in-process repeats;
- short-cycle, duplicate-prompt and repeat-identity summaries; and
- per-run failure details if engine construction or generation fails.

The controller writes `comparison.json`, including prompt-wise compiled versus
eager equality and the first different generated-token position. A failure is a
result, not an implicit fallback to a different runner or configuration. On a
per-cell timeout, Ctrl-C, or controller SIGTERM, it terminates and reaps the
worker's separate process group before returning control.

## Required environments

The [source and wheel review](moe_compile_SOURCE_REVIEW.md) records official
wheel URLs and hashes, isolated installation recipes, and the limits of each
diagnostic comparison. The recipes and GPU cells remain unrun in this follow-up.

Create two separate environments with **vLLM 0.28.0** and their intended torch
wheels. Do not install both CUDA variants into one environment. Confirm before
running that each imports the intended build:

```sh
/path/to/cu129/bin/python -c 'import torch, triton, vllm; print(torch.__version__, torch.version.cuda, triton.__version__, vllm.__version__)'
/path/to/cu130/bin/python -c 'import torch, triton, vllm; print(torch.__version__, torch.version.cuda, triton.__version__, vllm.__version__)'
```

Use the same `CUDA_VISIBLE_DEVICES` value for both commands, on one idle GPU and
the same loaded NVIDIA driver. Do not compare an H20/cu129 result with an
H100/cu130 result as though that isolates the wheel. The generated
`collect_env.txt` files are the record of whether the controlled variables were
actually held fixed.

At environment creation, retain the wheel URLs and SHA256 values from your
installer or a `pip install --report` file. The runtime evidence includes each
distribution's RECORD hash and `pip freeze --all`, which identifies the loaded
packages but cannot reconstruct the original wheel-download order.

## Minimum build comparison

Start with V2, compiled and graph-disabled versus eager and graph-disabled. This
matches the issue's compilation-versus-eager question while avoiding graph
capture during graph inspection:

```sh
CUDA_VISIBLE_DEVICES=0 /path/to/cu129/bin/python \
  probes/diagnostics/moe_compile/moe_compile_reproducer.py \
  --output-dir artefacts/moe_compile_cu129_v2_graphs_off \
  --runners v2 --compiles on off --graphs off

CUDA_VISIBLE_DEVICES=0 /path/to/cu130/bin/python \
  probes/diagnostics/moe_compile/moe_compile_reproducer.py \
  --output-dir artefacts/moe_compile_cu130_v2_graphs_off \
  --runners v2 --compiles on off --graphs off
```

Only after those cells are complete, add the graph-enabled and V1/V2 controls:

```sh
CUDA_VISIBLE_DEVICES=0 /path/to/cu130/bin/python \
  probes/diagnostics/moe_compile/moe_compile_reproducer.py \
  --output-dir artefacts/moe_compile_cu130_both_runners_graphs \
  --runners v1 v2 --compiles on off --graphs off on
```

V1 is intentionally an explicit environment selection via
`VLLM_USE_V2_MODEL_RUNNER=0`; it is not inferred from
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. The latter only keeps the engine core in the
launching process for the V1 code path. The original failure logged V2.

## Interface check

The harness was checked against the v0.28.0 source tag
`2cf0a6915ce544dc493a0990f2ea38d81601128a`:

- `VLLM_USE_V2_MODEL_RUNNER` accepts an explicit boolean and overrides the
  default model-runner choice;
- `VLLM_DEBUG_DUMP_PATH` overrides `CompilationConfig.debug_dump_path` and is
  made rank-aware by the engine; and
- `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` are read
  during process startup and are therefore set before importing vLLM.

The script requires the resolved runner, compilation mode and graph mode to
equal the requested cell, and records each assertion. In v0.28.0 this model is
a default V2 architecture when the environment override is unset, but the
diagnostic does not rely on that default.

`comparison.json` preserves the automatic backend-selection lines from each
worker. It labels a compiled/eager comparison as confounded when those recorded
selections are missing or differ; matching log lines do not prove kernel
identity.

## Limits

Fresh directory roots show that this harness avoids sharing the specified
vLLM/Inductor/Triton caches between its own cells. They cannot prove that no
other system-wide or driver cache exists. The result compares vLLM outputs to
vLLM eager output; it is not an external accuracy evaluation. It also preserves
the original one-GPU, TP=1, raw-prompt configuration, so it does not test other
models, tensor parallelism, drivers, GPUs or vLLM releases. Prefetch the pinned
model revision in both environments before timing a controlled comparison: vLLM
will otherwise use its normal Hugging Face download path when the revision is
not already present.
