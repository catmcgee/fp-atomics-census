# vLLM compilation-mode discriminator

**Result:** vLLM 0.28.0 compilation mode 2 (`DYNAMO_TRACE_ONCE`) reproduces
the Qwen1.5-MoE batch-dependent output failure without vLLM's custom
`VllmBackend`. Mode 1 (`STOCK_TORCH_COMPILE`) is rejected by Model Runner V2,
so this campaign does not supply a stock `torch.compile` result.

The run used one H100 80 GB, driver 580.126.09, torch 2.13.0+cu130, vLLM
0.28.0, Triton 3.7.1, V2, TP=1 and CUDA graphs disabled. The model and tokenizer
were pinned to `Qwen/Qwen1.5-MoE-A2.7B-Chat` revision
`ec052fda178e241c7c443468d2fa1db6618996be`. Each arm ran in a fresh process
with initially empty vLLM, Inductor and Triton cache roots. The shared pinned
model cache was reused. Each successful arm generated 32 greedy tokens twice
for the same 16-request batch: eight distinct raw prompts followed by exact
duplicates in the same order.

## Results

| Mode | Resolved execution | Duplicate pairs | Short cycles | Two repeats | Outcome |
|---|---|---:|---:|---|---|
| 0 `NONE` | mode 0, graphs `NONE`, V2 | 8/8 | 0/16 | identical | coherent eager reference |
| 1 `STOCK_TORCH_COMPILE` | rejected during `VllmConfig` validation | n/a | n/a | n/a | V2 does not support stock `torch.compile` |
| 2 `DYNAMO_TRACE_ONCE` | mode 2, backend `inductor`, graphs `NONE`, V2 | 0/8 | 8/16 | identical | failure reproduced |
| 3 `VLLM_COMPILE` | mode 3, backend `inductor`, graphs `NONE`, V2 | 0/8 | 9/16 | identical | failure reproduced |

Modes 2 and 3 each differ from mode 0 for all 16 prompts. Their first
difference from mode 0 is generated token 0 for 15 prompts and token 1 for the
remaining prompt. Modes 2 and 3 are token-identical to each other for 8 of 16
prompts, so mode 2 reproduces the failure class rather than every exact mode-3
token.

The post-initialisation mode-2 configuration has `backend="inductor"`,
`ir_enable_torch_wrap=false`, no splitting operations, and every recorded
vLLM pass disabled. Source review confirms that modes 1 and 2 resolve their
backend directly, while only mode 3 constructs `VllmBackend`. Therefore the
symptom does not require vLLM's custom backend, compile cache, piecewise
compilation, shape specialisation or custom passes. Mode 2 still uses vLLM's
compilation wrapper, a full-graph Dynamo trace, guard removal and model/operator
code, so this result does **not** attribute the defect to PyTorch or Inductor
alone.

The next high-information control is mode 2 with the `eager` Dynamo backend
against the recorded mode-2 Inductor result. If capture with backend `eager`
is coherent, the remaining boundary is Inductor code generation; if it fails,
the boundary moves earlier into Dynamo capture, the compile wrapper or traced
model/operator semantics. It should be run with a fresh mode-0 reference and
the same duplicate-prompt test.

## Evidence and verification

[`results/status.json`](results/status.json) and
[`results/comparison.json`](results/comparison.json) retain the controller's
incremental matrix. Each arm contains exact outputs or the validation failure,
resolved configuration, package and module provenance, backend-selection logs,
loaded CUDA libraries and fresh-cache snapshots. Compiler files are retained
under each successful arm's cache directory.

The controller emitted an immutable handoff after every arm and would not start
the next arm until the handoff had been copied locally, rehashed and acknowledged
with the exact manifest digest. The final audit verifies 1,412 file entries:

```sh
.venv/bin/python probes/diagnostics/moe_compile/moe_compile_reproducer.py \
  --audit-only \
  --output-dir probes/diagnostics/moe_compile/2026-09-16-h100-compile-modes/results
```

The first launch is retained under [`attempt-1-path-failed/`](attempt-1-path-failed/).
It reached no model output: FlashInfer warmup could not find the installed
`ninja` executable because the SSH command invoked the virtual-environment
Python by absolute path without adding its `bin` directory to `PATH`. Its
single failed handoff is independently auditable. The final matrix exported
the pinned environment's `bin` directory and passed `pip check` plus a CUDA
allocation smoke test before starting.

## Lifecycle

The pod was created as `census-vllm-56900-modes` at $3.49/hour with an original
90-minute provider termination request and an independent local watcher. Both
guards failed to act while the client machine slept; the original window
elapsed before the first PATH failure was acknowledged. After the user asked
to continue, a new 30-minute watcher was started. The final matrix completed
inside that extension, and the exact pod ID was explicitly deleted and observed
absent twice. The account balance changed by $7.28 over the whole interval,
including the unrelated $0.056/hour storage charge. After deletion only that
storage charge remained. Lifecycle responses and watcher logs are retained in
[`lifecycle/`](lifecycle/).
