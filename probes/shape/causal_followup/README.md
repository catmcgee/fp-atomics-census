# Causal follow-up experiments

This batch follows the [16 September completion campaign](../campaigns/2026-09-16-h100-e6-followup.md).
New recordings and diagnostic outputs belong to a separate dated campaign.
Historical data is never rewritten to incorporate new instrumentation.

## Mixed-schedule settings

`factorial.py` specifies all eight combinations of:

- `TORCHINDUCTOR_DETERMINISTIC=0/1`;
- `combo_kernels=false/true`;
- `benchmark_combo_kernel=false/true`.

Each cell has a fresh reference, one warm replay and three independent cold
replays. The default cell and previously successful combined cell run first.
The remaining cells complete the setting combinations. Every record and replay
uses the frozen census source, model revision, prompts and mixed schedule.

Torch 2.13 implements the environment variable in `torch._inductor.config`:
it enables a mode that skips device benchmarking when it may affect numerics.
This is separate from `torch.use_deterministic_algorithms`, which is not changed
by this experiment. Both combo settings are explicitly supplied and checked in
the resolved configuration. The benchmark setting may have no effect when
combo kernels are disabled; its effect must be interpreted conditional on the
combo setting, with retained compiler artefacts as supporting evidence.

The runner verifies the 210-entry installed-distribution map, fresh enumerated
cache roots, immutable reference trees and valid frozen-comparator results.
Every completed arm is copied to durable local storage and hash-verified before
an atomic acknowledgement permits another arm. A failed configuration stops the
invocation. After its logs and failure record are archived, the operator may
use `--cells` and a new output directory to execute the remaining cells.
Unsupported configurations are reported as failures, never successful replays.

```sh
python probes/shape/causal_followup/factorial.py --plan
```

Run commands and exact source snapshots are retained alongside experimental data.
The source worktree is frozen at `7dc6f469726d3eec0719c98e9bf6458945b961af`;
the orchestration code is separately hashed and retained. Small cold-repeat
counts remain observations, not estimates of universal reliability.

## Fixed TP=2 collective backend

[The TP=2 harness](tp2_README.md) creates a new record with a forced FlashInfer
workspace backend and uses that same setting in every replay. Instrumentation
retains source hashes, actual workspace initialisation, Python dispatch choices
and application graph modes on both ranks. It distinguishes Python decisions
made while constructing a graph from a trace of actual GPU kernel launches.

## Graph-safe MoE routing

[The routing harness](routing_README.md) uses the native routed-expert output
path in vLLM 0.29.0. It checks live graph dispatch, actual selected IDs, changed
inputs, eager agreement and whether enabling telemetry changes output tensors.
It does not infer routing from the old Python hook's null fields.

## Compiled MoE localisation

[The MoE boundary diagnostic](../../diagnostics/moe_compile/localise/README.md)
uses vLLM 0.28.0 on the reproducing CUDA 13.0 stack. It compares uninstrumented
and instrumented compiled outputs, then locates the earliest observed
compiled/eager boundary difference. A targeted tensor capture can supply an
operator-level replay. Boundary differences alone are not a kernel cause.
