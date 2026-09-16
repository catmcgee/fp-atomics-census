# H100 causal follow-up, 16 September 2026

This campaign follows the [C2 completion campaign](2026-09-16-h100-e6-followup.md).
It uses new references to separate compiler settings and collective selection,
and separate diagnostics to observe MoE routing during CUDA-graph replay.

## Questions and controls

The mixed-schedule experiment tests all eight combinations of Inductor's
`TORCHINDUCTOR_DETERMINISTIC`, `combo_kernels` and `benchmark_combo_kernel`.
Each runnable configuration owns a new record, one warm replay and three cold
replays on the same GPU. Every comparison uses 77 forwards and 555 request
rows, the frozen teacher-forcing harness, the pinned model files and the same
210 installed distributions. The benchmark flag is interpreted conditional on
combo kernels being enabled.

The separate TP=2 experiment requires a new record with a fixed FlashInfer
`trtllm` workspace backend on both ranks. Python dispatch observations cover
graph construction and capture; they are not a trace of every GPU launch
during graph replay. This removes the known `mnnvl`/`trtllm` reference mismatch
from the earlier campaign when the configuration passes its runtime gates.

The routing diagnostic uses vLLM's native selected-expert capture. It compares
three fresh processes: graph execution without telemetry, graph execution with
telemetry, and eager execution with telemetry. Checks cover actual dispatch,
selected IDs changing during decoding, count consistency, eager agreement, and
whether telemetry preserves consumed tokens and hidden/logit hashes.

## Evidence discipline

Execution uses frozen census commit
`7dc6f469726d3eec0719c98e9bf6458945b961af`. New orchestration and instrumentation
sources are separately retained and hashed. Each successful arm is copied to
durable storage and every manifest file is hash-verified before an atomic ACK
allows the next arm. Failed starts remain failures rather than replay results.
Full binary caches remain in the private durable archive; public derivatives
retain textual generated source/IR and the original cache-file hashes.

Warm replays reuse their record's cache. Cold replays begin with nine fresh
enumerated cache roots. That distinction is a statement about the recorded
roots, not every possible driver or library cache. A few matching cold replays
are observations, not a guarantee for other runs, hardware or schedules.

The independent vLLM 0.28.0 MoE localisation uses the earlier failing build,
the original 16 prompts and one generated token in each boundary scan. Its boundary recorder is
checked against an uninstrumented compiled process before compiled/eager
differences are interpreted. The smaller token limit tests the prefill and
first-logit failure; later decoding and short cycles remain covered by the
[earlier controlled build experiment](../../diagnostics/moe_compile/2026-09-16-h100-builds/README.md).

## Results

### Mixed-schedule settings

`d` is Inductor determinism, `c` is combo kernels and `b` is combo benchmarking.
Every completed warm replay was IDENTICAL. Cold counts below compare each cell
against its own fresh reference, removing the earlier reference-provenance
difference.

| d | c | b | IDENTICAL cold replays | Differing rows in cold 0 / 1 / 2 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0/3 | 5 / 5 / 552 |
| 0 | 0 | 1 | 1/3 | 5 / 5 / 0 |
| 0 | 1 | 0 | 1/3 | 552 / 552 / 0 |
| 0 | 1 | 1 | 1/3 | 555 / 0 / 555 |
| 1 | 0 | 0 | 3/3 | 0 / 0 / 0 |
| 1 | 0 | 1 | 3/3 | 0 / 0 / 0 |
| 1 | 1 | 0 | 3/3 | 0 / 0 / 0 |
| 1 | 1 | 1 | No valid record | Startup rejected non-vetted combo benchmarking |

The seven runnable cells produced seven records, seven warm replays and 21
cold replays. All 28 replay summaries reproduce byte for byte with the frozen
comparator. The eighth configuration was attempted and failed before producing
a valid reference; its [failure evidence](2026-09-16-h100-causal-followup/factorial-unsupported/)
is retained. The runner stopped as designed. The audit calls the replay matrix
`partial` because this cell has no successful record; the attempted eight-cell
experiment is complete.

Disabling both combo options alone did not remove cold differences. Enabling
Inductor determinism matched all three cold replays in each runnable setting.
When combo kernels remain enabled, combo benchmarking must be disabled for
this tested configuration to start. The benchmark flag had no active combo
path when combo kernels were disabled. The b=0/b=1 rows with combo kernels disabled therefore do not identify a
benchmark-flag effect.

[Per-kernel analyses](2026-09-16-h100-causal-followup/analysis/) associate the
`d0_c1_b1` cell's two divergent cold replays with changed RMSNorm reduction block
sizes. Its matching cold replay changed only a pointwise block size. These are
recorded associations, not isolated compiler-choice interventions.

### Actual routing during CUDA-graph replay

The [routing validation](2026-09-16-h100-causal-followup/routing/) passed all
three fresh-process conditions. Two prompts generated eight tokens each, with
19 and 17 real forwarded tokens respectively. Native selected IDs covered all
24 layers, four choices per token from 60 experts: 1,824 and 1,632 selected IDs.
Each request had seven distinct decode-step route signatures.

Graph routing matched eager routing exactly. Enabling telemetry preserved the
graph control's output tokens, consumed tokens and hidden/logit hashes. The
execution witness confirmed mode-0 compilation with real FULL graph dispatch,
versus NONE in the eager condition. All conditions explicitly disabled
asynchronous scheduling so the CPU hook observed materialised token IDs.

This validates native routing capture in this diagnostic configuration. It
does not retroactively fill the old Python hook's null fields, test compiled
MoE execution, or establish invariance across other scheduling modes.

### Offline reproduction

```sh
.venv/bin/python probes/shape/causal_followup/factorial_audit.py \
  --audit-public probes/shape/campaigns/2026-09-16-h100-causal-followup/factorial \
  --output /tmp/factorial-audit.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  probes/shape/causal_followup/routing_audit.py \
  probes/shape/campaigns/2026-09-16-h100-causal-followup/routing
```

The factorial auditor verifies exact checksum coverage before executing its
frozen comparator. The routing auditor recomputes counts, digests, dispatch,
decode diversity and instrumentation preservation from the retained raw IDs
and hook rows.

### Remaining campaign work

Compatible TP=2 capacity remains unavailable after multiple bounded allocation
attempts. Its fixed-backend runner is prepared but has no measured result.

The [MoE operation investigation](../../diagnostics/moe_compile/2026-09-16-h100-localisation/README.md)
narrows the earliest recorded numerical difference to post-attention RMSNorm.
Precision emulation and matched residual rounding do not remove the failure.
Disabling Inductor buffer reuse reproduces all 512 tokens of the earlier failing
compiled run exactly. The cause of the degenerate output remains unresolved.


## Lifecycle and unexecuted TP=2 work

The factorial pod was deleted at 18:57 UTC and the MoE/routing pod at 19:11 UTC;
provider absence was checked twice for each. One earlier two-GPU allocation
exposed driver 570 and was rejected before inference, then deleted. Subsequent
CUDA-13-compatible H100 SXM and PCIe requests failed allocation, including both
final attempts after releasing the MoE pod. The [capacity records](2026-09-16-h100-causal-followup/capacity/)
retain those failures. Prepared TP=2 code is not evidence that its experiment ran.

Account balance decreased by $8.25 during this batch, below its $20 cap. This
includes unrelated pre-existing storage charges of $0.056/hour. No campaign
compute remains; those unrelated resources were left alone.

Two routing setup failures preceded the successful validation: an unavailable
public `LLM.shutdown` method and asynchronous placeholders observed by the CPU
hook. The successful diagnostic uses the engine shutdown path and explicitly
synchronous scheduling. Their [failure records](../../diagnostics/moe_compile/2026-09-16-h100-localisation/operational-failures/)
remain separate from the measured validation.
