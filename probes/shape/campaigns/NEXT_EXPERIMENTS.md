# Next E6 GPU batch

Status: prepared only. No GPU has been provisioned and none of the commands in
this document have been run.

The next batch should finish the small set of missing C2 observations before
starting E4, operator, or cross-SKU work. C2's full binary caches and pods are
gone. Its records, raw hook logs, comparisons, light archives, package lists,
and model-file digests survive. A fresh run can therefore be a cold replay of a
historical record, but it cannot be described as a restored-cache replay.

## Required stack and source

Historical C2 replays are allowed only after `next_campaign/preflight.py`
reports `compatible: true` against the record being replayed. The requirements
are:

| Field | Required value |
|---|---|
| Census execution source | git commit `7dc6f469726d3eec0719c98e9bf6458945b961af` in a detached worktree |
| Inspected vLLM source pin | `98dff2a81d747d1dba01a47f939f48c3526d4206` (v0.29.0 tag source) |
| Python | 3.12.3 |
| vLLM | 0.29.0 |
| torch / CUDA | 2.13.0+cu130 / 13.0 |
| Triton | 3.7.1 |
| FlashInfer | 0.6.18.post1, installed over vLLM's `0.6.18` requirement with `--no-deps`; retain the resulting `pip check` failure |
| NCCL runtime | 2.29.7 (`nvidia-nccl-cu13==2.29.7`) |
| GPU / driver / VBIOS | NVIDIA H100 80GB HBM3 / 580.126.09 / 96.00.DA.00.0C |
| Python distributions | all 210 entries exactly equal to the selected historical record's `env.json` |
| Qwen2.5 model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Qwen1.5-MoE revision | `ec052fda178e241c7c443468d2fa1db6618996be` |

Use a one-GPU pod for the TP=1 batch and a two-GPU pod for the TP=2 replay so
`gpu_count` also matches the corresponding record. Set
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_USE_V2_MODEL_RUNNER=0`, and
`VLLM_DISABLE_COMPILE_CACHE=1` as C2 did. TP=2 also requires
`NCCL_NVLS_ENABLE=0` on record and replay. Preserve the historical model-file
digests by hashing the new download and comparing the relevant `(repo,
revision, file, size, sha256)` rows to C2's `weights_digests.json`.

If any required field differs, do not append the result to C2's counts. Create
a new dated campaign and new records, with the differing stack stated in its
heading. A matching preflight still cannot establish an identical container
image or compiled binary: C2 did not retain the image identity or binary
caches. Keep that limitation in the report.

## Practical order

### 1. TP=1 completion and controls on one H100

After a reviewed gated runner exists, execute `next_campaign/jobs_tp1.txt` in
order. Every engine process gets a cold cache unless the job says `warm`.

1. Finish batch-invariant replays **b** and **c** against C2's historical
   record, with `VLLM_BATCH_INVARIANT=1`. This changes the present count from
   one to three cold replays. Do not generalise from the current 0/1 result.
2. Replay the historical default mixed record once cold. The old record was
   DIFFERS in 3/3 cold trials, so this is the inexpensive, contemporaneous
   negative control for the intervention below. If it is IDENTICAL, report
   that the new machine supplied no failing contemporaneous control; the older
   0/3 evidence remains valid but is not a same-session control.
3. Create a newly named mixed-schedule record with
   `TORCHINDUCTOR_DETERMINISTIC=1` and both `combo_kernels` and
   `benchmark_combo_kernel` false. Run one warm replay as a positive
   record/replay plumbing control, then three independent cold replays. This is
   a new arm; it must not reuse C2's plain `det_combo_off` record or counts.
4. Create a graph-disabled, uncompiled MoE record (`--no-compile --cudagraph
   0 --tag routing_diag`) and replay it once. The eager record and replay are
   the positive diagnostic: routing counts must be present on all 37 real
   forward passes. The retained graph-enabled C2 record is the negative
   diagnostic: its 37 FULL-graph passes have null counts because CUDA graph
   replay bypassed the Python router wrapper.

The minimum useful intervention result is: the warm deterministic mixed replay
passes, at least one default mixed replay in the combined old/new evidence is
DIFFERS, and all three deterministic mixed cold replays complete. Even then,
the result is an intervention-level association: determinism and two combo
settings changed together.

### 2. TP=2 completion on two H100s

After a reviewed gated runner exists, execute `next_campaign/jobs_tp2.txt`:
one cold repeat **c** of the historical TP=2 plain record with
`NCCL_NVLS_ENABLE=0`. This completes the planned a/b/c set.
Compare both ranks. Rank 0 remains the headline E6 verdict, while the rank-1
audit must report whether verdict, differing-row count, hidden hashes, full
logits, and argmax agree.

This job is last because it needs two GPUs for one missing observation. It can
run independently if a short-lived compatible two-GPU allocation is cheaper
than holding it for the TP=1 work.

## MoE routing diagnostic and claim boundary

The current hook clears `moe_counts` before each model execution. It wraps
`FusedMoERouter.select_experts` and also installs a pre-hook on the first
`FusedMoE` layer. In ordinary eager execution the router wrapper records actual
selected expert IDs as `moe_counts_source="actual_router"`; the pre-hook's
top-k result is only an estimate fallback. During FULL CUDA graph replay,
neither Python callback is invoked, so null means **unobserved**, not zero.

Validate the eager diagnostic with `next_campaign/validate_moe_routing.py`.
It loads the pinned model config at runtime, requires a non-empty integer count
vector on every real pass, requires a consistent number of experts, and checks
that each count sum is `total_scheduled * num_experts_per_tok`. Prefer
`actual_router` on every pass. If the source is only
`estimated_from_router_logits`, label it estimated and do not present it as an
actual launched-routing count. If any eager pass is null, the diagnostic fails.

Do not backfill the historical graph-enabled rows, compare null with zero, or
write a sentence such as “routing was unchanged” from those rows. A successful
eager pair validates that the wrapper works when Python executes. It does not
measure what the already captured graph selected, and disabling graphs changes
the execution mode. Graph-safe telemetry would need a tensor written inside
the captured graph or a vLLM-supported routed-expert output path, followed by a
separate validation campaign.

## Stop, evidence, and teardown rules

- Use a new dated evidence directory under the repository or another durable
  user project directory. Never leave the sole copy under `/tmp` or only on a
  pod. Sync each completed arm's raw result, engine log, cache-before/after
  listing, comparison, light archive, and SHA-256 manifest before starting the
  next paid arm.
- Keep the new deterministic mixed record's full binary cache archive in a
  durable non-temporary project path if a later restored-cache control is
  intended. Light archives cannot replace it.
- The queue must stop before launching a new arm when `STOP` exists, the local
  deadline has passed, preflight failed, a cache-clear check is nonzero, an arm
  timed out, a comparison is INVALID, disk space is below the configured
  reserve, or durable sync/digest verification failed.
- Stop the MoE sequence immediately if the eager record lacks routing counts;
  a replay cannot repair an uninstrumented record. Stop the TP=2 sequence on an
  NCCL smoke failure or any topology/environment mismatch.
- Set the provider termination deadline and launch the detached local
  `probes/ops/watch_pod.py` guard with the exact pod id and `census-` pod name.
  Its log must be a new durable JSONL path. The watchdog is a second guard, not
  a guarantee: it depends on the local machine staying awake and connected.
- After the queue exits, sync and verify all evidence, create `STOP`, explicitly
  remove the exact pod id, and query until that id is absent even if its state
  is `EXITED`. Record creation, deadline, deletion, verification, rate, and
  estimated cost in the campaign note.

## Prepared files

- `next_campaign/jobs_tp1.txt` and `jobs_tp2.txt`: exact C2 queue-format
  execution specifications. They are intentionally non-executable until a
  reviewed runner implements the failure and durable-sync gates above; the
  recovered runner must not be used unchanged.
- `next_campaign/preflight.py`: fail-closed historical-stack and model-digest
  check; it writes a JSON report.
- `next_campaign/validate_moe_routing.py`: offline routing diagnostic audit.
- `next_campaign/README.md`: staging and execution contract. The recovered C2
  queue is retained only as a format/reference; the provider lifecycle remains
  a separate operation.
