# C2 follow-up runner and execution contract

These files prepared the missing C2 GPU work. The bounded follow-up has now
executed; its evidence is in
[`2026-09-16-h100-e6-followup.md`](../campaigns/2026-09-16-h100-e6-followup.md).
They do not provision or remove a pod. Read `../campaigns/NEXT_EXPERIMENTS.md`
before using this preparation for a future run.

The runner retained with the executed evidence has SHA-256
`1a1bad6ca3cdb810e1a36f5aa4f5464f7a31a21d9ac4a58070b2cd3dd49e0de6` and is
not rewritten. The malformed-ACK handling in the current source postdates that
execution; it is a future-run hardening rather than a change to the archived
evidence.

## Source and evidence layout

Keep two separate trees on the GPU host:

- `SOURCE_ROOT`: a detached worktree at census commit
  `7dc6f469726d3eec0719c98e9bf6458945b961af`. Install
  `SOURCE_ROOT/probes/shape/shape_hook_pkg` and execute its `run_e6.py`.
- `EVIDENCE_REPO`: a current checkout containing
  `campaigns/2026-09-15-h100-e6-modern`. Write the new dated campaign below
  this checkout or another durable user project path. Do not write the only
  copy to `/tmp`.

The old commit predates the committed C2 evidence, which is why one checkout
cannot serve both purposes.

Stage only these immutable historical record directories into the new
`RESULTS_ROOT`, preserving their arm names and `record/` child:

| Batch | Source below `2026-09-15-h100-e6-modern` | Destination arm |
|---|---|---|
| TP=1 | `results_m1/e6/...prefix0_batch_invariant/record` | same arm name |
| TP=1 | `results_m1/e6/...prefix0_mixed/record` | same arm name |
| TP=2 | `results_m3/e6/...tp2...prefix0/record` | same arm name |

Use `cp -a` or `rsync -a` into an empty destination and record a SHA-256
manifest before execution. Do not copy old derived summaries into the new
campaign. The new MoE eager and deterministic mixed arms create new records in
`RESULTS_ROOT`.

## Preflight

After installing the exact C2 runtime and downloading the pinned models, run:

```sh
python probes/shape/next_campaign/preflight.py \
  --source-root /project/census-c2-source \
  --reference-env /project/census/probes/shape/campaigns/2026-09-15-h100-e6-modern/results_m1/e6/Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0/record/env.json \
  --required-gpus 1 \
  --expected-vbios 96.00.DA.00.0C \
  --reference-weights /project/census/probes/shape/campaigns/2026-09-15-h100-e6-modern/logs_m1/weights_digests.json \
  --actual-weights /project/campaign/weights_digests.json \
  --required-model Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \
  --required-model Qwen/Qwen1.5-MoE-A2.7B-Chat@ec052fda178e241c7c443468d2fa1db6618996be \
  --required-env VLLM_DISABLE_COMPILE_CACHE=1 \
  --required-env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  --required-env VLLM_USE_V2_MODEL_RUNNER=0 \
  --output /project/campaign/preflight_tp1.json
```

For TP=2 select the historical TP=2 `env.json`, require two GPUs, use M3's
weight digest file, and require only Qwen2.5. M3's recorded VBIOS is
`96.00.89.00.01`; `96.00.DA.00.0C` in the TP=1 example belongs to M1 and is
not compatible evidence for TP=2. The script exits nonzero on any mismatch.
Also require `NCCL_NVLS_ENABLE=0`. Retain its JSON report even on failure.

## Queue specification (not directly executable)

The job files are reviewed execution specifications in the recovered C2
`c2_queue.sh` format. **Do not run them with the recovered runner unchanged.**
That runner continues after some arm/comparison failures and does not prove a
durable sync before the next arm, so it does not implement the stop contract in
`NEXT_EXPERIMENTS.md`.

The bounded runner below implements the reviewed execution contract:

- `ROOT` becomes the absolute durable `RESULTS_ROOT`.
- `LOGS` and `ART` become new durable directories below the campaign root.
- Full cache archives, if enabled, go to a durable project directory rather
  than `/tmp` or ephemeral local storage.
- `cd` targets `SOURCE_ROOT/probes/shape`.
- preflight, smoke, cache emptiness, process exit, comparison validity, free
  disk, durable sync, and digest verification are hard launch gates;
- any failed gate creates `STOP` and exits before another engine process;
- the runner copies and verifies each arm off the pod before continuing.

Retain the runner copy and its digest with the results. Run the job list only
after preflight passes and a CUDA/NCCL smoke succeeds. The queue's deadline and
`STOP` checks are launch gates; they do not terminate a process already in an
arm, so set the deadline far enough before the provider deadline to cover the
1500-second per-arm timeout, sync, digest verification, and explicit deletion.

When a handoff manifest is ready, the sync operator must copy and verify every
listed file before publishing the acknowledgement. Write and close
`ACK_DIR/LABEL.json.pending`, then atomically rename it to
`ACK_DIR/LABEL.json`. A direct streamed write to the final ACK path can be read
while incomplete by the polling runner. Keep the runner digest frozen for the
active queue. The current malformed-JSON hardening is a later runner revision.

The orchestration deliberately does not contain provider commands. Use the
provider deadline plus the detached `probes/ops/watch_pod.py` guard described
in `NEXT_EXPERIMENTS.md`.

## Exact environment installation

`env.json`, rather than the retained `pip-freeze.txt`, is the authority for the
210-distribution preflight comparison. The freeze omits 11 distributions that
the record's installed-distribution map contains. Generate the exact 209
wheel pins (everything except the local editable hook) from the selected TP=1
record:

```sh
python probes/shape/next_campaign/c2_constraints.py \
  --env-json probes/shape/campaigns/2026-09-15-h100-e6-modern/results_m1/e6/Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0_batch_invariant/record/env.json \
  --constraints /project/campaign/c2-209.txt \
  --plan /project/campaign/c2-install-plan.json
python3.12 -m venv /project/c2-venv
/project/c2-venv/bin/pip install --no-index --find-links /project/wheelhouse --no-deps -r /project/campaign/c2-209.txt
/project/c2-venv/bin/pip install -e /project/census-c2-source/probes/shape/shape_hook_pkg
```

The editable installation must come from the clean detached frozen source
worktree. It restores `shape-hook==0.1.0`, yielding the required 210 entries.
Keep the expected `pip check` failure: vLLM 0.29.0 declares
`flashinfer-python==0.6.18`, while C2 deliberately retained
`0.6.18.post1`. Any additional `pip check` failure is a failed installation
gate.

## Bounded runner and handoff protocol

`run_followup.py` is the reviewed replacement for the recovered queue. It
does not create, extend, or remove a pod. It uses direct argument execution
(never `eval`), accepts only the reviewed environment variables in the job
file, and runs `SOURCE_ROOT/probes/shape/run_e6.py` only after verifying the
frozen source, a successful preflight JSON, a successful supplied CUDA/NCCL
smoke command, free disk reserve, deadline margin, and `STOP` absence.

First stage the historical `record/` directories described above into the new
`RESULTS_ROOT/e6/ARM/record` locations. Then create the immutable baseline and
retain a copy plus SHA-256 of the runner:

```sh
python /project/evidence/probes/shape/next_campaign/run_followup.py \
  --initialize --results-root /project/campaign/results
```

For each batch, create an empty durable acknowledgement directory and run the
queue from the current evidence checkout. The cache root is deliberately below
the durable campaign path. Every cold arm gets a never-before-used directory;
the runner records all enumerated cache roots before and after execution.

```sh
mkdir -p /project/campaign/acks /project/campaign/results/cache
python /project/evidence/probes/shape/next_campaign/run_followup.py \
  --run \
  --source-root /project/census-c2-source \
  --results-root /project/campaign/results \
  --jobs /project/evidence/probes/shape/next_campaign/jobs_tp1.txt \
  --preflight-report /project/campaign/preflight_tp1.json \
  --smoke-command "python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'" \
  --cache-root /project/campaign/results/cache \
  --ack-dir /project/campaign/acks \
  --deadline-epoch PROVIDER_DEADLINE_EPOCH \
  --arm-timeout 1500 --ack-timeout 1800 --deadline-buffer 600 \
  --min-free-gb 30 --archive-full-cache
```

Use a TP=2 CUDA/NCCL smoke command and `jobs_tp2.txt` on the two-GPU pod. The
runner validates rank 1 with the frozen comparator after the TP=2 replay and
records the rank-0/rank-1 observation separately. It also runs the routing
validator after both graph-disabled MoE arms and stops if either has missing or
estimated counts.

After each arm, the runner writes `handoff/LABEL/manifest.json`, prints its
SHA-256, and waits. The operator copies every listed file off the pod, verifies
the listed digests, then writes this file in `ACK_DIR`:

```json
{
  "schema": 1,
  "label": "LABEL",
  "handoff_manifest_sha256": "the manifest sha256",
  "verified_manifest_sha256": "the same manifest sha256"
}
```

The runner refuses an old acknowledgement, an incorrect digest, a failed
comparison, changed record, timeout, cache gate, failed audit, low disk, or
deadline margin; each failure creates `RESULTS_ROOT/STOP` before another engine
process can start. The acknowledgement must be written only after the copied
manifest verifies, so it is the durable-sync launch gate rather than a claim
made by the pod itself.

## Post-run checks

Every replay must run `run_e6.py --compare` and retain the output. An INVALID
comparison stops the batch. Compare the TP=2 rank-1 traces separately with the
frozen `compare_replay` function and report them alongside rank 0. C2's archived
`audit_ranks.py` illustrates the method, but its paths and seven-comparison count
are fixed to C2; it cannot directly validate this new one-replay batch.
For the MoE diagnostic:

```sh
python probes/shape/next_campaign/validate_moe_routing.py \
  /project/campaign/results/e6/Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs0_prefix0_routing_diag/record \
  --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --revision ec052fda178e241c7c443468d2fa1db6618996be \
  --output /project/campaign/moe_routing_record.json
```

Repeat for `replay_warm`. The validator is intentionally offline with respect
to execution: it reads the saved hook rows and the already downloaded pinned
model config. It does not infer counts for graph-enabled runs.
