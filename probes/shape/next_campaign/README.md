# Prepared C2 follow-up batch

These files prepare the missing C2 GPU work. They do not provision or remove a
pod. Read `../campaigns/NEXT_EXPERIMENTS.md` before running them.

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
weight digest file, and require only Qwen2.5. The script exits nonzero on any
mismatch. Also require `NCCL_NVLS_ENABLE=0`. Retain its JSON report even on
failure.

## Queue specification (not directly executable)

The job files are reviewed execution specifications in the recovered C2
`c2_queue.sh` format. **Do not run them with the recovered runner unchanged.**
That runner continues after some arm/comparison failures and does not prove a
durable sync before the next arm, so it does not implement the stop contract in
`NEXT_EXPERIMENTS.md`.

When funding and compatible hardware are available, a bounded runner must be
prepared and reviewed with these properties:

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

The orchestration deliberately does not contain provider commands. Use the
provider deadline plus the detached `probes/ops/watch_pod.py` guard described
in `NEXT_EXPERIMENTS.md`.

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
