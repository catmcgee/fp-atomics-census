# Fixed TP=2 collective experiment

**Status, 16 September:** prepared and CPU-tested, but unexecuted. Compatible
two-GPU allocations failed; see the [campaign record](../campaigns/2026-09-16-h100-causal-followup.md).

This harness creates a new TP=2 record and replays it in one warm and three
fresh cold processes. It never uses the historical record whose FlashInfer
workspace backend differed from the cold replays.

The primary experiment explicitly sets
`VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`. Each arm fails before handoff
unless both ranks attest all of the following:

- the pinned vLLM collective source bytes match 0.29.0 commit
  `98dff2a81d747d1dba01a47f939f48c3526d4206`;
- a workspace was requested and created as `trtllm`;
- every observed TP all-reduce dispatcher call selected `FLASHINFER`;
- each rank retained all 37 shape-hook forward records, with compilation mode
  3 and CUDA graph dispatch in `FULL` or `PIECEWISE` mode.

The collective counters cover Python dispatcher decisions made while graphs
are built or captured. CUDA graph replay bypasses Python, so the evidence also
retains the per-forward graph dispatch counter; it does not claim to be a
CUPTI trace of every kernel launch encoded in a graph.

## Runtime

- Exactly two NVIDIA H100 80GB GPUs
- Frozen census source `7dc6f469726d3eec0719c98e9bf6458945b961af`
- vLLM 0.29.0 official wheel whose two collective-dispatch source files match
  reviewed commit `98dff2a81d747d1dba01a47f939f48c3526d4206` byte for byte
- FlashInfer 0.6.18.post1 / PyTorch 2.13.0+cu130
- `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`
- 60 minutes is a conservative allowance for setup plus five engine starts.

## Launch

Transfer this `causal_followup` directory outside the clean frozen source
checkout. Set absolute paths and a hard deadline:

```bash
SOURCE=/workspace/fp-atomics-census
PREP=/workspace/causal_followup
RESULTS=/workspace/tp2-fixed-trtllm-results
CACHE=/workspace/tp2-fixed-trtllm-cache
ACK=/workspace/tp2-fixed-trtllm-acks
DEADLINE_EPOCH=1789585200

python "$PREP/tp2_runner.py" \
  --initialize \
  --source-root "$SOURCE" \
  --results-root "$RESULTS" \
  --backend trtllm \
  --jobs "$PREP/tp2_jobs.json"

mkdir -p "$ACK" "$CACHE"

python "$RESULTS/runner/tp2_runner.py" \
  --run \
  --source-root "$SOURCE" \
  --results-root "$RESULTS" \
  --backend trtllm \
  --cache-root "$CACHE" \
  --ack-dir "$ACK" \
  --deadline-epoch "$DEADLINE_EPOCH" \
  --arm-timeout 1800 \
  --ack-timeout 900 \
  --deadline-buffer 300
```

The runner first executes a two-rank `torchrun` CUDA/NCCL smoke. After each
arm it prints `HANDOFF_READY` with a manifest SHA-256 and waits for an exact
durable-storage ACK before starting another engine. A failure writes `STOP`
and prevents any later launch from the results root.

The retained plugin bundle includes its reviewed entry-point metadata. The
runner adds that immutable directory to each child process's `PYTHONPATH`; no
post-initialisation package install may mutate the retained runner manifest.

An optional `mnnvl` run uses a separate results, cache and ACK root and changes
both `--backend` arguments to `mnnvl`. Explicit selection disables vLLM's
single-node fallback. If multicast workspace creation is unavailable, that
run fails closed and must remain a setup failure rather than comparison data.
