# H100 E6 follow-up evidence

This directory is the self-checking public evidence bundle for the 2026-09-16
H100 follow-up. It contains 11 ACK-verified completed labels: nine replay
comparisons and two record-only cache seeds. The audit never treats a
record-only arm as replay evidence.

## Reproduce the audit

From the repository root, run:

```bash
python3 probes/shape/campaigns/2026-09-16-h100-e6-followup/reproduce.py
```

The command is read-only. It verifies `SHA256.json`, every original handoff
manifest and ACK, immutable historical record trees, cache snapshots, teacher
forcing schedules, preflight environments, MoE router counts, both TP=2 ranks,
and the final run and lifecycle ledgers. It exports the comparator from frozen
Git commit `7dc6f469726d3eec0719c98e9bf6458945b961af` into a temporary directory and
regenerates all nine summaries byte for byte.

The only requirements are Python 3.10 or newer and a Git checkout containing
the frozen commit. The audit does not require a GPU, network access, or the
private work directories.

## Layout

- `e6/`, `logs/`, `handoff/`, and `evidence/` contain the TP=1 results.
- `results_tp2/`, `logs_tp2/`, `handoff_tp2/`, and `evidence_tp2/` keep the
  second host's TP=2 evidence separate.
- `generated_sources*/` contain deterministic source-and-IR-only archives.
  Full binary compiler caches remain in the durable private work root; their
  original sizes and hashes are recorded in
  `provenance/full_cache_archives.json`.
- `provenance/` preserves setup and recovery evidence, the failed TP=2 cache
  layout attempt, final runner ledgers, lifecycle controls, teardown and
  watchdog records, and the bounded collective-backend audit with its pinned
  vLLM source files.

Logs are stored as deterministic gzip streams. `SHA256.json` covers every
public file except itself.

## TP=2 interpretation boundary

The historical M3 record and warm replay initialised the FlashInfer `mnnvl`
workspace. The historical M4 cold replays and this follow-up cold replay fell
back to `trtllm`. No run traced per-call collective-kernel dispatch. The cold
TP=2 difference is therefore a valid observation of the complete run, while
its cause remains unresolved; it is not a fixed-collective-kernel comparison.
See `provenance/tp2_collective_backend_audit.md` for the bounded source audit.

## Maintainer staging

The public tree was created from durable, ACK-verified work roots with:

```bash
python3 probes/shape/campaigns/2026-09-16-h100-e6-followup/reproduce.py \
  --stage-from /path/to/2026-09-16-followup \
  --stage-tp2 /path/to/2026-09-16-followup/tp2-any-region
```

Staging accepts only labels with a verified ACK, checks each original handoff
byte against its immutable manifest, writes deterministic public artefacts,
and then runs the same read-only audit.
