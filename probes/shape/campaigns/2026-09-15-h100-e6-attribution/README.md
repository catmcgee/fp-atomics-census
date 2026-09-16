# C1 evidence bundle

Read the [campaign note](../2026-09-15-h100-e6-attribution.md) for scope and limitations. This directory is independent of the generated E6 results table.

- `results_pod/`: all 353 original merged result files, including raw hook and forcing logs, runtime environments, cache listings, run provenance and original summaries.
- `artefacts_m1/`, `artefacts_m2/`: all 40 original light archives plus the original per-file manifests. Full binary caches are omitted. Read archive members in place; archive names include absolute pod paths and should not be extracted into a live filesystem.
- `logs_m1/`, `logs_m2/`: original run, queue, bootstrap, text-check and pod-side digest logs.
- `remote/`: original campaign scripts and initial job files. Queue `ARM_START` records determine what actually ran; job files do not contain all later edits.
- `src/`, `COMMIT.txt`: the original analysis source snapshot and its census commit. The source is historical evidence; it is not the current runnable GPU harness.
- `arms_table.txt`, `table.json`, `pod_compare.txt`: original derived arm join and comparison output.
- `reproduce.py`, `reproduction.json`: portable CPU reanalysis and its verified output. The script removes copied replay and boundary summaries before regeneration and never changes the original records.
- `SHA256.json`: SHA-256 of every preserved evidence file and source file. Reproduction tooling and its output are outside this original-evidence manifest.

Run from the repository root:

```sh
.venv/bin/python probes/shape/campaigns/2026-09-15-h100-e6-attribution/reproduce.py
```

Expected: 729 files verified, 32 byte-identical regenerated summaries, 20 equal-choice IDENTICAL comparisons, 12 unequal-choice DIFFERS comparisons and four reference output classes. The original analyser digest is `fd45a1b297e9e377e7478a6ba322a58cc4c08b65b088f4f6635983274cd4dcd1`.

For a separate row-level check, `row_diff.py DIR_A DIR_B` compares two runs' rank-0 hook logs, including pairs of replays that the E6 provenance guard deliberately refuses to compare as record/replay. Use the record and any replay under the `_rmsnorm_custom` arm to recover the six-row residual; compare its replay directories with one another to recover zero differences. In the earlier `probes/shape/results/e6` plain Qwen2.5-7B bf16 arm, comparing `replay_cold` with `replay_cold2` recovers the five-row residual. These are observations on different vLLM releases.

The 0.28.0 record versus cold replay and C1's `_autotune_off` record versus replay both leave fifteen rows equal: slot 4 of passes 0 through 12, and slot 12 of passes 5 and 6. This positional match is corroboration; missing 0.28.0 record-side artefacts prevent direct kernel attribution.
