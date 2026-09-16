# C1: cold E6 attribution on H100, 15 September 2026

**In C1, ten of sixteen cold replays of one compiled bf16 record were IDENTICAL and six were DIFFERS. Every difference coincided with a different archived Inductor reduction configuration.** This is a strong association within these observations. C1 did not intervene on a single autotune choice, so it does not establish that the reduction block size was the sole operative cause.

This campaign ran vLLM 0.29.0 on torch 2.13.0+cu130. Its [raw evidence and reproduction script](2026-09-15-h100-e6-attribution/) are separate from `probes/shape/results/e6`, whose vLLM 0.28.0 records generate the README table. Arm names overlap between campaigns; copying these records into that results directory would mix versions and overwrite evidence. The conclusions below concern C1 only, including its four online FP8 replays. Later C2 observations must be assessed separately.

## Stack and source provenance

| Field | Recorded value |
|---|---|
| Runtime | vLLM 0.29.0; torch 2.13.0+cu130; CUDA 13.0; triton 3.7.1; NCCL 2.29.7 |
| Other packages | FlashInfer 0.6.18.post1; Python 3.12.3; 210 installed distributions equal between machines |
| GPU | H100 80GB HBM3 SXM; driver 580.126.09; VBIOS 96.00.DA.00.0C |
| Model | Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| Procedure | TP=1; plain arrival script; greedy sampling; 32 maximum tokens; prefix caching off; V1 runner; async scheduling off |
| Compiled arms | `CompilationMode` 3; `FULL_AND_PIECEWISE` graphs; FlashAttention 3 |
| Census source | `8a9faa7428ed0a2671ac0b00a40de76262ceeb9d`, archived before execution and unedited |
| Analyser SHA-256 | `fd45a1b297e9e377e7478a6ba322a58cc4c08b65b088f4f6635983274cd4dcd1` |

The vendored analysis files in `src/` are byte-identical to that census commit. They preserve the original analyser digest rather than regenerating historical results with today's hook. The repository's current source pins describe source inspection targets; the per-run `env.json` files describe this campaign's installed runtime.

vLLM required `flashinfer-python==0.6.18`. Installing 0.6.18.post1 over it with `--no-deps` left exactly that unsatisfied requirement in `pip check`. The attention backend was FlashAttention 3; no launched-kernel profiler was collected, so this does not establish that no FlashInfer kernel ran elsewhere.

## Machines and caches

All records were made on M1 in AP-IN-1. M2 was in US-CA-2. The recorded GPU UUIDs and serials, PCI buses, CPUs and host kernels differ; both GPUs have the same model, driver and VBIOS. M1 has an Intel Xeon Platinum 8480+ and host kernel 6.8.0-106-generic; M2 has a Xeon Platinum 8462Y+ and kernel 6.17.0-1008-nvidia. `machine_1.txt`, `machine_2.txt` and the raw run metadata retain the identity evidence. M1's public address matches an earlier 0.28.0 record pod, so that earlier pod may have used the same server. This does not affect the M1 versus M2 comparison.

The queue cleared the vLLM, Inductor, Triton, NVIDIA, DeepGEMM, TileLang and CUTLASS caches and temporary locations before cold runs and refused to launch unless its file count was zero. Independent harness listings report empty caches in 36 of 40 engine runs. The four warm runs are the warm replay, the boundary rebuild and two restored replays: 241 files in the harness listing and 470 to 472 in the wider queue listing. The queue listed caches after all 40 runs, reporting 43 to 701 files; the 39 successful runs also wrote harness listings, reporting 39 to 356 files. The failed deterministic record wrote neither `cache_after_run.json` nor `env.json`. No engine log contains `Directly load AOT compilation`.

Each restored replay began with cleared caches, then restored the record's full cache archive. Per-file manifests show all 173 record artefacts preserved unchanged on both machines, plus the same twelve additional sampler artefacts. The full binary caches are not included here; their original archive digests and per-file manifests are retained. The original light archives, including generated code, kernel sources and tuning configurations, are included byte for byte.

## Results

| Configuration | Cold replays | IDENTICAL | DIFFERS |
|---|---:|---:|---:|
| Default compiled bf16, M1 | 6 | 3 | 3 |
| Default compiled bf16, M2 | 10 | 7 | 3 |
| Default compiled bf16, total | 16 | 10 | 6 |
| Online FP8 per-token | 4 | 4 | 0 |
| Uncompiled bf16, FULL graphs | 1 | 1 | 0 |
| Compiled bf16, combo kernels off | 2 | 2 | 0 |
| Compiled bf16, `--custom-ops +rms_norm` | 4 | 0 | 4, each 6 of 512 rows |
| Compiled bf16, three autotune keys false | 1 | 0 | 1, 497 of 512 rows |

Two additional restored replays, one per machine, and the warm reference replay were IDENTICAL. These are warm controls and are not counted as cold replays. The boundary rebuild was DIFFERS in all 16 rows but used a differing cold replay's caches, so it cannot isolate the boundary effect. The deterministic-mode record failed to compile.

All 31 replay comparisons meet their requirements. The thirty-second comparison is the boundary summary, which carries `verdict_P2_boundary` and no `requirements_met` field. All 32 have empty validation errors. Teacher forcing preserved the recorded continuation tokens; output differences concern logprobs. In each of the six 512-row differences the unforced argmax changed in one row, pass 3 slot 0. It changed in none of the four six-row differences.

### Reduction signatures and output classes

The reference graph has three separately autotuned fused add-and-RMSNorm reduction instances. They have hidden width 3,584 and reduction size hint 4,096. Their configuration hashes begin `d899c7b6` (five loads, one store), `4e10c47a` (seven loads, one store, in-place output) and `e0f56c5c` (seven loads, two stores). `XBLOCK=1`, `num_warps=16` and `num_stages=1` stay fixed; `R0_BLOCK` varies.

| `R0_BLOCK` in that hash order | Reference-arm runs, including record and warm controls | Rows differing from the record |
|---|---:|---:|
| 4096, 4096, 4096 | 14 | 0 of 512 |
| 2048, 4096, 4096 | 3, across both machines | 512 of 512 |
| 2048, 4096, 2048 | 2, across both machines | 512 of 512 |
| 2048, 2048, 4096 | 1 | 512 of 512 |

These twenty runs form four reduction signatures and four distinct digests of all 512 rows' `h` and `logits_h`, in one-to-one correspondence. Runs with a shared signature agree row for row across machines. Over all 32 comparisons, twenty have equal archived choices and are IDENTICAL; twelve have unequal choices and are DIFFERS. One equal comparison is vacuous: the uncompiled arm has no autotune choices.

Every arm's full set of choices is archived: ten in the reference, autotune-off and custom-op arms; nine in online FP8; eight with combo kernels off; one before the deterministic record failed; zero without compilation. Counts of extern calls remain `addmm: 2, mm: 6` in every compiled bf16 run, and empty in the online FP8 and uncompiled runs. Kernel-family counts remain equal within every arm. These coarse observables do not prove equal generated machine code or equal launched kernels.

### The small effect and the earlier campaign

The custom-op arm's record selected 2048 only for the in-place `4e10c47a` instance. All four cold replays selected 4096 for all three and differ from their record in the same six decode rows: pass 9 slot 0, pass 14 slot 1, pass 17 slot 8, pass 24 slot 6, pass 27 slot 6 and pass 29 slot 14. The replays agree with one another in all 512 rows across both machines.

The last five positions are exactly the five differences between two cold replays of the earlier 0.28.0 plain record. Separately, C1's autotune-off arm changed both `_2` instances and differed in 497 of 512 rows. Its fifteen surviving rows match the fifteen surviving rows of the earlier plain record versus its cold replay. This corroborates a similar mechanism across versions; the earlier campaign retained no record-side artefacts, so neither that arm nor its other 493 to 555-row effects can be assigned directly to a particular reduction instance. C1 does not establish that every earlier difference is explained.

## What the controls establish

Both machines produced both verdicts. Restoring the record's caches reproduced its rows once on each machine, and independently compiled runs sharing a reduction signature also shared all row hashes. Machine identity or cache emptiness alone therefore does not determine C1's verdict. These observations do not exclude every machine-dependent or execution-dependent source of variation. Every record was made on M1, and the restored M2 result has only one replay.

The earlier suggestion that `norm_quant` fusion removes RMSNorm from Inductor is contradicted by C1's generated code: `fuse_norm_quant` is false and the online FP8 graph still contains RMSNorm reductions fused with quantisation. C1's four FP8 replays chose equal configurations and were IDENTICAL. This small sample establishes neither FP8 immunity nor whether quantisation absorbs upstream differences. These counts are specific to C1, not a summary of all 0.29.0 evidence.

## Switches and failed controls

1. Setting `max_autotune`, `max_autotune_gemm` and `coordinate_descent_tuning` false did not disable the reduction benchmarking. They were already false in this configuration. vLLM 0.29.0's `set_inductor_config` sets the first and third from environment defaults that are on for single-size compile ranges; C1 requested no `compile_sizes` and used range endpoint 16384, so that path did not run. This is not a claim that those keys are always false at vLLM defaults.
2. `TORCHINDUCTOR_DETERMINISTIC=1` failed with the observed default combo-kernel settings. The traceback runs through `Scheduler._init`, `create_combo_kernel_nodes`, `speedup_by_combo_kernel`, `benchmark_fused_nodes`, `TritonScheduling.benchmark_codegened_module`, `Benchmarker.benchmark`, `benchmark_gpu` and `may_ban_benchmarking`. This is verified from the recorded traceback and pinned vLLM source, not a torch 2.13 checkout. C1's combo-off arm did not set deterministic mode. Combining both switches and using `VLLM_BATCH_INVARIANT=1` were untested candidates in C1, not ruled-out remedies. Later campaigns must report their own results.
3. `--custom-ops +rms_norm` retained the native IR priority and the same Inductor reductions. On 0.29.0, apart from the batch-invariant branch, `RMSNorm.forward_cuda` calls `forward_native`; both lower through `ir.ops.rms_norm` and `ir.ops.fused_add_rms_norm`. The custom-op switch therefore did not perform the intended removal from Inductor. Its six-row result is an observed compile association, not a controlled single-choice intervention.

The source facts above were checked at vLLM source commit `98dff2a81d747d1dba01a47f939f48c3526d4206` (tag v0.29.0). This source pin is distinct from installed-runtime provenance.

## Procedure and bounds

Archive record-side as well as replay-side compiled artefacts, and report repeated cold replay counts. Restoring the original caches is a demonstrated reproduction control on these two machines, not a universal guarantee or a proof that cache transport is the only remedy. `VLLM_DISABLE_COMPILE_CACHE` only governs vLLM's cache and does not disable Inductor's per-kernel benchmarking.

The reference IDENTICAL rate was 10 of 16, with a Wilson 95 per cent interval of roughly 39 to 82 per cent. Assuming independent draws at the observed rate, three repeats expose at least one DIFFERS with probability 75.6 per cent and seven with probability 96.3 per cent. Three IDENTICAL repeats are compatible with this observed failure rate and do not establish determinism. No p-value is used as a claim of causal isolation.

One model, one revision, one arrival script, one GPU model, one release and TP=1 bound this campaign. Generated code, PTX and cubins were not compared byte for byte, and launched-kernel profiling and cuBLASLt algorithm identifiers were omitted. Whole-cache restoration changes more than a single tuning choice. TP=2, MoE and mixed-prefill controls were not run. The boundary run inherited A3c's differing caches and is confounded.

## Smoke checks and deviations

The smoke test ran on M1 only. M2's compatibility evidence is the equality of all 210 installed distributions and run environments. The original `smoke.txt` ends in a FAIL for point 10: the test expected `enable_batch_sharded_sampling=False`, while the source default is `None`. `smoke_point10.txt` records the corrected rerun. The row-hashing check inspected only the first forward pass. TP=2 and MoE smoke checks were not applicable.

Self-comparison correctly returned INVALID with 39 errors: 37 forcing-accounting errors, the run-id check and the record-digest check. The original self-compare output was not in the pod result listing; the audit reran it separately. Smoke lacked a positive DIFFERS control; independent `row_diff.py` checks supplied one during audit. Five arm directories were checked for decoded text: continuations had 32 tokens, at least twelve distinct tokens and no cycle of period eight or less.

The archived job files are the initially written versions. They omit some later jobs and still list a deterministic M2 replay that never launched; the `QUEUE ARM_START` lines in the queue logs are the record of what actually ran. A shell variable-expansion error interrupted the first queue after its record; the complete record was retained and remaining jobs restarted. Combo-off and online FP8 controls and extra repeats were added during the campaign. A MoE model was prefetched but not run. The failed deterministic record has no environment file; its setting is attested by the queue. Future campaigns should capture `TORCHINDUCTOR_*` explicitly.

## Cost and removal

At 3.49 USD per GPU-hour, the rejected driver-mismatch pod's 30 seconds cost 0.03 USD; M1's 54 min 29 s cost 3.17 USD; M2's 53 min 2 s cost 3.09 USD. Total lifetime was 6,481 seconds, approximately 108 minutes, and GPU cost about 6.29 USD against an 8 USD cap. Partially settled rates including disk were 1.0 to 1.6 per cent above the base, implying approximately 6.35 to 6.38 USD; the campaign-time balance movement, after baseline spend, was consistent with about 6.34 USD. These were estimates, not final settled billing. Termination limits were 60 minutes ahead for M1 and 59 minutes for M2. The audit verified all three campaign pods removed after digest-checked sync; no campaign pod remained in the final listing.

## Offline reproduction

From the repository root, with the existing analysis environment:

```sh
.venv/bin/python probes/shape/campaigns/2026-09-15-h100-e6-attribution/reproduce.py
```

This verifies the evidence manifest, copies inputs to a temporary directory, removes both replay and boundary summaries, reruns all 32 comparisons with the archived analyser and requires byte-identical summaries. It reads original archive members without extracting absolute paths, joins the queue's executed arms to their configurations and rederives the four row-digest classes. `reproduction.json` records the checked result. No GPU, vLLM installation or network is needed for this analysis; use the repository's CPU analysis dependencies.

`SHA256.json` covers the original raw records, logs, light archives, pod-side manifests and source snapshot. The original result trees merged to 353 files; 64 shared record files were identical across pods. The audit checked 182 M1 and 235 M2 result files and 63/58 artefact files against pod-side digests. Full binary archives are omitted here and cannot be restored from this bundle; the light archives suffice for the reported configuration association. Account billing payloads and unrelated pod listings are omitted. The bounded cost/removal account above reports the audit's findings rather than claiming those omitted payloads are independently reproducible here.
