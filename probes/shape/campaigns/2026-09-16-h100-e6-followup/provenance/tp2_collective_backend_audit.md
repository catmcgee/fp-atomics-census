# TP2 collective backend audit

**State: collective subbackend mismatch; cause unresolved.**

The original M3 TP2 record and same-host warm replay initialized FlashInfer's
`mnnvl` workspace. Both old M4 cold replays and the new cold replay first failed
`mnnvl` initialization with `CUDA_ERROR_ILLEGAL_STATE`, then initialized the
`trtllm` workspace. The cold comparisons therefore changed a collective
workspace subbackend relative to the record.

This is a material control break, not a cause finding. The logs establish which
workspace initialized. They do not record per-call dispatch, so they do not
prove that an `mnnvl` or `trtllm` kernel launched for any particular model
all-reduce.

## Observations

| cell | visible machine | initialized workspace | P2 | differing rows/passes |
|---|---|---|---|---:|
| old M3 record | H100 x2, driver 580.126.09, VBIOS 96.00.89.00.01 | `mnnvl` | record | n/a |
| old M3 warm replay | same UUIDs/host as record | `mnnvl` | IDENTICAL | 0/0 |
| old M4 cold a | H100 x2, driver 580.126.09, VBIOS 96.00.DA.00.0C | `trtllm` after error 401 | DIFFERS | 494/37 |
| old M4 cold b | same UUIDs/host as cold a | `trtllm` after error 401 | DIFFERS | 493/37 |
| new cold followup c | H100 x2, driver 580.126.09, VBIOS 96.00.89.00.01 | `trtllm` after error 401 | DIFFERS | 494/37 |

The new replay matches the record's installed-distribution map, pinned
revisions, torch/CUDA/cuDNN versions, driver, H100 model, SM, Python, relevant
environment and VBIOS family. Its GPU UUIDs differ. `NCCL_NVLS_ENABLE=0` is
present in the record and every replay. The new preflight returned
`compatible=true`, but it did not create a multicast workspace or assert the
FlashInfer subbackend. Matching VBIOS, kernel family and recorded software stack
were therefore insufficient to reproduce `mnnvl` availability.

All three fallback logs also contain earlier Torch symmetric-memory multicast
warnings. This association is consistent with unavailable multicast support or
state, but it does not distinguish topology, provider exposure, or CUDA driver
initialization state.

## What the vLLM source proves

The audited checkout is vLLM `v0.29.0` commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`.

- `flashinfer_all_reduce.py:31-40` checks only whether the FlashInfer API is
  importable.
- `flashinfer_all_reduce.py:135-160` makes `auto` prefer `mnnvl` and allows a
  single-node `trtllm` fallback.
- `flashinfer_all_reduce.py:163-226` creates and logs the common workspace.
- `flashinfer_all_reduce.py:421-472` applies per-tensor eligibility and buffer
  gates after workspace creation.
- `cuda_communicator.py:278-309` launches the FlashInfer allreduce only when
  `should_use_fi_ar` returns true; otherwise dispatch continues to other
  collectives.
- `cuda_communicator.py:212-276` explicitly describes its backend list as
  potential per-call choices.

The log text says “Allreduce norm fusion workspace,” but these runs resolved
`fuse_allreduce_rms=false`. The source says the workspace is shared by
standalone allreduce and non-quant norm fusion. The log must not be reported as
proof that the fused allreduce+RMS kernel ran.

## Interpretation update

The old M4 cold a/b and new cold followup c results are **mnnvl-record versus
trtllm-replay** comparisons. Their `P2=DIFFERS` verdicts remain valid complete-run
observations, but they do not isolate cold-process repeatability within one
collective implementation. The old M3 warm result is mnnvl-to-mnnvl and
`P2=IDENTICAL`; it also changes warmness/host reuse and cannot isolate a backend
effect.

Plausible explanations remain hypotheses: topology or exposed multicast
capability may vary between instances; CUDA initialization state may vary; and,
if eligible calls actually used FlashInfer, the two backends may reduce values
in different orders. The retained evidence does not choose among them.

A future isolating control should force the same
`VLLM_FLASHINFER_ALLREDUCE_BACKEND` in record and replay, fail when that backend
cannot initialize, and retain per-call backend counters or traces.

## Evidence integrity

Exact paths, SHA-256 digests, line ranges, hardware identities, outcomes and
source semantics are in `tp2_collective_backend_audit.json`.
