# flash-attention: deterministic configuration allowlist

Repository `Dao-AILab/flash-attention` at `1f7ce2f7cb503473559f3d44d575ae05b1ed8557`.
Derived from `inventory/flash-attention.jsonl`. A configuration is listed
when every class-A site it can reach is A1, A2 or A3 with the gate set to the
deterministic side.

## Forward pass (inference)

| API | kernels | class-A sites reachable | flags needed |
|---|---|---|---|
| `flash_attn_func`, `flash_attn_varlen_func`, `flash_attn_with_kvcache` (FA2, `csrc/flash_attn`) | `flash_fwd_kernel.h`, `flash_fwd_splitkv_kernel` plus combine | none found | none |
| `flash_attn_func`, `flash_attn_varlen_func`, `flash_attn_with_kvcache` (FA3, `hopper/`) | `flash_fwd_kernel_sm90.h`, `flash_fwd_kernel_sm80.h`, `flash_fwd_combine_kernel.h` | none found | none |

The scanner walked every file under `csrc/flash_attn` and `hopper/` and
found no floating-point atomic, PTX reduction or TMA reduce in any forward
kernel (`candidates/flash-attention.coverage.json`). Split-KV partial
outputs and log-sum-exp values are written to separate buffers and merged by
a second kernel that loops over splits in index order
(`hopper/flash_fwd_combine_kernel.h:360-370`; FA2
`flash_fwd_splitkv_combine_kernel`). The only atomics on the forward path
are the integer tile counter in the persistent scheduler
(`hopper/tile_scheduler.hpp:338`, class B) and the block-local integer sum
in `flash_prepare_scheduler.cu:152` (class B). Neither changes a
floating-point reduction order.

The forward allowlist is therefore unconditional: every forward
configuration (any head dim, causal or local, paged or contiguous KV,
varlen, GQA, split-KV with any number of splits) is free of order-dependent
atomics at this sha. The forward result still depends on the number of
splits chosen, which is a batch-invariance question rather than a
run-to-run one; a verifier must record `num_splits` (or the inputs to the
heuristic that picks it) alongside the batch composition.

## Backward pass (training only)

| build | head dim | GQA | deterministic=False (default) | deterministic=True |
|---|---|---|---|---|
| FA2 `csrc/flash_attn` | any | any | class A (`flash_bwd_kernel.h:678`) | A1 plus fixed-order `convert_dQ`: allowed |
| FA3 `hopper/` sm90 | < 256 | no | class A3, gate off (`mainloop_bwd_sm90_tma_gmma_ws.hpp:647`) | A2 via `dq_semaphore`: allowed |
| FA3 `hopper/` sm90 | < 256 | yes | as above plus dK/dV A3 (`epilogue_bwd.hpp:471,502`) | A2 via `dk_semaphore`, `dv_semaphore`: allowed |
| FA3 `hopper/` sm90 | 256 | any | class A (`mainloop_bwd_sm90_tma_gmma_ws.hpp:964,983,992`) | rejected by `flash_api.cpp:1370` |
| FA3 `hopper/` sm80-89 | any | any | class A (`mainloop_bwd_sm80.hpp:848`) | still class A: the sm80 mainloop has no barrier around the dQ atomicAdd |

No inference engine in this census calls a flash-attention backward kernel,
so the backward table matters for training reproducibility, not for
verifying inference.

## What a verifier should require

- Forward: nothing beyond recording the configuration Cankaya 2026 lists
  (software version, attention backend, batch composition, and the
  split-KV count where the heuristic is input dependent).
- Backward: `deterministic=True`, head dim below 256, and a sm90 build. On
  sm80-89 the flag does not make the FA3 tree's dQ accumulation
  deterministic at this sha; use the FA2 tree with `deterministic=True`
  instead.
