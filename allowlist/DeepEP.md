# DeepEP: deterministic configuration allowlist

Repository `deepseek-ai/DeepEP` at `099d5f2bad488b9c534ea785062b12f2e91d1d41`.
Derived from `inventory/DeepEP.jsonl`.

| path | class-A sites | status | notes |
|---|---|---|---|
| Low-latency dispatch and combine (legacy `internode_ll.cu`, used by vLLM `deepep_low_latency` and SGLang's DeepEP dispatchers) | none. Receive-buffer slots are atomic tickets (DeepEP-0001, B-indirect); the combine sums a token's experts in top-k index order (`internode_ll.cu:1082-1120`) | allowed | the expert GEMM must be row independent, which holds for DeepGEMM masked grouped GEMM and the engines' Triton MoE kernels |
| Normal (high-throughput) dispatch and combine (`deep_ep/include`) | none. Counts are exact; `combine_reduce` reads slots in top-k order (DeepEP-0006) | allowed | the non-bypass combine path (more than two experts with bias) was not read line by line |
| Hybrid dispatch / Engram fetch | counters only | allowed on this reading | |
| IBGDA RDMA, barriers, timeouts | integer queue indices, locks and flags (DeepEP-0002..0005, 0008) | allowed | |
| Elastic buffer post-processing | DeepEP-0010 (A1: one writer per slot) | allowed | |
| NCCL in reference implementations | class C, tests only | not on the engine path | |

DeepEP moves data; it does not reduce floating-point values across
arrival order. The only ordering effect it introduces is the row order
inside expert buffers, which the downstream kernels do not depend on.
Under expert parallelism the reduction that matters is the per-token sum
over experts, and both combine implementations perform it in a fixed
order.
