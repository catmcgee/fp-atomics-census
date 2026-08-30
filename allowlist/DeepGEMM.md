# DeepGEMM: deterministic configuration allowlist

Repository `deepseek-ai/DeepGEMM` at `1f6f3f378920ccb5cc036ef43eb3f5972e921713`.
Derived from `inventory/DeepGEMM.jsonl`.

| API | class-A sites | status | notes |
|---|---|---|---|
| `fp8_gemm_nt/nn/tn/tt`, `bf16_gemm_*` (dense, sm90 and sm100) | none when `c` is not given; with accumulation the epilogue is a TMA reduce-add with one writer per tile (DeepGEMM-0003..0005, A1) | allowed | no split-K exists in these kernels (only the tf32 pre-norm kernel has a K split, and it is out of the engines' use) |
| `m_grouped_fp8_gemm_nt_contiguous`, `m_grouped_fp8_gemm_nt_masked`, bf16 variants (MoE GEMMs used by vLLM and SGLang) | none | allowed | row independent; the engines' alignment tickets are therefore order invariant |
| `k_grouped_fp8_gemm_nt/tn_contiguous` (weight gradients) | DeepGEMM-0006 (A1) | allowed | training side |
| `einsum bmk,bnk->mn` (`bmk_bnk_mn`) | DeepGEMM-0001, 0002 (A): batch blocks add into one output with `atomicAdd` (sm90) or TMA reduce-add (sm100) | **forbidden** | not called by vLLM or SGLang at their pinned shas |
| Mega-MoE (`sm100_bf16_mega_moe`, `sm100_fp8_fp4_mega_moe`) | counters B-indirect with an unread combine (DeepGEMM-0007) | **gate** | not used by the engines at their pinned shas |
| `cublaslt_gemm_*` | class C (DeepGEMM-0009) | probe | |
| DeepGEMM cubins shipped through FlashInfer (`flashinfer/deep_gemm.py`) | class C | probe | a different build than the source scanned here |

The kernels the inference engines actually call (`fp8_gemm_nt` and the
m-grouped MoE GEMMs) contain no order-dependent atomics. They are not batch
invariant (tile configuration depends on shape), which is Cankaya 2026's
"record the batch size" requirement rather than a run-to-run problem.
