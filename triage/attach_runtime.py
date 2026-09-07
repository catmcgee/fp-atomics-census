"""Attach probe results to the inventory rows they bear on.

    python -m triage.attach_runtime [--results probes/results] inventory/*.jsonl

Reads every report under probes/results/<stack>/ and writes a
``runtime_evidence`` list into the rows named in ROWS below. The mapping is
by hand: a probe name (regex) to the rows it settles and a one-line note on
how. Rows not named keep whatever they had. Idempotent: existing evidence
for the same stack and probe is replaced.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# (probe name regex, [row ids], note)
ROWS: list[tuple[str, list[str], str]] = [
    (r"^index_add_default$", ["vllm-0045", "vllm-0046"], "index_add_ on contended indices, stock mode"),
    (r"^index_add_det$", ["vllm-0045", "vllm-0046"], "index_add_ with torch.use_deterministic_algorithms(True)"),
    (r"^scatter_add_default$", ["flash-attention-0026"], "scatter_add_ on contended indices, stock mode"),
    (r"^cumsum_float_default$", ["vllm-0049", "vllm-0050", "sglang-0031", "sglang-0035", "flash-attention-0027", "flashinfer-0048"], "float cumsum on CUDA, stock mode"),
    (r"^fi_top_p_renorm_detFalse$", ["flashinfer-0015"], "top_p_renorm_probs with the default is_deterministic=False"),
    (r"^fi_top_p_renorm_detTrue$", ["flashinfer-0015"], "top_p_renorm_probs with is_deterministic=True"),
    (r"^fi_top_k_renorm_multicta$", ["flashinfer-0016"], "top_k_renorm_probs on a 262144-wide vocabulary"),
    (r"^fi_radix_topk_order_detFalse$", ["flashinfer-0017", "flashinfer-0018", "vllm-0038", "vllm-0039", "vllm-0040", "vllm-0041", "vllm-0042", "vllm-0043", "vllm-0044", "sglang-0017"], "radix top-k output order with deterministic=False; the selected set is the same, its order is not"),
    (r"^fi_radix_topk_order_detTrue$", ["flashinfer-0017", "flashinfer-0018"], "radix top-k with deterministic=True"),
    (r"^fi_decode_fa2$", [], "FA2 decode template, no atomics expected"),
    (r"^fi_cutlass_fused_moe_autotuned_fused_finalize_True_topk4$", ["flashinfer-0001", "flashinfer-0002"], "fused finalize selected by autotuning, top-k 4: three or more red.add contributions per row"),
    (r"^fi_cutlass_fused_moe_autotuned_fused_finalize_True_topk2$", ["flashinfer-0001"], "fused finalize at top-k 2: two addends onto zero commute, so no order dependence is possible"),
    (r"^fi_cutlass_fused_moe_autotuned_fused_finalize_True_topk8$", ["flashinfer-0001"], "fused finalize at top-k 8 of 8 experts, 3 repeats"),
    (r"^fi_cutlass_fused_moe_autotuned_fused_finalize_False_topk8$", ["flashinfer-0001"], "unfused finalize (fixed-order sum), top-k 8"),
    (r"^cuBLASLt (mm|linear) (bf16|fp16)$", ["vllm-0180", "vllm-0181", "sglang-0264", "sglang-0265", "sglang-0266", "flashinfer-0040", "flashinfer-0041", "DeepGEMM-0009", "flash-attention-0014"], "cuBLASLt sweep over 60 projection shapes; the algorithm chosen is in the sweep JSON"),
    (r"^vllm_qwen2.5_7b_gptq_marlin$", ["vllm-0010"], "GPTQ model through Marlin with VLLM_MARLIN_USE_ATOMIC_ADD unset"),
    (r"^vllm_qwen2.5_7b_gptq_machete_fp16_x6$", ["vllm-0187"], "GPTQ model through Machete (stream-K, CUTLASS default reduction), fp16, 6 repeats"),
    (r"^vllm_zephyr_lora_split_k$", ["vllm-0018"], "LoRA adapter with the default split-K shrink"),
    (r"^vllm_zephyr_lora_batch_invariant$", ["vllm-0018"], "LoRA adapter with VLLM_BATCH_INVARIANT=1 (split_k = 1)"),
    (r"^vllm_qwen1.5_moe_gptq_marlin_x6$", ["vllm-0022", "vllm-0016", "marlin-0001"], "GPTQ MoE through the Marlin MoE backend, 6 repeats"),
    (r"^sglang_qwen2.5_7b_gptq_marlin_x6$", ["sglang-0008"], "GPTQ 7B through SGLang's Marlin at TP=1: every fused projection has n >= 2048, so should_use_atomic_add_reduce returns False and the stub is not exercised"),
    (r"^vllm_qwen3_8b_bf16_x12$", ["vllm-0180", "vllm-0181"], "dense bf16 default path, stock kernels, 12 repeats per process; differences, when present, are whole requests and track batch composition"),
    (r"^vllm_qwen3_8b_bf16_batch_invariant_x12$", ["vllm-0180", "vllm-0181"], "dense bf16 default path with VLLM_BATCH_INVARIANT=1, 12 repeats per process"),
    (r"^vllm_qwen2.5_7b_dense_bf16_one_seq_per_batch_x6$", ["vllm-0180", "vllm-0181"], "dense bf16 with max_num_seqs=1, stock kernels: batch composition fixed by construction"),
    (r"^sglang_qwen3_8b_bf16_no_overlap_x6$", ["sglang-0264"], "dense bf16 default path, radix cache off, overlap scheduler off, stock kernels"),
    (r"^sglang_qwen3_8b_bf16_deterministic_mode_x6$", ["sglang-0264"], "dense bf16 with --enable-deterministic-inference"),
    (r"^nccl_allreduce_rank\d+_default$", ["vllm-0182", "sglang-0269", "flashinfer-0043", "DeepEP-0012"], "NCCL all-reduce, two ranks, default algorithm and protocol"),
    (r"^nccl_allreduce_rank\d+_Tree$", ["vllm-0182", "sglang-0269", "flashinfer-0043", "DeepEP-0012"], "NCCL all-reduce, two ranks, NCCL_ALGO=Tree NCCL_PROTO=Simple, one channel"),
    (r"^flashinfer_allreduce_fusion_rank\d+$", ["flashinfer-0023"], "trtllm all-reduce fusion in plain all-reduce mode, two ranks"),
    (r"^marlin_gemm_atomicTrue_fp32reduce_", ["sglang-0008", "vllm-0010"], "Marlin GEMM with use_atomic_add=True and use_fp32_reduce=True, the pair SGLang's apply_gptq_marlin_linear passes, at an n < 2048, k >= 2048 shape"),
    (r"^marlin_gemm_atomicTrue_m", ["sglang-0008", "vllm-0010"], "Marlin GEMM on random 4-bit weights with use_atomic_add=True at an n < 2048, k >= 2048 shape"),
    (r"^sglang_qwen2.5_1.5b_gptq_marlin_x6$", ["sglang-0008"], "SGLang GPTQ 1.5B model: down_proj has n=1536, k=8960, so the stub turns atomic add on"),
    (r"^sglang_qwen2.5_7b_gptq_marlin_tp2_x6$", ["sglang-0008"], "SGLang GPTQ 7B at TP=2: o_proj shard has n=1792, k=3584"),
    (r"^vllm_qwen2.5_1.5b_gptq_marlin_atomic_add_x6$", ["vllm-0010"], "vLLM Marlin with VLLM_MARLIN_USE_ATOMIC_ADD=1 on the 1.5B GPTQ model"),
    (r"^vllm_qwen2.5_1.5b_gptq_marlin_x6$", ["vllm-0010"], "vLLM Marlin with the flag unset on the 1.5B GPTQ model"),
    (r"^vllm_qwen2.5_1.5b_gptq_marlin_one_seq_x6$", ["vllm-0010"], "vLLM Marlin, flag unset, one sequence per batch (composition fixed)"),
    (r"^vllm_qwen2.5_1.5b_gptq_marlin_atomic_add_one_seq_x6$", ["vllm-0010"], "vLLM Marlin with VLLM_MARLIN_USE_ATOMIC_ADD=1, one sequence per batch"),
    (r"^vllm_qwen2.5_1.5b_gptq_machete_one_seq_x6$", ["vllm-0187"], "vLLM Machete on the 1.5B GPTQ model, one sequence per batch"),
    (r"^vllm_qwen2.5_1.5b_gptq_x6$", ["vllm-0187"], "vLLM Machete on the 1.5B GPTQ model, stock scheduler"),
    (r"^sglang_qwen2.5_1.5b_gptq_marlin_deterministic_mode_x6$", ["sglang-0008"], "SGLang 1.5B GPTQ with --enable-deterministic-inference: the batch-invariant attention and sampler, Marlin unchanged"),
    (r"^marlin_gemm_atomicFalse_", ["vllm-0010"], "Marlin GEMM with use_atomic_add=False (fp32 global reduce)"),
    (r"^deepgemm_bmk_bnk_mn$", ["DeepGEMM-0001", "DeepGEMM-0002"], "deep_gemm.einsum bmk,bnk->mn, batch reduced across CTAs with float atomicAdd"),
    (r"^fi_decode_trtllm-gen$", ["flashinfer-0038", "sglang-0267", "vllm-0183"], "paged decode through the TensorRT-LLM cubin backend"),
    (r"^vllm_qwen3_8b_bf16_tp2_custom_allreduce_x6$", ["vllm-0182"], "TP=2 with vLLM's custom all-reduce, stock scheduler, 6 repeats"),
    (r"^vllm_qwen3_8b_bf16_tp2_nccl_x6$", ["vllm-0182"], "TP=2 with custom all-reduce disabled (NCCL, NVLS off), stock scheduler, 6 repeats"),
    (r"^vllm_qwen3_8b_bf16_tp2_custom_allreduce_one_seq_x6$", ["vllm-0182"], "TP=2 custom all-reduce, one sequence per batch"),
    (r"^vllm_qwen3_8b_bf16_tp2_nccl_one_seq_x6$", ["vllm-0182"], "TP=2 NCCL all-reduce (NVLS off), one sequence per batch"),
    (r"^vllm_qwen3_8b_bf16_tp2_batch_invariant_x6$", ["vllm-0182"], "TP=2 with VLLM_BATCH_INVARIANT=1"),
    (r"^vllm_mixtral_gptq_auto_x6$", ["vllm-0022", "vllm-0016", "marlin-0001"], "Mixtral 4-bit GPTQ through the Marlin MoE backend, stock scheduler"),
    (r"^vllm_mixtral_gptq_marlin_moe_one_seq_x6$", ["vllm-0022", "vllm-0016", "marlin-0001"], "Mixtral 4-bit GPTQ through the Marlin MoE backend, one sequence per batch"),
    (r"^sglang_qwen3_8b_bf16_tp2_x6$", ["sglang-0269"], "SGLang TP=2 bf16 dense, stock scheduler"),
    (r"^vllm_mixtral_gptq_moe_wna16_x6$", ["vllm-0014"], "4-bit GPTQ Mixtral through the moe_wna16 method, small batch, 6 repeats"),
    (r"^vllm_qwen1.5_moe_bf16_flashinfer_cutlass_x6$", ["flashinfer-0001"], "bf16 MoE through vLLM's flashinfer_cutlass MoE backend, 6 repeats"),
    (r"^sglang_qwen3_8b_bf16_trtllm_mha_x6$", ["sglang-0267"], "SGLang with the trtllm_mha attention backend, 6 repeats"),
    (r"^sglang_qwen3_8b_fp8_default_x6$", ["sglang-0011"], "FP8 block-quantised model on the auto backend for this GPU; the SM90 stream-K kernel is not involved"),
    (r"^vllm_qwen1.5_moe_bf16_x6$", ["vllm-0016"], "bf16 MoE through vLLM's default MoE backend on this GPU, stock scheduler"),
    (r"^vllm_qwen1.5_moe_bf16_one_seq_x6$", ["vllm-0016"], "bf16 MoE through vLLM's default MoE backend, one sequence per batch"),
    (r"^vllm_zephyr_lora_split_k_x6$", ["vllm-0018"], "LoRA adapter with the default split-K shrink, 6 repeats"),
    (r"^sglang_qwen3_8b_fp8_blockwise_default$", ["sglang-0011"], "FP8 block-quantised model; the release wheel routes Hopper to DeepGEMM, and its CUTLASS FP8 GEMM is SM120-only, so the stream-K kernel is not reached"),
]


def load_reports(results: Path) -> dict[str, dict[str, dict[str, dict]]]:
    """stack -> probe -> tag -> report"""
    out: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for stack_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        for f in stack_dir.glob("*.json"):
            if f.name.startswith("cublaslt_sweep_"):
                continue
            d = json.loads(f.read_text())
            if "probe" not in d:
                continue
            out[stack_dir.name][d["probe"]][d["env"].get("run_tag") or "-"] = d
    return out


def summarise(by_tag: dict[str, dict]) -> tuple[str, str, int]:
    inproc = "identical" if all(r["verdict"] == "bitwise-identical" for r in by_tag.values()) else "DIFFERS"
    hashes = {r["first_hash"] for r in by_tag.values()}
    fresh = "n/a" if len(by_tag) < 2 else ("identical" if len(hashes) == 1 else "DIFFERS")
    repeats = sum(len(r["runs"]) + 1 for r in by_tag.values())
    return inproc, fresh, repeats


def collapse_cublaslt(probes: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Fold cublaslt_<variant>_m.._n.._k.._<dtype> probes into one synthetic entry per variant and dtype."""
    groups: dict[str, list[tuple[str, dict[str, dict]]]] = defaultdict(list)
    rest: dict[str, dict[str, dict]] = {}
    for name, by_tag in probes.items():
        m = re.match(r"^cublaslt_(mm|linear)_m\d+_n\d+_k\d+_(\w+)$", name)
        if m:
            groups[f"cuBLASLt {m.group(1)} {m.group(2)}"].append((name, by_tag))
        else:
            rest[name] = by_tag
    for key, members in groups.items():
        verdict = "bitwise-identical" if all(r["verdict"] == "bitwise-identical" for _, bt in members for r in bt.values()) else "DIFFERS"
        tags: dict[str, dict] = {}
        for tag in ("a", "b"):
            if all(tag in bt for _, bt in members):
                tags[tag] = {"verdict": verdict, "first_hash": "".join(bt[tag]["first_hash"] for _, bt in members), "runs": [{}] * 3}
        rest[key] = tags or {"-": {"verdict": verdict, "first_hash": "x", "runs": [{}] * 3}}
    return rest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("probes/results"))
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)
    reports = load_reports(args.results)
    evidence: dict[str, list[dict]] = defaultdict(list)
    for stack, probes in reports.items():
        for name, by_tag in collapse_cublaslt(probes).items():
            inproc, fresh, repeats = summarise(by_tag)
            for pat, rows, note in ROWS:
                if re.search(pat, name):
                    for rid in rows:
                        evidence[rid].append({"stack": stack, "probe": name, "in_process": inproc, "fresh_process": fresh, "repeats": repeats, "note": note})
    touched = 0
    for f in args.files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for r in rows:
            if r["id"] in evidence:
                keep = [e for e in r.get("runtime_evidence", []) if (e["stack"], e["probe"]) not in {(x["stack"], x["probe"]) for x in evidence[r["id"]]}]
                r["runtime_evidence"] = keep + evidence[r["id"]]
                touched += 1
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"attached runtime evidence to {touched} rows from {sum(len(p) for p in reports.values())} reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
