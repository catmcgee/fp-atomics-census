"""Tabulate probe results by inventory row.

    python probes/report.py [probes/results/<stack>]

Reads every JSON report under the results directory, maps probe names to
the inventory rows they settle (ROWS below), checks that RUN_TAG=a and
RUN_TAG=b reports of the same probe carry the same first-run hash (the
fresh-process test), and prints one Markdown table per stack.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# probe name prefix (regex) -> inventory rows it bears on
ROWS = [
    (r"^cublaslt_", "vllm-0180, vllm-0181, sglang-0264..0266, flashinfer-0040, flashinfer-0041, DeepGEMM-0009"),
    (r"^index_add_", "vllm-0045, vllm-0046"),
    (r"^scatter_add_", "flash-attention-0026"),
    (r"^cumsum_float_", "vllm-0049, vllm-0050, sglang-0031, sglang-0035, flash-attention-0027, flashinfer-0048"),
    (r"^bincount_", "(no row)"),
    (r"^fi_top_p_renorm_", "flashinfer-0015"),
    (r"^fi_top_k_renorm_", "flashinfer-0016"),
    (r"^fi_radix_topk_", "flashinfer-0017, flashinfer-0018, vllm-0038..0044, sglang-0017"),
    (r"^fi_decode_trtllm", "flashinfer-0038, sglang-0267, vllm-0183"),
    (r"^fi_decode_", "(FA2 template, no atomics expected)"),
    (r"^fi_cutlass_fused_moe(_autotuned)?_fused_finalize_True", "flashinfer-0001, flashinfer-0002 (fused finalize, red.add per expert contribution)"),
    (r"^fi_cutlass_fused_moe(_autotuned)?_fused_finalize_False", "flashinfer-0001 (unfused finalize path)"),
    (r"^sglang_fp8_blockwise_streamk", "sglang-0011"),
    (r"^deepgemm_bmk_bnk_mn", "DeepGEMM-0001, DeepGEMM-0002"),
    (r"^nccl_allreduce_", "vllm-0182, sglang-0269, flashinfer-0043, DeepEP-0012"),
    (r"^flashinfer_allreduce_fusion_", "flashinfer-0023, flashinfer-0026"),
    (r"^vllm_.*moe.*batch_invariant", "MoE with VLLM_BATCH_INVARIANT=1 (batch-invariant Triton MoE)"),
    (r"^vllm_.*dense.*batch_invariant|^vllm_qwen3_8b_bf16_batch_invariant", "dense path with VLLM_BATCH_INVARIANT=1 (batch composition removed as a variable)"),
    (r"^vllm_.*one_seq_per_batch", "dense path with max_num_seqs=1 (fixed batch composition, stock kernels)"),
    (r"^vllm_.*moe.*marlin", "vllm-0022 (Marlin MoE, use_atomic_add constant False), vllm-0016 (moe_align tickets), marlin-0001"),
    (r"^vllm_.*moe", "vllm-0014 (moe_wna16 CUDA kernel), MoE routing rows"),
    (r"^vllm_.*machete", "vllm-0187 (Machete stream-K, CUTLASS default reduction mode)"),
    (r"^vllm_.*dense", "class C linear layers incl. the bias path (cuBLASLt split-K COMPUTE_TYPE), no inventory A row"),
    (r"^vllm_.*lora_batch_invariant", "vllm-0018 with VLLM_BATCH_INVARIANT=1 (split_k = 1)"),
    (r"^vllm_.*lora", "vllm-0018 (LoRA shrink split-K atomic add)"),
    (r"^vllm_.*gptq_marlin", "vllm-0010 (Marlin, VLLM_MARLIN_USE_ATOMIC_ADD unset)"),
    (r"^vllm_.*(GPTQ|gptq)", "vllm-0187 (Machete stream-K), class C linear layers"),
    (r"^vllm_", "class C rows (cuBLASLt, cubins) on the dense default path"),
    (r"^sglang_.*fp8_blockwise_cutlass", "sglang-0011 (stream-K Nondeterministic when k > 3n)"),
    (r"^sglang_.*fp8_blockwise_default", "sglang-0011 not selected (DeepGEMM auto backend)"),
    (r"^sglang_.*gptq", "sglang-0008 (Marlin atomic add for n < 2048, k >= 2048)"),
    (r"^sglang_.*deterministic_mode", "dense path with --enable-deterministic-inference (batch-invariant kernels)"),
    (r"^sglang_.*no_overlap", "dense path with --disable-overlap-schedule (stock kernels)"),
    (r"^sglang_", "sglang-0264 (cuBLASLt) on the dense default path"),
]


def rows_for(name: str) -> str:
    for pat, rows in ROWS:
        if re.search(pat, name):
            return rows
    return "?"


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv] or sorted(p for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.exists() else []
    if not dirs:
        print("no results yet")
        return 0
    for d in dirs:
        reports = defaultdict(dict)
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("cublaslt_sweep_"):
                continue
            data = json.loads(f.read_text())
            if "probe" not in data:
                continue
            tag = data["env"].get("run_tag") or "-"
            reports[data["probe"]][tag] = data
        if not reports:
            continue
        env = next(iter(next(iter(reports.values())).values()))["env"]
        print(f"\n### {d.name}\n")
        print(f"GPU {env.get('gpu')} x{env.get('gpu_count')}, driver {env.get('driver')}, torch {env['torch']}, CUDA {env['cuda']}, "
              + ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v) + "\n")
        print("| Probe | In-process | Fresh process | Rows |")
        print("|---|---|---|---|")
        sweeps: dict[tuple[str, str], list] = defaultdict(list)
        for name, by_tag in sorted(reports.items()):
            verdicts = {t: r["verdict"] for t, r in by_tag.items()}
            inproc = "identical" if all(v == "bitwise-identical" for v in verdicts.values()) else "DIFFERS"
            hashes = {r["first_hash"] for r in by_tag.values()}
            fresh = "n/a (one run)" if len(by_tag) < 2 else ("identical" if len(hashes) == 1 else "DIFFERS")
            m = re.match(r"^cublaslt_(mm|linear)_m(\d+)_n(\d+)_k(\d+)_(\w+)$", name)
            if m:
                first = next(iter(by_tag.values()))
                entries = first.get("extra", {}).get("heuristic", {}).get("entries", [])
                algo = entries[0] if entries else {}
                sweeps[(m.group(1), m.group(5))].append((inproc, fresh, int(algo.get("numSplitsK", 1)), algo.get("reductionScheme", "-"), int(m.group(2))))
                continue
            print(f"| {name} | {inproc} | {fresh} | {rows_for(name)} |")
        for (variant, dtype), rows in sorted(sweeps.items()):
            n = len(rows)
            ok_in = sum(1 for r in rows if r[0] == "identical")
            ok_fresh = sum(1 for r in rows if r[1] == "identical")
            splitk = [r for r in rows if r[2] > 1]
            schemes = sorted({r[3] for r in splitk})
            ms = sorted({r[4] for r in splitk})
            detail = f"{n} shapes; split-K chosen for {len(splitk)} (M in {ms}, schemes {schemes})" if splitk else f"{n} shapes; no split-K chosen"
            print(f"| cuBLASLt {variant} {dtype}: {detail} | {ok_in}/{n} identical | {ok_fresh}/{n} identical | {rows_for('cublaslt_')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
