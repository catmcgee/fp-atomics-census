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
    (r"^fi_cutlass_fused_moe_fused_finalize_True", "flashinfer-0001, flashinfer-0002"),
    (r"^fi_cutlass_fused_moe_fused_finalize_False", "flashinfer-0001 (unfused path)"),
    (r"^sglang_fp8_blockwise_streamk", "sglang-0011"),
    (r"^deepgemm_bmk_bnk_mn", "DeepGEMM-0001, DeepGEMM-0002"),
    (r"^nccl_allreduce_", "vllm-0182, sglang-0269, flashinfer-0043, DeepEP-0012"),
    (r"^flashinfer_allreduce_fusion_", "flashinfer-0023, flashinfer-0026"),
    (r"^vllm_.*moe_wna16", "vllm-0014"),
    (r"^vllm_.*_lora_", "vllm-0018, vllm-0019"),
    (r"^vllm_.*GPTQ", "vllm-0010 (Marlin), class C linear layers"),
    (r"^vllm_", "class C rows (cuBLASLt, NCCL, cubins) for the dense path"),
    (r"^sglang_.*flashinfer_cutlass", "flashinfer-0001 through sglang-0271"),
    (r"^sglang_", "sglang-0008 (Marlin shapes), sglang-0264..0266"),
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
        for name, by_tag in sorted(reports.items()):
            verdicts = {t: r["verdict"] for t, r in by_tag.items()}
            inproc = "identical" if all(v == "bitwise-identical" for v in verdicts.values()) else "DIFFERS"
            hashes = {r["first_hash"] for r in by_tag.values()}
            fresh = "n/a (one run)" if len(by_tag) < 2 else ("identical" if len(hashes) == 1 else "DIFFERS")
            print(f"| {name} | {inproc} | {fresh} | {rows_for(name)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
