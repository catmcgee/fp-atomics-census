"""Generate and check README tables from the inventory and recorded results."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

from triage.attach_runtime import collapse_cublaslt, load_reports, summarise
from triage.summary import load, markdown_table, summarise as inventory_summary

ROOT = Path(__file__).resolve().parents[1]


def render() -> dict[str, str]:
    summary = inventory_summary(load(sorted((ROOT / "inventory").glob("*.jsonl"))))
    inventory = f"{summary['rows']} inventory records; {summary['sites']} sites; {summary['definition_only']} helper definitions.\n\n" + markdown_table(summary)
    coverage = ["| Repository | Eligible scanner candidates | Explicitly linked | Unresolved |", "|---|---|---|---|"]
    for path in sorted((ROOT / "triage/dispositions").glob("*.jsonl")):
        rows = list(map(json.loads, path.read_text().splitlines()))
        linked = sum(r["status"] == "linked" for r in rows)
        coverage.append(f"| {path.stem} | {len(rows)} | {linked} | {len(rows) - linked} |")
    runtime = ["| Stack | Probe | In process | Fresh process | Evaluations |", "|---|---|---|---|---|"]
    patterns = [r"^cuBLASLt", r"^flashinfer_allreduce_fusion_rank0$", r"^sglang_qwen3_8b_bf16(_trtllm_mha)?_x6$",
                r"^fi_b12x_moe_nvfp4_tokens16_topk2$", r"^sglang_qwen2.5_1.5b_gptq_marlin(_deterministic_mode)?_x6$",
                r"^moe_wna16_(cuda|triton)_", r"^deepgemm_bmk_bnk_mn$", r"^fi_top_p_renorm_", r"^nccl_allreduce_rank0_default$"]
    for stack, probes in sorted(load_reports(ROOT / "probes/results").items()):
        short = stack.replace("NVIDIA-", "").replace("-Blackwell-Server-Edition", "").replace("-80GB-HBM3", "")
        for name, reports in sorted(collapse_cublaslt(probes).items()):
            if not any(re.search(p, name) for p in patterns):
                continue
            inside, fresh, count = summarise(reports)
            runtime.append(f"| {short} | `{name}` | {inside} | {fresh} | {count} |")
    sys.path.insert(0, str(ROOT / "probes/shape"))
    from tables import main as shape_tables
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        shape_tables([str(ROOT / "probes/shape/results")])
    return {"inventory": inventory, "coverage": "\n".join(coverage), "runtime": "\n".join(runtime), "shape": output.getvalue().strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = ROOT / "README.md"
    before = path.read_text()
    after = before
    for key, block in render().items():
        start, end = f"<!-- generated:{key}:start -->", f"<!-- generated:{key}:end -->"
        if after.count(start) != 1 or after.count(end) != 1:
            raise ValueError(f"missing or duplicated README markers: {key}")
        prefix, rest = after.split(start)
        _, suffix = rest.split(end)
        after = prefix + start + "\n" + block + "\n" + end + suffix
    if args.check:
        print("README generated blocks are current." if before == after else "README tables are stale; run make docs.")
        return int(before != after)
    path.write_text(after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
