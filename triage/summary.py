"""Print the headline counts of the inventory, so that no narrative number is typed by hand.

    python -m triage.summary inventory/*.jsonl [--markdown]

Findings exclude definition-only rows. "Inference" means path.direction is not
backward. A default-path row is one whose default_path is true.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["A", "A1", "A2", "A3", "B", "B-indirect", "C"]
ENGINE_ORDER = ["vllm", "sglang", "flashinfer", "flash-attention", "marlin", "DeepGEMM", "DeepEP", "cutlass"]


def load(files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for f in files:
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def summarise(rows: list[dict]) -> dict:
    per: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r.get("definition_only"):
            continue
        per[r["engine"]][r["class"]] += 1
    findings = [r for r in rows if not r.get("definition_only")]
    inference = [r for r in findings if r["path"].get("direction") != "backward"]
    default_a = [r for r in inference if r["class"] == "A" and r["default_path"] is True]
    training_a = [r for r in findings if r["class"] == "A" and r["path"].get("direction") == "backward"]
    a3_default = [r for r in inference if r["class"] == "A3" and r["default_path"] is True]
    atomic_side = [r for r in a3_default if (r.get("gate") or {}).get("default_is_atomic") is True]
    b_ind = [r for r in findings if r["class"] == "B-indirect"]
    b_ind_by = Counter(str(r["downstream"]["order_invariant"]) for r in b_ind)
    c_rows = [r for r in findings if r["class"] == "C"]
    c_by = Counter(r["opaque_target"]["library"] for r in c_rows)
    low = [r for r in rows if r["confidence"] == "low"]
    unchecked = [r for r in default_a if not r.get("cross_checked")]
    return {
        "rows": len(rows), "sites": len(findings), "per_engine": {e: dict(per[e]) for e in ENGINE_ORDER if e in per},
        "totals": dict(sum((per[e] for e in per), Counter())),
        "definition_only": sum(1 for r in rows if r.get("definition_only")),
        "default_path_A_inference": [r["id"] for r in default_a],
        "A_backward_training_only": [r["id"] for r in training_a],
        "A3_default_path_inference": [r["id"] for r in a3_default],
        "A3_default_atomic_side": [r["id"] for r in atomic_side],
        "B_indirect_by_order_invariant": dict(b_ind_by),
        "C_by_library": dict(c_by),
        "low_confidence": len(low), "default_A_not_cross_checked": [r["id"] for r in unchecked],
        "provenance": dict(Counter(r.get("provenance", "missing") for r in rows)),
        "rows_without_entry_points": sum(1 for r in rows if not r["path"].get("entry_points")),
    }


def markdown_table(s: dict) -> str:
    out = ["| Engine | Sites | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
    for e in ENGINE_ORDER:
        if e in s["per_engine"]:
            c = s["per_engine"][e]
            out.append(f"| {e} | {sum(c.values())} | " + " | ".join(str(c.get(k, 0)) for k in CLASSES) + " |")
    t = s["totals"]
    out.append(f"| Total | {sum(t.values())} | " + " | ".join(str(t.get(k, 0)) for k in CLASSES) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--markdown", action="store_true", help="print the class table as Markdown")
    args = ap.parse_args(argv)
    s = summarise(load(args.files))
    if args.markdown:
        print(markdown_table(s))
        return 0
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
