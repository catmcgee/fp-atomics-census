"""Flatten inventory/*.jsonl into a single CSV.

    python -m triage.to_csv inventory/*.jsonl > inventory.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "id", "engine", "class", "kind", "dtype", "memory_space", "file", "line_start", "line_end", "function",
    "default_path", "direction", "phase", "moe", "quant", "attention_backend", "parallelism",
    "gate_flag", "gate_default", "confidence", "path_description", "evidence", "downstream_order_invariant",
    "opaque_library", "default_path_condition", "definition_only", "provenance", "candidate_ids", "entry_points",
    "cross_checked", "runtime_evidence", "notes", "sha",
]


def flatten(row: dict) -> dict:
    p = row.get("path", {})
    g = row.get("gate") or {}
    d = row.get("downstream") or {}
    o = row.get("opaque_target") or {}
    return {
        "id": row["id"], "engine": row["engine"], "class": row["class"], "kind": row["kind"],
        "dtype": row["dtype"], "memory_space": row["memory_space"], "file": row["file"],
        "line_start": row["line_start"], "line_end": row["line_end"], "function": row["function"],
        "default_path": row["default_path"], "direction": p.get("direction", ""), "phase": p.get("phase", ""),
        "moe": p.get("moe", ""), "quant": ";".join(p.get("quant", [])),
        "attention_backend": ";".join(p.get("attention_backend", [])),
        "parallelism": ";".join(p.get("parallelism", [])),
        "gate_flag": g.get("flag", ""), "gate_default": g.get("default", ""),
        "confidence": row["confidence"], "path_description": p.get("description", ""),
        "evidence": " | ".join(f"{e['ref']}: {e['claim']}" for e in row.get("evidence", [])),
        "downstream_order_invariant": d.get("order_invariant", ""),
        "opaque_library": o.get("library", ""),
        "default_path_condition": row.get("default_path_condition", ""),
        "definition_only": row.get("definition_only", False), "provenance": row.get("provenance", ""),
        "candidate_ids": ";".join(row.get("candidate_ids", [])), "entry_points": ";".join(p.get("entry_points", [])),
        "cross_checked": row.get("cross_checked", False),
        "runtime_evidence": " | ".join(f"{e['probe']}: {e['in_process']}/{e['fresh_process']} ({e['stack']}; relation={e['relation']}; evaluations={e['evaluations']}; kernel_identity_verified={e['kernel_identity_verified']})" for e in row.get("runtime_evidence", [])),
        "notes": row.get("notes", ""), "sha": row["sha"],
    }


def main(argv: list[str] | None = None) -> int:
    files = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                w.writerow(flatten(json.loads(line)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
