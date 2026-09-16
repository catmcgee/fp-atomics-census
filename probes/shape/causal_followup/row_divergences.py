#!/usr/bin/env python3
"""Describe exact hook-row digest differences in completed factorial replays.

This is read-only trace analysis: it reports observations from an already
compared record/replay pair and makes no claim about their cause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PHASES = ("warm", "cold_0", "cold_1", "cold_2")
DIGEST_FIELDS = ("h", "logits_h", "argmax")
ROW_CONTEXT_FIELDS = ("phase", "computed", "kv", "q")


def read_steps(path: Path) -> list[dict[str, Any]]:
    parsed = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [step for step in parsed if step.get("event", "forward") == "forward" and step.get("requests")]


def slot(request: str | None) -> str:
    return str(request or "").split("-", 1)[0]


def shape(step: dict[str, Any]) -> Any:
    raw = step.get("shape_vector")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def row_view(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in (*ROW_CONTEXT_FIELDS, *DIGEST_FIELDS)}


def compare(record: list[dict[str, Any]], replay: list[dict[str, Any]]) -> dict[str, Any]:
    if len(record) != len(replay):
        raise ValueError(f"pass count differs: {len(record)} != {len(replay)}")
    classes: Counter[str] = Counter()
    rows, total = [], 0
    for pass_index, (left, right) in enumerate(zip(record, replay, strict=True)):
        left_slots = [slot(row.get("req")) for row in left["requests"]]
        right_slots = [slot(row.get("req")) for row in right["requests"]]
        if left_slots != right_slots:
            raise ValueError(f"pass {pass_index}: request order differs")
        for left_row, right_row in zip(left["requests"], right["requests"], strict=True):
            total += 1
            changed = [field for field in DIGEST_FIELDS if left_row.get(field) != right_row.get(field)]
            classes["equal" if not changed else "+".join(changed)] += 1
            if changed:
                rows.append({
                    "pass": pass_index,
                    "step": left.get("step"),
                    "slot": slot(left_row.get("req")),
                    "digest_fields_differing": changed,
                    "pass_shape": shape(left),
                    "dispatch": left.get("dispatch"),
                    "record": row_view(left_row),
                    "replay": row_view(right_row),
                })
    return {"rows_compared_from_hook": total, "row_digest_classes": dict(sorted(classes.items())), "divergent_rows": rows}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(results: Path, cell: str, phases: tuple[str, ...]) -> dict[str, Any]:
    arm = next((path for path in (results / "e6").glob(f"*_factorial_{cell}_mixed") if path.is_dir()), None)
    if arm is None:
        raise ValueError(f"missing arm for {cell}")
    record_hook = arm / "record" / "hook" / "rank0.jsonl"
    record = read_steps(record_hook)
    comparisons = []
    for phase in phases:
        replay_dir = arm / f"replay_{phase}"
        summary_path = arm / f"summary_replay_{phase}.json"
        replay_hook = replay_dir / "hook" / "rank0.jsonl"
        if not summary_path.is_file() or not replay_hook.is_file():
            raise ValueError(f"missing completed replay files for {cell}_{phase}")
        summary = json.loads(summary_path.read_text())
        result = compare(record, read_steps(replay_hook))
        if result["rows_compared_from_hook"] != summary.get("rows_compared"):
            raise ValueError(f"summary row count disagrees with hook for {cell}_{phase}")
        comparisons.append({
            "replay": f"replay_{phase}",
            "summary_sha256": sha256(summary_path),
            "record_hook_sha256": sha256(record_hook),
            "replay_hook_sha256": sha256(replay_hook),
            "verdict_P2": summary.get("verdict_P2"),
            **result,
        })
    return {
        "schema": 1,
        "cell": cell,
        "scope": "descriptive hook-trace association only; no operator or compiler causality is inferred",
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--phases", nargs="+", choices=PHASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refuse existing output: {args.output}")
    report = analyse(args.results, args.cell, tuple(args.phases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell": args.cell, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
