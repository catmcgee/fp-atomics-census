#!/usr/bin/env python3
"""Audit eager MoE routing observations without inventing graph-replay counts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def hook_records(record_dir: Path) -> list[dict]:
    path = record_dir / "hook" / "rank0.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def real_rows(records: list[dict]) -> list[dict]:
    return [
        row
        for row in records
        if row.get("event", "forward") == "forward"
        and row.get("requests")
        and row.get("total_scheduled", 0) > 0
        and not all(str(req.get("req", "")).startswith("_warmup") for req in row["requests"])
    ]


def validate_rows(rows: list[dict], experts: int, top_k: int, allow_estimated: bool) -> tuple[list[str], dict]:
    errors = []
    sources = {}
    for index, row in enumerate(rows):
        if row.get("resolved_compile") != "0" or row.get("resolved_cudagraph") != "NONE":
            errors.append(
                f"pass {index}: resolved compile/graphs are "
                f"{row.get('resolved_compile')!r}/{row.get('resolved_cudagraph')!r}, expected 0/NONE"
            )
        mode = (row.get("dispatch") or {}).get("cudagraph_mode")
        if mode != "NONE":
            errors.append(f"pass {index}: dispatched CUDA graph mode {mode!r}, expected NONE")
        counts = row.get("moe_expert_counts_first_layer")
        source = row.get("moe_counts_source")
        sources[source] = sources.get(source, 0) + 1
        if not isinstance(counts, list) or len(counts) != experts:
            errors.append(f"pass {index}: count vector is not a list of {experts}: {counts!r}")
            continue
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in counts):
            errors.append(f"pass {index}: count vector contains a non-negative-integer violation")
            continue
        expected = int(row["total_scheduled"]) * top_k
        if sum(counts) != expected:
            errors.append(f"pass {index}: count sum {sum(counts)} != {expected}")
        if source != "actual_router" and not (allow_estimated and source == "estimated_from_router_logits"):
            errors.append(f"pass {index}: source {source!r} is not actual_router")
    return errors, sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-estimated", action="store_true")
    ap.add_argument("--config-json", type=Path, help="local config fixture; otherwise use pinned AutoConfig")
    args = ap.parse_args()

    if args.config_json:
        config = json.loads(args.config_json.read_text())
        experts = int(config["num_experts"])
        top_k = int(config["num_experts_per_tok"])
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model, revision=args.revision, local_files_only=True, trust_remote_code=False
        )
        experts = int(getattr(config, "num_experts"))
        top_k = int(getattr(config, "num_experts_per_tok"))
    records = hook_records(args.record_dir)
    rows = real_rows(records)
    errors, sources = validate_rows(rows, experts, top_k, args.allow_estimated)
    hook_errors = [row for row in records if row.get("error")]
    if hook_errors:
        errors.append(f"hook log contains {len(hook_errors)} error record(s): {hook_errors[:3]!r}")

    if len(rows) != 37:
        errors.append(f"real forward pass count {len(rows)} != 37")
    run = json.loads((args.record_dir / "run.json").read_text())
    run_args = run.get("args") or {}
    if run_args.get("model") != args.model or run_args.get("revision") != args.revision:
        errors.append(
            f"requested model/revision {args.model!r}/{args.revision!r} do not match "
            f"run args {run_args.get('model')!r}/{run_args.get('revision')!r}"
        )
    if run.get("resolved_compile") != "0" or run.get("resolved_cudagraph") != "NONE":
        errors.append(
            f"arm resolved to compile={run.get('resolved_compile')!r}, "
            f"cudagraph={run.get('resolved_cudagraph')!r}; expected 0/NONE"
        )

    report = {
        "schema": 1,
        "valid": not errors,
        "record_dir": str(args.record_dir),
        "model": args.model,
        "revision": args.revision,
        "num_experts": experts,
        "num_experts_per_tok": top_k,
        "forward_passes": len(rows),
        "sources": {str(k): v for k, v in sources.items()},
        "claim": (
            "Counts are actual selected-expert observations from eager Python execution."
            if not errors and sources and set(sources) == {"actual_router"}
            else "Validation failed or at least one count is absent/estimated; do not describe all counts as actual routing."
        ),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
