#!/usr/bin/env python3
"""Fail closed unless both TP ranks used one explicit FlashInfer workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PINNED = {
    "cuda_communicator.py": "03ece681b0bef349cb78dabc74585efea2cc160ae28bd711c81dfd807ca8dff5",
    "flashinfer_all_reduce.py": "607a9090f141afab59bbac73d2f6d06f557b8b13e150c2af9bc06d66d2ef3825",
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def load_events(trace_dir: Path) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    errors: list[str] = []
    paths = sorted(trace_dir.glob("rank*.jsonl"))
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        try:
            named_rank = int(path.stem.removeprefix("rank"))
        except ValueError:
            errors.append(f"unexpected trace filename: {path.name}")
            continue
        rows = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                continue
            if row.get("schema") != 1 or row.get("rank") != named_rank:
                errors.append(f"{path.name}:{line_number}: schema/rank mismatch")
            rows.append(row)
        by_rank[named_rank] = rows
    return by_rank, errors


def tensor_signature(row: dict[str, Any]) -> str:
    tensor = row["tensor"]
    value = {
        "backend": row["backend"],
        "group": row["group"],
        "world_size": row["world_size"],
        "shape": tensor["shape"],
        "dtype": tensor["dtype"],
        "nbytes": tensor["nbytes"],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_trace(
    trace_dir: Path,
    expected_backend: str,
    expected_run_id: str,
    expected_ranks: int = 2,
    hook_dir: Path | None = None,
) -> dict[str, Any]:
    by_rank, errors = load_events(trace_dir)
    if set(by_rank) != set(range(expected_ranks)):
        errors.append(f"trace ranks {sorted(by_rank)} != {list(range(expected_ranks))}")
    rank_reports = []
    for rank in range(expected_ranks):
        rows = by_rank.get(rank, [])
        run_ids = {row.get("run_id") for row in rows}
        if run_ids != {expected_run_id}:
            errors.append(f"rank {rank}: run ids {sorted(map(str, run_ids))} != {expected_run_id!r}")
        registrations = [row for row in rows if row.get("event") == "registered"]
        workspaces = [row for row in rows if row.get("event") == "workspace"]
        dispatches = [row for row in rows if row.get("event") == "dispatch"]
        dispatch_errors = [row for row in rows if row.get("event") == "dispatch_error"]
        if not registrations:
            errors.append(f"rank {rank}: instrumentation registration was not observed")
        if any(
            row.get("expected_backend") != expected_backend
            or row.get("configured_backend") != expected_backend
            or row.get("source_sha256") != PINNED
            for row in registrations
        ):
            errors.append(f"rank {rank}: registration backend/source attestation is invalid")
        if not workspaces:
            errors.append(f"rank {rank}: no FlashInfer workspace creation was observed")
        for row in workspaces:
            if (
                not row.get("created")
                or row.get("requested_backend") != expected_backend
                or row.get("actual_backend") != expected_backend
            ):
                errors.append(
                    f"rank {rank}: workspace was not created as explicit {expected_backend}: {row}"
                )
        tp_dispatches = [
            row
            for row in dispatches
            if row.get("world_size") == expected_ranks
            and str(row.get("group", "")).split(":", 1)[0] == "tp"
        ]
        if not tp_dispatches:
            errors.append(f"rank {rank}: no TP all-reduce dispatch was observed")
        non_fi = [row for row in tp_dispatches if row.get("backend") != "FLASHINFER"]
        if non_fi:
            errors.append(
                f"rank {rank}: {len(non_fi)} TP calls used a backend other than FLASHINFER"
            )
        if dispatch_errors:
            errors.append(f"rank {rank}: {len(dispatch_errors)} dispatch errors were observed")
        signatures = Counter(tensor_signature(row) for row in tp_dispatches)
        rank_reports.append(
            {
                "rank": rank,
                "events": len(rows),
                "workspace_events": len(workspaces),
                "tp_dispatch_calls": len(tp_dispatches),
                "backend_counts": dict(sorted(Counter(row["backend"] for row in tp_dispatches).items())),
                "tensor_signature_counts": dict(sorted(signatures.items())),
            }
        )
    files = []
    for path in sorted(trace_dir.glob("rank*.jsonl")):
        data = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    graph_reports = []
    if hook_dir is None:
        errors.append("shape-hook directory is required to attest graph dispatches")
    else:
        for rank in range(expected_ranks):
            path = hook_dir / f"rank{rank}.jsonl"
            if not path.is_file():
                errors.append(f"rank {rank}: shape-hook trace is missing")
                continue
            forwards = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip() and json.loads(line).get("event") == "forward"
            ]
            modes = Counter(
                str((row.get("dispatch") or {}).get("cudagraph_mode"))
                for row in forwards
            )
            if len(forwards) != 37:
                errors.append(f"rank {rank}: {len(forwards)} application forwards != 37")
            if not forwards or any(
                row.get("resolved_compile") != "3"
                or row.get("resolved_cudagraph") != "FULL_AND_PIECEWISE"
                for row in forwards
            ):
                errors.append(f"rank {rank}: compile/graph resolution was not fixed")
            if not modes or set(modes) - {"FULL", "PIECEWISE"} or not modes.get("FULL"):
                errors.append(f"rank {rank}: graph dispatch modes are not fully traced: {dict(modes)}")
            graph_reports.append(
                {"rank": rank, "application_forwards": len(forwards), "graph_dispatch_counts": dict(sorted(modes.items()))}
            )
    return {
        "schema": 1,
        "valid": not errors,
        "errors": errors,
        "expected_backend": expected_backend,
        "expected_run_id": expected_run_id,
        "expected_ranks": expected_ranks,
        "ranks": rank_reports,
        "graph_dispatch": graph_reports,
        "trace_files": files,
        "interpretation": (
            "Counts are Python dispatcher decisions, including graph construction/capture. "
            "They establish the backend encoded by each observed TP collective call; they "
            "are not a CUPTI trace of every CUDA graph replay launch."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--expected-backend", choices=("trtllm", "mnnvl"), required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-ranks", type=int, default=2)
    parser.add_argument("--hook-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_trace(
        args.trace_dir,
        args.expected_backend,
        args.expected_run_id,
        args.expected_ranks,
        args.hook_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_bytes(canonical(report))
    temporary.replace(args.output)
    if not report["valid"]:
        for error in report["errors"]:
            print(error)
        return 1
    print(
        f"valid {args.expected_backend} trace: "
        + ", ".join(
            f"rank {row['rank']}={row['tp_dispatch_calls']} TP calls"
            for row in report["ranks"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
