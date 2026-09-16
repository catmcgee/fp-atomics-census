#!/usr/bin/env python3
"""Associate per-kernel Inductor choices with factorial replay outcomes.

This is CPU-only association analysis.  It does not infer a causal mechanism:
it normalises ``.best_config`` values, compares them by cache key, and reports
the observed hook-row digest classes beside the frozen comparator verdict.
It accepts the private durable results tree or the compressed public bundle.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


IGNORE = {"time_taken_ms", "triton_cache_hash"}
PHASES = ("warm", "cold_0", "cold_1", "cold_2")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def public_json(bundle: Path, relative: Path) -> Any:
    with gzip.open(bundle / "raw" / relative.with_name(relative.name + ".gz"), "rt") as handle:
        return json.load(handle)


def normalise_choice(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(value) if key not in IGNORE}


def source_kernel_name(source: str | None) -> str | None:
    if source is None:
        return None
    # Generated Inductor Python records this name twice: in ``inductor_meta``
    # and as the Python Triton function.  Keep the name, rather than treating
    # the two-character cache directory as a kernel identity.
    match = re.search(r"['\"]kernel_name['\"]\s*:\s*['\"](triton_[A-Za-z0-9_]+)", source)
    if match is None:
        match = re.search(r"\bdef\s+(triton_[A-Za-z0-9_]+)\s*\(", source)
    return match.group(1) if match else None


def source_by_config_from_archive(archive: Path) -> dict[str, dict[str, str | None]]:
    """Map ``inductor_env/ab/key.best_config`` to the sibling source file."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) < 4 or parts[-2] in {"inductor", "triton"}:
                continue
            try:
                position = parts.index("inductor")
            except ValueError:
                continue
            if len(parts) != position + 3 or Path(parts[-1]).suffix != ".py":
                continue
            handle = bundle.extractfile(member)
            if handle is not None:
                grouped.setdefault(parts[-2], []).append((member.name, handle.read().decode(errors="replace")))
    result = {}
    for key, values in grouped.items():
        # There is normally one generated source per two-letter best-config key.
        path, text = sorted(values)[0]
        result[key] = {"source_path": path, "kernel_name": source_kernel_name(text)}
    return result


def source_by_config_from_public(bundle: Path, label: str) -> dict[str, dict[str, str | None]]:
    archive = bundle / "sources_ir" / f"{label}.sources.tar.gz"
    if not archive.is_file():
        return {}
    return source_by_config_from_archive(archive)


def hook_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        request for step in rows
        if step.get("event", "forward") == "forward" and step.get("requests")
        for request in step["requests"]
        if not str(request.get("req", "")).startswith("_warmup")
    ]


def row_classes(recorded: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> dict[str, Any]:
    left, right = hook_rows(recorded), hook_rows(replayed)
    if len(left) != len(right):
        return {"comparable": False, "record_rows": len(left), "replay_rows": len(right), "classes": {}}
    classes: Counter[str] = Counter()
    for a, b in zip(left, right, strict=True):
        changed = [field for field in ("h", "logits_h", "argmax") if a.get(field) != b.get(field)]
        classes["equal" if not changed else "+".join(changed)] += 1
    return {
        "comparable": True,
        "record_rows": len(left),
        "replay_rows": len(right),
        "classes": dict(sorted(classes.items())),
        "record_digest_sha256": hashlib.sha256(canonical([[row.get(k) for k in ("h", "logits_h", "argmax")] for row in left])).hexdigest(),
        "replay_digest_sha256": hashlib.sha256(canonical([[row.get(k) for k in ("h", "logits_h", "argmax")] for row in right])).hexdigest(),
    }


def changed_kernels(record: dict[str, Any], replay: dict[str, Any], sources: dict[str, dict[str, str | None]]) -> list[dict[str, Any]]:
    before, after = record.get("best_configs", {}), replay.get("best_configs", {})
    changes = []
    for path in sorted(set(before) | set(after)):
        left, right = before.get(path), after.get(path)
        if not isinstance(left, dict) or not isinstance(right, dict):
            changes.append({"best_config_path": path, "state": "missing_on_one_side"})
            continue
        a, b = normalise_choice(left), normalise_choice(right)
        if a == b:
            continue
        prefix = Path(path).parent.name
        changes.append({
            "best_config_path": path,
            "cache_key_prefix": prefix,
            "record_choice": a,
            "replay_choice": b,
            "changed_fields": [key for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)],
            **sources.get(prefix, {"source_path": None, "kernel_name": None}),
        })
    return changes


def analyse_private(results: Path, cell: str, phases: tuple[str, ...] = PHASES) -> dict[str, Any]:
    arm = next((path for path in (results / "e6").glob(f"*_factorial_{cell}_mixed") if path.is_dir()), None)
    if arm is None:
        raise ValueError(f"missing arm for {cell}")
    record = load_json(arm / "record" / "artefacts.json")
    record_hook = [json.loads(line) for line in (arm / "record" / "hook" / "rank0.jsonl").read_text().splitlines() if line]
    archive_root = results / "archives"
    comparisons = []
    for phase in phases:
        replay_dir = arm / f"replay_{phase}"
        summary = load_json(arm / f"summary_replay_{phase}.json")
        replay = load_json(replay_dir / "artefacts.json")
        replay_hook = [json.loads(line) for line in (replay_dir / "hook" / "rank0.jsonl").read_text().splitlines() if line]
        sources = source_by_config_from_archive(archive_root / f"{cell}_{phase}" / "cache.tar.gz")
        comparisons.append({"replay": f"replay_{phase}", "verdict_P2": summary.get("verdict_P2"), "rows_compared": summary.get("rows_compared"), "changed_kernels": changed_kernels(record, replay, sources), "row_digest_classes": row_classes(record_hook, replay_hook)})
    return {"schema": 1, "cell": cell, "scope": "association only; changed compilation choices are not a demonstrated cause of row/output divergence", "comparisons": comparisons}


def analyse_public(bundle: Path, cell: str, phases: tuple[str, ...] = PHASES) -> dict[str, Any]:
    arm = next(path.name for path in (bundle / "raw" / "e6").glob(f"*_factorial_{cell}_mixed"))
    def data(*parts: str) -> Any:
        return public_json(bundle, Path("e6", arm, *parts))
    record, record_hook = data("record", "artefacts.json"), [json.loads(line) for line in gzip.open(bundle / "raw" / "e6" / arm / "record" / "hook" / "rank0.jsonl.gz", "rt") if line]
    comparisons = []
    for phase in phases:
        replay = data(f"replay_{phase}", "artefacts.json")
        summary = data(f"summary_replay_{phase}.json")
        hook_path = bundle / "raw" / "e6" / arm / f"replay_{phase}" / "hook" / "rank0.jsonl.gz"
        replay_hook = [json.loads(line) for line in gzip.open(hook_path, "rt") if line]
        sources = source_by_config_from_public(bundle, f"{cell}_{phase}")
        comparisons.append({"replay": f"replay_{phase}", "verdict_P2": summary.get("verdict_P2"), "rows_compared": summary.get("rows_compared"), "changed_kernels": changed_kernels(record, replay, sources), "row_digest_classes": row_classes(record_hook, replay_hook)})
    return {"schema": 1, "cell": cell, "scope": "association only; changed compilation choices are not a demonstrated cause of row/output divergence", "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results", type=Path)
    mode.add_argument("--public-bundle", type=Path)
    parser.add_argument("--cell", default="d0_c1_b1")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=PHASES,
                        help="completed replay phases to inspect; defaults to all four")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refuse existing output: {args.output}")
    phases = tuple(args.phases)
    report = analyse_private(args.results, args.cell, phases) if args.results else analyse_public(args.public_bundle, args.cell, phases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell": args.cell, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
