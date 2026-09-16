#!/usr/bin/env python3
"""Compare retained token matrices without loading a model or GPU runtime."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    handle = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")
    with handle:
        return json.load(handle)


def extract_tokens(value: Any) -> tuple[list[list[int]], str]:
    # Public moe_compile_reproducer result.
    if isinstance(value, dict) and isinstance(value.get("repeats"), list):
        repeat = value["repeats"][0]
        return [[int(token) for token in row["token_ids"]] for row in repeat], "repeats[0]"
    # Targeted control comparison/result.
    if isinstance(value, dict) and isinstance(value.get("result"), dict):
        output = value["result"].get("output")
        if isinstance(output, dict) and isinstance(output.get("token_ids"), list):
            return [[int(token) for token in row] for row in output["token_ids"]], "result.output"
    if isinstance(value, dict) and isinstance(value.get("output"), dict):
        tokens = value["output"].get("token_ids")
        if isinstance(tokens, list):
            return [[int(token) for token in row] for row in tokens], "output"
    raise ValueError("unrecognized result schema; no token matrix found")


def cycle_period(token_ids: list[int], tail: int = 16, max_period: int = 3) -> int | None:
    tail_ids = token_ids[-tail:]
    for period in range(1, max_period + 1):
        if len(tail_ids) > period and all(
            tail_ids[index] == tail_ids[index + period]
            for index in range(len(tail_ids) - period)
        ):
            return period
    return None


def matrix_summary(tokens: list[list[int]]) -> dict[str, Any]:
    duplicate_pairs = (
        [tokens[index] == tokens[index + 8] for index in range(8)]
        if len(tokens) == 16
        else None
    )
    periods = [cycle_period(row) for row in tokens]
    return {
        "rows": len(tokens),
        "row_lengths": [len(row) for row in tokens],
        "canonical_sha256": canonical_sha256(tokens),
        "first_tokens": [row[0] if row else None for row in tokens],
        "duplicate_pairs": duplicate_pairs,
        "duplicate_pairs_equal": sum(duplicate_pairs) if duplicate_pairs is not None else None,
        "cycle_period_last_16": periods,
        "short_cycles": sum(period is not None for period in periods),
    }


def source_record(argument: str, path: Path, selector: str) -> dict[str, Any]:
    return {
        "path": argument,
        "resolved_path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "token_selector": selector,
    }


def audit(left_arg: str, right_arg: str) -> dict[str, Any]:
    left_path, right_path = Path(left_arg), Path(right_arg)
    left, left_selector = extract_tokens(load_json(left_path))
    right, right_selector = extract_tokens(load_json(right_path))
    if len(left) != len(right):
        raise ValueError(f"row count differs: {len(left)} != {len(right)}")
    compare_lengths = [min(len(a), len(b)) for a, b in zip(left, right)]
    left_prefix = [row[:length] for row, length in zip(left, compare_lengths)]
    right_prefix = [row[:length] for row, length in zip(right, compare_lengths)]
    differing_elements = [
        sum(a != b for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left_prefix, right_prefix)
    ]
    return {
        "schema": 1,
        "sources": {
            "left": source_record(left_arg, left_path, left_selector),
            "right": source_record(right_arg, right_path, right_selector),
        },
        "comparison": {
            "compared_lengths": compare_lengths,
            "exact_prefix_identity": left_prefix == right_prefix,
            "differing_rows": sum(count != 0 for count in differing_elements),
            "differing_elements_by_row": differing_elements,
            "left_prefix": matrix_summary(left_prefix),
            "right_prefix": matrix_summary(right_prefix),
        },
        "full_matrices": {
            "left": matrix_summary(left),
            "right": matrix_summary(right),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.left, arguments.right)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
