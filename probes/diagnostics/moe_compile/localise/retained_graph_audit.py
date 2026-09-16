#!/usr/bin/env python3
"""Audit the retained cu129/cu130 vLLM computation graphs without extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any


BUILD_PATH = re.compile(rb"/opt/issue56900/cu(?:129|130)/")
CALL = re.compile(r"torch\.ops\.vllm\.moe_forward_shared\(([^\n]*)\)")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_graph(archive: Path) -> tuple[str, bytes]:
    with tarfile.open(archive, "r:gz") as bundle:
        members = [
            name
            for name in bundle.getnames()
            if name.endswith("/computation_graph.py") and "/compile-on_" in name
        ]
        if len(members) != 1:
            raise ValueError(f"expected one compiled computation graph in {archive}, found {members}")
        stream = bundle.extractfile(members[0])
        if stream is None:
            raise ValueError(f"could not read {members[0]} from {archive}")
        return members[0], stream.read()


def graph_report(archive: Path) -> tuple[dict[str, Any], bytes]:
    member, raw = read_graph(archive)
    normalised = BUILD_PATH.sub(b"/opt/issue56900/BUILD/", raw)
    calls = CALL.findall(raw.decode())
    arguments = [[item.strip() for item in call.split(",")] for call in calls]
    return (
        {
            "archive": str(archive),
            "member": member,
            "bytes": len(raw),
            "sha256": sha256(raw),
            "normalised_sha256": sha256(normalised),
            "moe_forward_shared_calls": len(arguments),
            "all_calls_have_six_arguments": all(len(items) == 6 for items in arguments),
            "all_calls_alias_hidden_and_shared_input": all(
                len(items) == 6 and items[0] == items[2] for items in arguments
            ),
        },
        normalised,
    )


def audit(archives: list[Path]) -> dict[str, Any]:
    reports = []
    normalised = []
    for archive in archives:
        report, graph = graph_report(archive)
        reports.append(report)
        normalised.append(graph)
    return {
        "schema": 1,
        "graphs": reports,
        "normalised_graphs_identical": bool(normalised)
        and all(graph == normalised[0] for graph in normalised[1:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.archives)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
