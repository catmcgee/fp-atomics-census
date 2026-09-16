#!/usr/bin/env python3
"""Derive an exact, reviewable C2 installation manifest from retained env.json.

The historical stack reports 210 installed distributions.  ``shape-hook`` is
the sole local editable distribution, so this emits the other 209 pins and a
short JSON plan saying which frozen checkout must provide the hook.  It never
contacts an index or installs packages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
LOCAL_EDITABLE = "shape-hook"


def requirement_lines(distributions: dict[str, str]) -> list[str]:
    if LOCAL_EDITABLE not in distributions:
        raise ValueError(f"historical distribution list lacks {LOCAL_EDITABLE!r}")
    return [f"{name}=={version}" for name, version in sorted(distributions.items(), key=lambda row: row[0].lower()) if name != LOCAL_EDITABLE]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-json", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    environment = json.loads(args.env_json.read_text())
    distributions = environment.get("installed_distributions")
    if not isinstance(distributions, dict):
        raise SystemExit("env.json has no installed_distributions object")
    lines = requirement_lines(distributions)
    args.constraints.parent.mkdir(parents=True, exist_ok=True)
    args.constraints.write_text("\n".join(lines) + "\n")
    plan = {
        "schema": 1,
        "historical_distribution_count": len(distributions),
        "wheel_install_count": len(lines),
        "local_editable": {
            "name": LOCAL_EDITABLE,
            "version": distributions[LOCAL_EDITABLE],
            "source_commit": EXPECTED_SOURCE,
            "install": "pip install -e SOURCE_ROOT/probes/shape/shape_hook_pkg",
        },
        "required_runtime": {
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "cuda": environment.get("cuda"),
            "cudnn": environment.get("cudnn"),
            "nccl_runtime_version": environment.get("nccl_runtime_version"),
        },
        "intentional_pip_check_failure": "vllm 0.29.0 requires flashinfer-python==0.6.18, while C2 retained flashinfer-python==0.6.18.post1",
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"constraints": str(args.constraints), "pins": len(lines), "plan": str(args.plan)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
