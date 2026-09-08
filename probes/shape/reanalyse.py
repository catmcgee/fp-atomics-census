"""Rebuild derived shape results, or check them in an isolated temporary copy."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import run_e2
import run_e3
import run_e4
import run_e5

HERE = Path(__file__).resolve().parent


def rebuild(base: Path) -> int:
    errors = 0
    root = base / "results"
    for out in sorted((root / "e2").glob("*/outputs.json")):
        repeats = len(json.loads(out.read_text()))
        errors += run_e2.analyse(out.parent, out.parent.name, repeats)
    for out in sorted((root / "e3").glob("*/a/outputs.json")):
        if (out.parent.parent / "b/outputs.json").exists():
            errors += run_e3.compare(out.parent.parent)
    for out in sorted((root / "e4").glob("*/runs.json")):
        runs = json.loads(out.read_text())
        meta = json.loads((out.parent / "run.json").read_text()) if (out.parent / "run.json").exists() else {}
        target = meta.get("args", {}).get("target", runs[0]["request_ids"].index(runs[0]["target_rid"]))
        errors += run_e4.analyse(out.parent, out.parent.name, max(r["repeat"] for r in runs) + 1, target)
    for other in sorted((base / "results_sm120/e3").glob("*/a/steps.json")):
        first = root / "e3" / other.parent.parent.name / "a"
        if first.exists():
            errors += run_e5.main([str(first), str(other.parent)])
    return errors


def derived(base):
    paths = [*base.glob("results/*/*/summary.json"), *base.glob("results/e3/*/e5_vs_*.json")]
    return {str(p.relative_to(base)): p.read_bytes() for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        return rebuild(HERE)
    before = derived(HERE)
    with tempfile.TemporaryDirectory(prefix="shape-reanalyse-") as temp:
        base = Path(temp)
        for name in ("results", "results_sm120"):
            if (HERE / name).exists():
                shutil.copytree(HERE / name, base / name)
        with contextlib.redirect_stdout(io.StringIO()):
            errors = rebuild(base)
        after = derived(base)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed:
        print("Stale derived results: " + ", ".join(changed))
    else:
        print("Derived shape results match the raw observations and analyser.")
    return int(bool(errors or changed))


if __name__ == "__main__":
    raise SystemExit(main())
