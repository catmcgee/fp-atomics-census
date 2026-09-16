"""E5, cross-SKU: compare the hook logs of the same arm from two results directories.

    python probes/shape/run_e5.py probes/shape/results/e3/<arm>/a probes/shape/results_b200/e3/<arm>/a

Joins step by step on the shape vector and reports IDENTICAL or DIFFERS for
the per-request hidden hashes and argmax, plus the two environment records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from shape_common import slot_of, write_json


def main(argv: list[str]) -> int:
    a, b = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    from comparison import compare_traces, load_steps, normalise_outputs, output_trace_errors
    from shape_common import analysis_metadata
    sa, sb = load_steps(a), load_steps(b)
    comparison = compare_traces(sa, sb)
    for tag, run, steps in (("a", a, sa), ("b", b, sb)):
        outputs = normalise_outputs(json.loads((run / "outputs.json").read_text()))
        comparison["validation_errors"].extend(f"{tag}: {e}" for e in output_trace_errors(steps, outputs))
    if comparison["validation_errors"]:
        comparison["verdict"] = "INVALID"
    ea = json.loads((a / "env.json").read_text())
    eb = json.loads((b / "env.json").read_text())
    rows = comparison["rows"]
    summary = {"experiment": "E5", "a": {k: ea.get(k) for k in ("gpu", "driver", "torch")},
               "b": {k: eb.get(k) for k in ("gpu", "driver", "torch")},
               "steps": len(rows), "hidden_identical_steps": sum(r["hidden_identical"] for r in rows),
               "argmax_identical_steps": sum(r["argmax_identical"] for r in rows),
               "shape_histories_equal": comparison["shape_histories_equal"],
               "validation_errors": comparison["validation_errors"],
               "verdict_recorded_stacks": comparison["verdict"], "verdict_P5": "NOT ISOLATED",
               "claim_scope": "Free-running trajectories on two recorded stacks; drivers and GPU differ. After token divergence later model inputs differ; this is not a teacher-forced GPU-only comparison."}
    common = Path(__import__("os").path.commonpath([a.resolve(), b.resolve()]))
    summary.update(analysis_metadata(common, [p for run in (a, b) for p in
                   [run / "steps.json", run / "env.json", run / "outputs.json", *sorted((run / "hook").glob("rank*.jsonl"))]]))
    write_json(a.parent / f"e5_vs_{eb.get('gpu', 'b').replace(' ', '-')}.json", summary)
    print(json.dumps(summary, indent=1))
    return 1 if comparison["validation_errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
