"""E5, cross-SKU: compare the hook logs of the same arm from two results directories.

    python probes/shape/run_e5.py probes/shape/results/e3/<arm>/a probes/shape/results_b200/e3/<arm>/a

Joins step by step on the shape vector and reports IDENTICAL or DIFFERS for
the per-request hidden hashes and argmax, plus the two environment records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from shape_common import write_json


def main(argv: list[str]) -> int:
    a, b = Path(argv[0]), Path(argv[1])
    sa = json.loads((a / "steps.json").read_text())
    sb = json.loads((b / "steps.json").read_text())
    ea = json.loads((a / "env.json").read_text())
    eb = json.loads((b / "env.json").read_text())
    rows = []
    for x, y in zip(sa, sb):
        hx = {r["req"]: (r["h"], r["argmax"]) for r in x["requests"]}
        hy = {r["req"]: (r["h"], r["argmax"]) for r in y["requests"]}
        rows.append({"step": x["step"], "shape_identical": x["shape_vector"] == y["shape_vector"], "hidden_identical": {k: hx[k][0] for k in hx} == {k: hy.get(k, (None,))[0] for k in hx},
                     "argmax_identical": {k: hx[k][1] for k in hx} == {k: hy.get(k, (None, None))[1] for k in hx}})
    summary = {"experiment": "E5", "a": {"gpu": ea.get("gpu"), "driver": ea.get("driver"), "torch": ea.get("torch")}, "b": {"gpu": eb.get("gpu"), "driver": eb.get("driver"), "torch": eb.get("torch")},
               "steps": len(rows), "hidden_identical_steps": sum(r["hidden_identical"] for r in rows), "argmax_identical_steps": sum(r["argmax_identical"] for r in rows),
               "verdict_P5": "IDENTICAL" if all(r["hidden_identical"] for r in rows) else "DIFFERS"}
    write_json(a.parent / f"e5_vs_{eb.get('gpu', 'b').replace(' ', '-')}.json", summary)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
