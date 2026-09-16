#!/usr/bin/env python3
"""Compare TP=2 rank-1 traces and cross-rank output fields without a GPU."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src/probes/shape"))
from run_e6 import compare_replay
from shape_common import read_hook, real_steps


def steps(run, rank):
    raw = read_hook(run / "hook", rank)
    assert raw and not any("error" in row for row in raw), run
    return real_steps(raw)


comparisons = []
for item in json.loads((ROOT / "comparisons.json").read_text())["comparisons"]:
    if "_tp2_" not in item["arm"]:
        continue
    arm = ROOT / f"results_m{item['machine']}/e6" / item["arm"]
    report = compare_replay(steps(arm / "record", 1), steps(arm / item["replay"], 1))
    assert not report["validation_errors"] and report["first_mismatch"] is None
    assert report["rows_differing"] == item["rows_differing"]
    comparisons.append({"label": item["label"], "rank1": report})
assert len(comparisons) == 7

cross_rank = []
seen = set()
for path in sorted(ROOT.glob("results_m*/e6/*/*/hook/rank1.jsonl")):
    run = path.parent.parent
    run_id = json.loads((run / "run.json").read_text())["run_id"]
    if run_id in seen:
        continue
    seen.add(run_id)
    ranks = [steps(run, rank) for rank in (0, 1)]
    assert len(ranks[0]) == len(ranks[1])
    counts = dict.fromkeys(("h", "logits_h", "argmax"), 0)
    for a, b in zip(*ranks):
        assert len(a["requests"]) == len(b["requests"])
        for x, y in zip(a["requests"], b["requests"]):
            assert x["req"] == y["req"] and x["new_token_ids"] == y["new_token_ids"]
            for field in counts:
                counts[field] += x[field] != y[field]
    assert counts["logits_h"] == counts["argmax"] == 0
    cross_rank.append({"run": str(run.relative_to(ROOT)), "differing_rows": counts})
print(json.dumps({"comparisons": comparisons, "cross_rank": cross_rank}, indent=2))
