#!/usr/bin/env python3
"""Row-level difference between two arm directories' rank-0 hook logs, for pairs the analyser cannot compare
(two replays of one record). Usage: row_diff.py DIR_A DIR_B [--rank N]"""
import json, sys
from pathlib import Path
def rows(d, rank=0):
    p = Path(d) / "hook" / f"rank{rank}.jsonl"
    out = []
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("event") != "forward": continue
        for slot, q in enumerate(r.get("requests", [])):
            if str(q.get("req", "")).startswith("_warmup"): continue
            out.append(((r["step"], slot, str(q.get("req")).split("-")[0]), (q.get("h"), q.get("logits_h"), q.get("argmax"))))
    return out
rank = 0
args = [a for a in sys.argv[1:]]
if "--rank" in args:
    i = args.index("--rank"); rank = int(args[i+1]); del args[i:i+2]
a, b = rows(args[0], rank), rows(args[1], rank)
if [k for k, _ in a] != [k for k, _ in b]:
    print("ROW KEYS DIFFER", len(a), len(b)); sys.exit(2)
diff = [(k, x, y) for (k, x), (_, y) in zip(a, b) if x != y]
print(f"{args[0]}\n  vs {args[1]}\n  rows {len(a)}, differing {len(diff)}")
for k, x, y in diff[:40]:
    fields = [f for f, i in (("h", 0), ("logits_h", 1), ("argmax", 2)) if x[i] != y[i]]
    print(f"    pass {k[0]:>3} slot {k[1]:>2} req {k[2]:>3}: {','.join(fields)}  argmax {x[2]} -> {y[2]}")
