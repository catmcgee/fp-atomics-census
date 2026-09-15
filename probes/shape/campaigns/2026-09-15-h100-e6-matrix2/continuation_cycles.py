"""Token-level cycle check of every continuation, from the audit of the 15 September campaign (no tokenizer needed).
For each request: the smallest period p <= 16 such that the last L >= max(2p, 12) tokens are p-periodic
(the longest such suffix is reported). A cycle is explained when the prompt itself is a repeated sentence
(the prompt's token ids contain the continuation's periodic unit at least twice). The audit's looser
variant replaces max(2 * p, 12) with max(p + 4, 12).

    python continuation_cycles.py ARM/record [ARM/replay ...]
"""
import json, sys
from pathlib import Path

def slot(k): return str(k).split("-")[0]

def periodic_suffix(t, p):
    n = 0
    for i in range(len(t) - 1, p - 1, -1):
        if t[i] == t[i - p]:
            n += 1
        else:
            break
    return n + p if n else 0  # length of the p-periodic suffix

def best_cycle(t, pmax=16):
    for p in range(1, pmax + 1):
        L = periodic_suffix(t, p)
        if L >= max(2 * p, 12):
            return p, L
    return None, 0

def count_sub(hay, needle):
    return sum(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))

for rec in sys.argv[1:]:
    rec = Path(rec)
    outs = json.loads((rec / "outputs.json").read_text())
    prompts = {}
    for l in (rec / "hook" / "rank0.jsonl").read_text().splitlines():
        if l.strip():
            for a in json.loads(l).get("admitted") or []:
                prompts[slot(a["req"])] = a["prompt_token_ids"]
    rows, unexplained = [], 0
    for k in sorted(outs, key=lambda k: int(slot(k))):
        t = outs[k]["tokens"]; t = json.loads(t) if isinstance(t, str) else t
        p, L = best_cycle(t)
        explained = None
        if p:
            unit = t[-p:]
            # rotate-invariant: any rotation of the unit counted in the prompt
            explained = any(count_sub(prompts.get(slot(k), []), unit[r:] + unit[:r]) >= 2 for r in range(p))
            unexplained += not explained
        rows.append((slot(k), len(t), len(set(t)), p, L, explained))
    name = rec.parent.name + "/" + rec.name
    print(f"{name}: requests {len(rows)}; cycles (p<=16, suffix>=max(2p,12)) {sum(r[3] is not None for r in rows)}; unexplained {unexplained}; min distinct {min(r[2] for r in rows)}")
    for r in rows:
        if r[3] is not None:
            print(f"   slot {r[0]}: period {r[3]} over last {r[4]} of {r[1]} tokens; distinct {r[2]}; prompt repeats the unit: {r[5]}")
