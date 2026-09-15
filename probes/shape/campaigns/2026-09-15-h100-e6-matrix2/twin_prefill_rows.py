"""Per-row batch coupling signature, from the audit of the 15 September campaign. For request pairs with identical prompt token ids that are
prefilled in different passes (different neighbours and batch sizes), compare the hidden-state hash of
their full-prefill row and of each later decode row whose consumed token ids and position are equal.
Per-token activation scales leave a row's quantisation independent of its neighbours; a per-tensor scale
is a batch statistic. Equality is therefore expected more often under per-token than under per-tensor,
but kernel choice by batch size can also break equality, so this is a signature, not a proof.

    python twin_prefill_rows.py LABEL ARM/record/hook/rank0.jsonl [LABEL ARM/replay/hook/rank0.jsonl ...]
"""
import json, sys
from collections import defaultdict

def load(p):
    return [json.loads(l) for l in open(p) if l.strip() and json.loads(l).get("requests")]

def slot(r): return str(r["req"]).split("-")[0]

def analyse(path):
    steps = load(path)
    prompt = {}
    for s in steps:
        for a in s.get("admitted") or []:
            prompt[slot(a)] = tuple(a["prompt_token_ids"])
    # per slot: list of (computed, q, phase, tuple(new_token_ids), h, pass index, batch rows)
    rows = defaultdict(list)
    for i, s in enumerate(steps):
        for r in s["requests"]:
            rows[slot(r)].append((r["computed"], r["q"], r["phase"], tuple(r["new_token_ids"]), r["h"], i, len(s["requests"]), s["total_scheduled"]))
    groups = defaultdict(list)
    for k, p in prompt.items():
        groups[p].append(k)
    eq = ne = 0; pre_eq = pre_ne = 0; detail = []
    for p, ks in groups.items():
        ks = sorted(ks, key=int)
        for a_i in range(len(ks)):
            for b_i in range(a_i + 1, len(ks)):
                A = {(c, q, tuple(t)): (h, i, n, tot) for c, q, ph, t, h, i, n, tot in rows[ks[a_i]]}
                B = {(c, q, tuple(t)): (h, i, n, tot) for c, q, ph, t, h, i, n, tot in rows[ks[b_i]]}
                common = [key for key in A if key in B and A[key][1] != B[key][1]]
                for key in common:
                    same = A[key][0] == B[key][0]
                    if key[0] == 0:
                        pre_eq += same; pre_ne += not same
                        detail.append((ks[a_i], ks[b_i], "prefill", same, A[key][3], B[key][3]))
                    else:
                        eq += same; ne += not same
    return pre_eq, pre_ne, eq, ne, detail

for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
    pe, pn, e, n, d = analyse(path)
    print(f"{label}: prefill rows of identical prompts in different passes: equal h {pe}, differing {pn}; later rows with identical inputs and position in different passes: equal h {e}, differing {n}")
    for x in d:
        print(f"    slots {x[0]} and {x[1]} prefill: h equal {x[3]} (tokens in pass: {x[4]} vs {x[5]})")
