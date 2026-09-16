#!/usr/bin/env python
"""Decode every continuation of an arm directory and report degeneracy, so a verdict is never trusted on hashes alone."""
import json, sys, os
from pathlib import Path
from collections import Counter
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
def cycles(ids, max_p=16):
    for p in range(1, max_p + 1):
        if len(ids) >= 2 * p and all(ids[-i] == ids[-i - p] for i in range(1, 2 * p + 1) if i + p <= len(ids)):
            return p
    return None
for d in sys.argv[1:]:
    d = Path(d)
    run = json.loads((d / "run.json").read_text())
    model = run["args"]["model"]; rev = run["args"].get("revision")
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    outs = json.loads((d / "outputs.json").read_text())
    print(f"== {d} model={model}")
    for k in sorted(outs, key=lambda s: int(s) if s.isdigit() else s):
        ids = list(outs[k]["tokens"])
        text = tok.decode(ids)
        print(f"  slot {k:>2} n={len(ids)} distinct={len(set(ids))} cycle_period={cycles(list(ids))} :: {text!r}")
