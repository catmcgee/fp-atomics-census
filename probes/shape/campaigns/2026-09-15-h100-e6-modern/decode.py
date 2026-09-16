#!/usr/bin/env python3
"""Decode recorded outputs using pinned public tokenizers; prints JSON.

Run with: uv run --no-project --with tokenizers==0.22.2 python PATH/decode.py
This downloads tokenizer.json for each recorded model revision, never weights.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
tokenizers = {}
sources = {}
records = []
seen = set()
for path in sorted(ROOT.glob("results_m*/e6/*/record/outputs.json")):
    meta = json.loads(path.with_name("run.json").read_text())
    if meta["run_id"] in seen:
        continue
    seen.add(meta["run_id"])
    model, revision = (meta["args"][key] for key in ("model", "revision"))
    key = model + "@" + revision
    if key not in tokenizers:
        url = f"https://huggingface.co/{model}/resolve/{revision}/tokenizer.json"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        tokenizers[key] = Tokenizer.from_str(data.decode())
        sources[key] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    outputs = json.loads(path.read_text())
    for slot, output in sorted(outputs.items(), key=lambda item: int(item[0])):
        ids = output["tokens"]
        records.append({"record": str(path.relative_to(ROOT)), "slot": slot,
                        "tokenizer": key, "token_ids": ids,
                        "distinct_tokens": len(set(ids)),
                        "text": tokenizers[key].decode(ids, skip_special_tokens=False)})
assert len(seen) == 10 and len(records) == 160
assert all(len(record["token_ids"]) == 32 for record in records)
assert min(record["distinct_tokens"] for record in records) >= 12
print(json.dumps({"tokenizers": sources, "records": records}, indent=2, ensure_ascii=False))
