"""Account for every eligible scanner candidate without inventing triage decisions.

Unlinked candidates remain unresolved. This ledger does not turn a source
census into a completeness proof. Generate with python -m triage.dispositions.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def records(rows, candidates):
    links = defaultdict(list)
    for row in rows:
        for cid in row.get("candidate_ids", []):
            links[cid].append(row["id"])
    return [{"candidate_id": c["id"], "sha": c["sha"], "file": c["file"], "line": c["line"],
             "status": "linked" if links[c["id"]] else "unresolved",
             "inventory_ids": sorted(links[c["id"]]),
             "reason": "Explicit inventory provenance link; see that row's source evidence." if links[c["id"]] else
                       "No explicit disposition established; may be a duplicate, helper or additional site. Requires triage."}
            for c in candidates if c.get("in_scope") and not c.get("excluded_reason")]


def validate_dispositions(rows, candidates_dir, dispositions_dir):
    errors = []
    for engine in sorted({r["engine"] for r in rows}):
        path = dispositions_dir / f"{engine}.jsonl"
        candidates = candidates_dir / f"{engine}.jsonl"
        if not path.is_file() or not candidates.is_file():
            errors.append(f"{engine}: missing candidate disposition ledger or candidate file")
            continue
        expected = records([r for r in rows if r["engine"] == engine], list(map(json.loads, candidates.read_text().splitlines())))
        actual = list(map(json.loads, path.read_text().splitlines()))
        if actual != expected:
            errors.append(f"{path}: incomplete, duplicate or stale candidate dispositions; regenerate after reviewing source links")
    return errors


def main():
    out = Path("triage/dispositions")
    out.mkdir(exist_ok=True)
    for path in sorted(Path("inventory").glob("*.jsonl")):
        rows = list(map(json.loads, path.read_text().splitlines()))
        candidates = list(map(json.loads, (Path("candidates") / path.name).read_text().splitlines()))
        ledger = records(rows, candidates)
        (out / path.name).write_text("".join(json.dumps(r) + "\n" for r in ledger))
        print(f"{path.stem}: {len(ledger)} eligible, {sum(r['status'] == 'unresolved' for r in ledger)} unresolved")


if __name__ == "__main__":
    main()
