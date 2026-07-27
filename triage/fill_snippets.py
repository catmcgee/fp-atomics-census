"""Fill the ``snippet`` field of inventory rows verbatim from the pinned checkout.

    python -m triage.fill_snippets --repos-dir repos inventory/<engine>.jsonl

Rows whose ``snippet`` is empty or missing get the lines ``line_start`` to
``line_end`` of ``file`` in the checkout at the manifest sha. Rows that
already carry a snippet are left alone, so hand-trimmed snippets survive.
The sha in every row must match the checkout HEAD; the script refuses to
run otherwise, because the snippet would not correspond to the recorded
line numbers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos-dir", type=Path, default=Path("repos"))
    ap.add_argument("--force", action="store_true", help="refill snippets that are already present")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for f in args.files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        filled = 0
        for row in rows:
            if row.get("snippet") and not args.force:
                continue
            repo = args.repos_dir / row["engine"]
            head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            if head != row["sha"]:
                print(f"{f}: {row['id']}: checkout {head[:8]} != row sha {row['sha'][:8]}", file=sys.stderr)
                return 2
            src = (repo / row["file"]).read_text(errors="replace").split("\n")
            a, b = row["line_start"], row["line_end"]
            if b > len(src):
                print(f"{f}: {row['id']}: line_end {b} beyond file length {len(src)}", file=sys.stderr)
                return 2
            row["snippet"] = "\n".join(src[a - 1:b])
            filled += 1
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{f}: filled {filled} of {len(rows)} snippets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
