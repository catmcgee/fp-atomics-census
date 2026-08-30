"""Print the per-directory scan coverage of every engine as Markdown tables.

    python -m triage.coverage_table candidates/*.coverage.json

Distinguishes directories that were walked and scanned from those outside
the manifest's scan_paths (walked for counting only, not triaged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    files = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    for f in sorted(files):
        cov = json.loads(f.read_text())
        eng, sha = cov["engine"], cov["sha"][:8]
        t = cov["totals"]
        print(f"### {eng} @ {sha}\n")
        print(f"Files walked: {t['files_walked']}. Files scanned (CUDA/C++/Python): {t['files_scanned']}. "
              f"Candidates: {t['candidates']} ({t['candidates_in_scope']} in scope, {t['candidates_excluded']} flagged as comment, string, declaration or host).\n")
        print("| directory | in scope | files walked | files scanned | files with candidates | candidates |")
        print("|---|---|---|---|---|---|")
        for d, v in cov["directories"].items():
            if v["files_scanned"] == 0 and v["candidates"] == 0:
                continue
            print(f"| `{d}` | {'yes' if v['in_scope'] else 'no'} | {v['files_walked']} | {v['files_scanned']} | {v['files_with_candidates']} | {v['candidates']} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
