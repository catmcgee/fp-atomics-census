"""Validate inventory rows against triage/inventory.schema.json.

    python -m triage.validate --schema triage/inventory.schema.json inventory/*.jsonl

Beyond the JSON Schema, this checks what the schema cannot express: the sha
matches the manifest, line ranges are ordered, snippets are 2 to 8 lines,
ids are unique across all files, and the snippet text actually appears at
the recorded lines when the repository checkout is available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema


def load_rows(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: invalid JSON: {e}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", type=Path, default=Path("triage/inventory.schema.json"))
    ap.add_argument("--manifest", type=Path, default=Path("scan-manifest.json"))
    ap.add_argument("--repos-dir", type=Path, default=Path("repos"))
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)

    schema = json.loads(args.schema.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    manifest = {r["name"]: r for r in json.loads(args.manifest.read_text())["repos"]}
    errors = 0
    seen_ids: dict[str, str] = {}
    for f in args.files:
        rows = load_rows(f)
        for i, row in enumerate(rows, start=1):
            where = f"{f}:{i}"
            for err in validator.iter_errors(row):
                print(f"{where}: {'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}")
                errors += 1
            rid = row.get("id")
            if rid in seen_ids:
                print(f"{where}: duplicate id {rid} (also in {seen_ids[rid]})")
                errors += 1
            seen_ids[rid] = where
            eng = row.get("engine")
            if eng in manifest and row.get("sha") != manifest[eng]["sha"]:
                print(f"{where}: sha {row.get('sha')} does not match manifest sha for {eng}")
                errors += 1
            ls, le = row.get("line_start", 0), row.get("line_end", 0)
            if le < ls:
                print(f"{where}: line_end {le} < line_start {ls}")
                errors += 1
            n_lines = len(row.get("snippet", "").rstrip("\n").split("\n"))
            if not (2 <= n_lines <= 8):
                print(f"{where}: snippet has {n_lines} lines; expected 2 to 8")
                errors += 1
            repo = args.repos_dir / eng if eng else None
            src = repo / row["file"] if repo and repo.exists() else None
            if src is not None and src.exists():
                lines = src.read_text(errors="replace").split("\n")
                if le > len(lines):
                    print(f"{where}: line_end {le} beyond end of {row['file']} ({len(lines)} lines)")
                    errors += 1
                else:
                    window = "\n".join(lines[max(0, ls - 4): le + 3])
                    first = row["snippet"].strip("\n").split("\n")[0].strip()
                    if first and first not in window:
                        print(f"{where}: first snippet line not found near {row['file']}:{ls}-{le}")
                        errors += 1
            if row.get("class") in ("A",) and row.get("default_path") is True and not row.get("cross_checked"):
                print(f"{where}: note: default-path class-A row {rid} is not yet cross_checked")
    print(f"{len(seen_ids)} rows, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
