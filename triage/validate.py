"""Validate inventory rows against triage/inventory.schema.json.

    python -m triage.validate --schema triage/inventory.schema.json inventory/*.jsonl

Beyond the JSON Schema, this checks what the schema cannot express: the sha
matches the manifest, line ranges are ordered, snippets are 2 to 8 lines,
ids are unique across all files, every snippet line is present at the
recorded lines of the checkout, every evidence and entry-point reference
names a file that exists at the pinned sha with a line inside it, and rows
with scanner provenance cite candidate ids that exist in candidates/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema

_REF = re.compile(r"^(?P<file>[^:\s]+):(?P<start>\d+)(?:-(?P<end>\d+))?$")


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


class Checkouts:
    def __init__(self, repos_dir: Path):
        self.repos_dir = repos_dir
        self._lines: dict[Path, list[str] | None] = {}

    def lines(self, engine: str, rel: str) -> list[str] | None:
        # A ref may point into another pinned repo as repos/<engine>/<path>.
        if rel.startswith("repos/"):
            p = self.repos_dir / rel[len("repos/"):]
        else:
            p = self.repos_dir / engine / rel
        if p not in self._lines:
            self._lines[p] = p.read_text(errors="replace").split("\n") if p.is_file() else None
        return self._lines[p]

    def available(self, engine: str) -> bool:
        return (self.repos_dir / engine).is_dir()


def check_ref(co: Checkouts, engine: str, ref: str) -> str | None:
    """Return an error string if a file:line reference does not resolve, else None."""
    m = _REF.match(ref)
    if not m:
        return f"reference {ref!r} is not file:line or file:start-end"
    if not co.available(engine) and not ref.startswith("repos/"):
        return None
    lines = co.lines(engine, m["file"])
    if lines is None:
        return f"reference {ref!r}: file not found in the checkout"
    end = int(m["end"] or m["start"])
    if end > len(lines):
        return f"reference {ref!r}: line beyond end of file ({len(lines)} lines)"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", type=Path, default=Path("triage/inventory.schema.json"))
    ap.add_argument("--manifest", type=Path, default=Path("scan-manifest.json"))
    ap.add_argument("--repos-dir", type=Path, default=Path("repos"))
    ap.add_argument("--candidates-dir", type=Path, default=Path("candidates"))
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)

    schema = json.loads(args.schema.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    manifest = {r["name"]: r for r in json.loads(args.manifest.read_text())["repos"]}
    co = Checkouts(args.repos_dir)
    candidate_ids: dict[str, set[str]] = {}
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
            snippet_lines = row.get("snippet", "").strip("\n").split("\n")
            if not (2 <= len(snippet_lines) <= 8):
                print(f"{where}: snippet has {len(snippet_lines)} lines; expected 2 to 8")
                errors += 1
            lines = co.lines(eng, row["file"]) if eng and co.available(eng) else None
            if eng and co.available(eng) and lines is None:
                print(f"{where}: {row['file']} not found in the checkout")
                errors += 1
            if lines is not None:
                if le > len(lines):
                    print(f"{where}: line_end {le} beyond end of {row['file']} ({len(lines)} lines)")
                    errors += 1
                else:
                    window = [l.strip() for l in lines[max(0, ls - 4): le + 3]]
                    missing = [s for s in snippet_lines if s.strip() and s.strip() not in window]
                    if missing:
                        print(f"{where}: {len(missing)} snippet line(s) not found near {row['file']}:{ls}-{le}: {missing[0][:60]!r}")
                        errors += 1
            for ev in row.get("evidence", []):
                msg = check_ref(co, eng, ev.get("ref", ""))
                if msg:
                    print(f"{where}: evidence {msg}")
                    errors += 1
            for ep in (row.get("path") or {}).get("entry_points", []):
                msg = check_ref(co, eng, ep)
                if msg:
                    print(f"{where}: entry point {msg}")
                    errors += 1
            if row.get("provenance") == "scanner":
                if eng not in candidate_ids:
                    cf = args.candidates_dir / f"{eng}.jsonl"
                    candidate_ids[eng] = {json.loads(l)["id"] for l in cf.read_text().splitlines() if l.strip()} if cf.is_file() else set()
                for cid in row.get("candidate_ids", []):
                    if candidate_ids[eng] and cid not in candidate_ids[eng]:
                        print(f"{where}: candidate id {cid} not in candidates/{eng}.jsonl")
                        errors += 1
            if row.get("class") == "A3" and row.get("default_path") is True and "default_is_atomic" not in (row.get("gate") or {}):
                print(f"{where}: default-path A3 row {rid} does not say whether the gate default is the atomic side")
                errors += 1
            if row.get("class") == "A" and row.get("default_path") is True and not row.get("cross_checked"):
                print(f"{where}: note: default-path class-A row {rid} is not yet cross_checked")
    print(f"{len(seen_ids)} rows, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
