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
import subprocess
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
    def __init__(self, repos_dir: Path, manifest: dict | None = None):
        self.repos_dir = repos_dir
        self.manifest = manifest or {r["name"]: r for r in json.loads(Path("scan-manifest.json").read_text())["repos"]}
        self._lines = {}
        self._heads = {}

    def lines(self, engine: str, rel: str) -> list[str] | None:
        if rel.startswith("repos/"):
            engine, rel = rel[len("repos/"):].split("/", 1)
        if engine not in self.manifest or Path(rel).is_absolute() or ".." in Path(rel).parts:
            return None
        key = (engine, rel)
        if key not in self._lines:
            try:
                blob = subprocess.check_output(["git", "-C", str(self.repos_dir / engine), "show",
                                                f"{self.manifest[engine]['sha']}:{rel}"], stderr=subprocess.DEVNULL)
                self._lines[key] = blob.decode(errors="replace").split("\n")
            except (OSError, subprocess.CalledProcessError):
                self._lines[key] = None
        return self._lines[key]

    def available(self, engine: str) -> bool:
        if engine not in self._heads:
            try:
                self._heads[engine] = subprocess.check_output(["git", "-C", str(self.repos_dir / engine), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                self._heads[engine] = None
        return engine in self.manifest and self._heads[engine] == self.manifest[engine]["sha"]


def candidate_matches(row: dict, candidate: dict) -> bool:
    if candidate.get("engine") != row.get("engine") or candidate.get("sha") != row.get("sha") or candidate.get("file") != row.get("file"):
        return False
    refs = [f"{row['file']}:{row['line_start']}-{row['line_end']}"] + [e["ref"] for e in row.get("evidence", [])]
    return any(m and m["file"] == candidate["file"] and int(m["start"]) <= candidate["line"] <= int(m["end"] or m["start"])
               for ref in refs if (m := _REF.match(ref)))


def check_ref(co: Checkouts, engine: str, ref: str) -> str | None:
    """Return an error string if a file:line reference does not resolve, else None."""
    m = _REF.match(ref)
    if not m:
        return f"reference {ref!r} is not file:line or file:start-end"
    start, end = int(m["start"]), int(m["end"] or m["start"])
    if start < 1 or end < start:
        return f"reference {ref!r}: invalid line range"
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
    ap.add_argument("--schema-only", action="store_true", help="explicitly omit source, candidate and disposition checks")
    ap.add_argument("--dispositions-dir", type=Path)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)

    schema = json.loads(args.schema.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    manifest = {r["name"]: r for r in json.loads(args.manifest.read_text())["repos"]}
    co = Checkouts(args.repos_dir, manifest)
    candidate_ids: dict[str, set[str]] = {}
    errors = 0
    seen_ids: dict[str, str] = {}
    all_rows = []
    checked_engines = set()
    for f in args.files:
        rows = load_rows(f)
        all_rows.extend(rows)
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
            if not args.schema_only and eng not in checked_engines:
                checked_engines.add(eng)
                if not co.available(eng):
                    print(f"{where}: missing checkout or HEAD does not match pinned manifest for {eng}")
                    errors += 1
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
            lines = co.lines(eng, row["file"]) if eng and not args.schema_only else None
            if eng and not args.schema_only and lines is None:
                print(f"{where}: {row['file']} not found in the checkout")
                errors += 1
            if lines is not None:
                if le > len(lines):
                    print(f"{where}: line_end {le} beyond end of {row['file']} ({len(lines)} lines)")
                    errors += 1
                else:
                    if row["snippet"] != "\n".join(lines[ls - 1:le]):
                        print(f"{where}: snippet is not the exact, ordered pinned source range {row['file']}:{ls}-{le}")
                        errors += 1
            for ev in row.get("evidence", []):
                msg = None if args.schema_only else check_ref(co, eng, ev.get("ref", ""))
                if msg:
                    print(f"{where}: evidence {msg}")
                    errors += 1
            for ep in (row.get("path") or {}).get("entry_points", []):
                msg = None if args.schema_only else check_ref(co, eng, ep)
                if msg:
                    print(f"{where}: entry point {msg}")
                    errors += 1
            if row.get("provenance") == "scanner" and not args.schema_only:
                if eng not in candidate_ids:
                    cf = args.candidates_dir / f"{eng}.jsonl"
                    candidate_ids[eng] = {c["id"]: c for c in load_rows(cf)} if cf.is_file() else {}
                    if not cf.is_file():
                        print(f"{where}: missing candidate file {cf}")
                        errors += 1
                for cid in row.get("candidate_ids", []):
                    if cid not in candidate_ids[eng]:
                        print(f"{where}: candidate id {cid} not in candidates/{eng}.jsonl")
                        errors += 1
                    elif not candidate_matches(row, candidate_ids[eng][cid]):
                        print(f"{where}: candidate {cid} does not match the pinned file and cited source range")
                        errors += 1
            if row.get("gate") and row["gate"].get("scope") != "external_runtime" and not args.schema_only:
                msg = check_ref(co, eng, row["gate"]["read_at"])
                if msg:
                    print(f"{where}: gate {msg}")
                    errors += 1
            if row.get("class") == "A3" and row.get("default_path") is True and "default_is_atomic" not in (row.get("gate") or {}):
                print(f"{where}: default-path A3 row {rid} does not say whether the gate default is the atomic side")
                errors += 1
            if row.get("class") == "A" and row.get("default_path") is True and not row.get("cross_checked"):
                print(f"{where}: note: default-path class-A row {rid} is not yet cross_checked")
    if args.dispositions_dir and not args.schema_only:
        from triage.dispositions import validate_dispositions
        for message in validate_dispositions(all_rows, args.candidates_dir, args.dispositions_dir):
            print(message)
            errors += 1
    print(f"{len(seen_ids)} rows, {errors} errors" + (" (schema only; source provenance NOT checked)" if args.schema_only else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
