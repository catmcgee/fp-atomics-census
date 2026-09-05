"""Run the candidate scanner over the repositories in scan-manifest.json.

    python -m scan.run --manifest scan-manifest.json --repos-dir repos --out candidates [--only vllm]

Writes ``<out>/<engine>.jsonl`` (one candidate per line, sorted by file and
line, ids stable for a given sha) and ``<out>/<engine>.coverage.json`` (files
walked and scanned per directory, so that "no atomics found" can be told
apart from "not scanned").
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .scanner import CUDA_EXTENSIONS, PYTHON_EXTENSIONS, Candidate, scan_path, scan_wrapper_calls, wrapper_definitions

SKIP_DIRS = {".git", ".github", "__pycache__", "node_modules", ".venv", "build", "dist"}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _in_scope(rel: str, scan_paths: list[str], excludes: list[str]) -> bool:
    if any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in excludes):
        return False
    return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in scan_paths)


def _coverage_key(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) == 1:
        return "."
    depth = 2 if parts[0] in ("include", "csrc", "python", "vllm", "sgl-kernel", "flashinfer", "deep_gemm", "deep_ep") and len(parts) > 2 else 1
    return "/".join(parts[:depth])


def scan_repo(name: str, repo: Path, sha: str, scan_paths: list[str], excludes: list[str],
              languages: set[str], python_scanner=None) -> tuple[list[Candidate], dict]:
    cands: list[Candidate] = []
    cov: dict[str, dict] = defaultdict(lambda: {"files_walked": 0, "files_scanned": 0, "files_with_candidates": 0,
                                                "candidates": 0, "in_scope": False, "skipped_extensions": defaultdict(int)})
    sources: dict[str, bytes] = {}
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(files):
            p = Path(root) / f
            rel = str(p.relative_to(repo))
            key = _coverage_key(rel)
            entry = cov[key]
            entry["files_walked"] += 1
            scope = _in_scope(rel, scan_paths, excludes)
            entry["in_scope"] = entry["in_scope"] or scope
            ext = p.suffix.lower()
            if ext in CUDA_EXTENSIONS and "cuda" in languages:
                entry["files_scanned"] += 1
                found = scan_path(p, rel, name, sha)
                if found:
                    sources[rel] = p.read_bytes()
                for c in found:
                    c.in_scope = scope
                cands.extend(found)
                if found:
                    entry["files_with_candidates"] += 1
                    entry["candidates"] += len(found)
            elif ext in PYTHON_EXTENSIONS and "python" in languages and python_scanner is not None:
                entry["files_scanned"] += 1
                found = python_scanner(p, rel, name, sha)
                for c in found:
                    c.in_scope = scope
                cands.extend(found)
                if found:
                    entry["files_with_candidates"] += 1
                    entry["candidates"] += len(found)
            elif ext in CUDA_EXTENSIONS | PYTHON_EXTENSIONS:
                entry["skipped_extensions"][ext] += 1
            else:
                entry["skipped_extensions"][ext or "<none>"] += 1

    _assign_ids(name, cands)

    # second pass: call sites of small device helpers that wrap a float atomic
    defs = wrapper_definitions(cands)
    if defs and "cuda" in languages:
        extra: list[Candidate] = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(files):
                p = Path(root) / f
                if p.suffix.lower() not in CUDA_EXTENSIONS:
                    continue
                rel = str(p.relative_to(repo))
                src = sources.get(rel) or p.read_bytes()
                found = scan_wrapper_calls(src, rel, name, sha, defs,
                                           language="cuda" if p.suffix.lower() in (".cu", ".cuh") else "cpp")
                for c in found:
                    c.in_scope = _in_scope(rel, scan_paths, excludes)
                extra.extend(found)
        cands.extend(extra)
        _assign_ids(name, cands)
        for k in cov:
            pass
        for c in extra:
            cov[_coverage_key(c.file)]["candidates"] += 1

    coverage = {
        "engine": name, "sha": sha, "scan_paths": scan_paths, "exclude": excludes,
        "languages": sorted(languages),
        "directories": {k: {**v, "skipped_extensions": dict(v["skipped_extensions"])} for k, v in sorted(cov.items())},
        "totals": {
            "files_walked": sum(v["files_walked"] for v in cov.values()),
            "files_scanned": sum(v["files_scanned"] for v in cov.values()),
            "candidates": len(cands),
            "candidates_in_scope": sum(1 for c in cands if c.in_scope),
            "candidates_excluded": sum(1 for c in cands if c.excluded_reason),
            "wrapper_definitions": {k: v.id for k, v in defs.items()},
        },
    }
    return cands, coverage


def _assign_ids(name: str, cands: list[Candidate]) -> None:
    cands.sort(key=lambda c: (c.file, c.line, c.col, c.pattern))
    for i, c in enumerate(cands, start=1):
        c.id = f"{name}-c{i:05d}"


def write_outputs(out_dir: Path, name: str, cands: list[Candidate], coverage: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.jsonl").open("w") as fh:
        for c in cands:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / f"{name}.coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("scan-manifest.json"))
    ap.add_argument("--repos-dir", type=Path, default=Path("repos"))
    ap.add_argument("--out", type=Path, default=Path("candidates"))
    ap.add_argument("--only", action="append", help="engine name (repeatable)")
    ap.add_argument("--sha", help="override the manifest sha for --only (must match the checkout)")
    ap.add_argument("--languages", default="cuda,python", help="comma separated: cuda,python")
    ap.add_argument("--allow-sha-mismatch", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    languages = {s.strip() for s in args.languages.split(",") if s.strip()}
    python_scanner = None
    if "python" in languages:
        try:
            from .py import scan_python_path as python_scanner  # noqa: F401
        except ImportError:
            print("python scanner not available; scanning cuda only", file=sys.stderr)
            languages.discard("python")

    for repo in manifest["repos"]:
        name = repo["name"]
        if args.only and name not in args.only:
            continue
        sha = args.sha if (args.sha and args.only and len(args.only) == 1) else repo["sha"]
        path = args.repos_dir / name
        if not path.exists():
            print(f"{name}: not cloned at {path}; run make clone", file=sys.stderr)
            return 2
        head = git_head(path)
        if head != sha:
            msg = f"{name}: checkout is {head} but manifest/--sha says {sha}"
            if not args.allow_sha_mismatch:
                print(msg, file=sys.stderr)
                return 3
            print(msg + " (continuing)", file=sys.stderr)
        cands, coverage = scan_repo(name, path, sha, repo.get("scan_paths", []), repo.get("exclude", []),
                                    languages, python_scanner)
        write_outputs(args.out, name, cands, coverage)
        t = coverage["totals"]
        print(f"{name} @ {sha[:8]}: {t['files_scanned']} files scanned, {t['candidates']} candidates "
              f"({t['candidates_in_scope']} in scope, {t['candidates_excluded']} flagged as comment/string/host)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
