"""Clone or update the repositories in scan-manifest.json at their pinned shas.

    python -m scan.clone --manifest scan-manifest.json --dest repos [--only vllm] [--sha <sha>]

Uses a depth-1 fetch of the exact commit so the clone is small and the
checkout is byte-identical to what the inventory line numbers refer to.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure(name: str, url: str, sha: str, dest: Path) -> None:
    repo = dest / name
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], cwd=repo)
        run(["git", "remote", "add", "origin", url], cwd=repo)
    head = ""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        pass
    if head == sha:
        print(f"{name}: already at {sha[:8]}")
        return
    for attempt in range(1, 5):
        try:
            run(["git", "fetch", "-q", "--depth", "1", "origin", sha], cwd=repo)
            break
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            print(f"{name}: fetch attempt {attempt} failed, retrying", file=sys.stderr)
    run(["git", "checkout", "-q", "--detach", sha], cwd=repo)
    print(f"{name}: checked out {sha[:8]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("scan-manifest.json"))
    ap.add_argument("--dest", type=Path, default=Path("repos"))
    ap.add_argument("--only", action="append")
    ap.add_argument("--sha", help="override the manifest sha (only with a single --only)")
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    for repo in manifest["repos"]:
        if args.only and repo["name"] not in args.only:
            continue
        sha = args.sha if (args.sha and args.only and len(args.only) == 1) else repo["sha"]
        ensure(repo["name"], repo["url"], sha, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
