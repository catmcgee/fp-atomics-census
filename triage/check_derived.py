"""Check generated runtime attachments and CSV without changing tracked files."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

from triage.attach_runtime import main as attach

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    paths = sorted((ROOT / "inventory").glob("*.jsonl"))
    with tempfile.TemporaryDirectory(prefix="inventory-check-") as temp:
        copies = []
        for path in paths:
            copy = Path(temp) / path.name
            copy.write_bytes(path.read_bytes())
            copies.append(copy)
        with contextlib.redirect_stdout(io.StringIO()):
            attach(["--results", str(ROOT / "probes/results"), *map(str, copies)])
        for original, copy in zip(paths, copies):
            if original.read_bytes() != copy.read_bytes():
                errors.append(f"stale runtime attachments: {original.name}")
    csv = subprocess.check_output([sys.executable, "-m", "triage.to_csv", *map(str, paths)], cwd=ROOT)
    if csv != (ROOT / "inventory.csv").read_bytes():
        errors.append("inventory.csv is stale")
    print("\n".join(errors) if errors else "Runtime attachments and CSV are current.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
