#!/usr/bin/env python3
"""Verify recovered C2 evidence and reproduce its comparisons without a GPU."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "SHA256.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    jobs = []
    for machine in range(1, 5):
        for path in sorted((ROOT / f"logs_m{machine}").glob("queue_*.log")):
            for line in path.read_text().splitlines():
                match = re.match(
                    r"QUEUE ARM_START (\S+) \S+ :: kind=(\S+) cache=(\S+) "
                    r"env='[^']*' caches_before=\S+ :: (.*)$", line)
                if match and match[2] in ("replay", "boundary"):
                    arm, output, *_ = match[4].split()
                    jobs.append((machine, match[1], match[2], match[3], arm, output))
    assert len(jobs) == len({job[1] for job in jobs}) == 37

    rates = defaultdict(lambda: {"identical": 0, "comparisons": 0, "differing_rows": []})
    results = []
    with tempfile.TemporaryDirectory(prefix="c2-reanalysis-") as temporary:
        target = Path(temporary)
        for machine in range(1, 5):
            shutil.copytree(ROOT / f"results_m{machine}", target / f"results_m{machine}")
        # A saved boundary comparison must not masquerade as a fresh result.
        for pattern in ("summary*.json", "boundary_*.json"):
            for path in target.rglob(pattern):
                path.unlink()
        for machine, label, kind, cache, arm, output in jobs:
            relative = Path(f"results_m{machine}/e6") / arm
            run = target / relative
            subprocess.run(
                [sys.executable, str(ROOT / "src/probes/shape/run_e6.py"),
                 "--compare", str(run / "record"), str(run / output)],
                check=True, capture_output=True, text=True)
            name = f"{output}.json" if kind == "boundary" else f"summary_{output}.json"
            actual = (run / name).read_bytes()
            assert actual == (ROOT / relative / name).read_bytes(), (label, name)
            summary = json.loads(actual)
            assert not summary["validation_errors"], label
            if kind == "replay":
                assert summary["requirements_met"], label
                assert summary["verdict_P2"] in ("IDENTICAL", "DIFFERS"), label
            else:
                assert summary["verdict_P2_boundary"] in ("IDENTICAL", "DIFFERS"), label
            item = {"label": label, "arm": arm, "replay": output,
                    "machine": machine, "cache": cache,
                    "rows_compared": summary["rows_compared"],
                    "rows_differing": summary["rows_differing"]}
            results.append(item)
            if cache == "cold" and kind == "replay":
                before = ROOT / f"logs_m{machine}/{label}.cache_before.txt"
                assert "CACHE_TOTAL_FILES 0" in before.read_text().splitlines(), label
                rate = rates[arm]
                rate["comparisons"] += 1
                rate["identical"] += summary["rows_differing"] == 0
                rate["differing_rows"].append(summary["rows_differing"])

    report = {"files_verified": len(manifest), "summaries_byte_identical": len(results),
              "cold_replays": dict(sorted(rates.items())), "comparisons": results}
    expected = json.loads((ROOT / "comparisons.json").read_text())
    assert report["cold_replays"] == expected["cold_replays"]
    assert results == expected["comparisons"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
