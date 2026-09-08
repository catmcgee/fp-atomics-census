"""Persist an invocation outcome even when a probe fails before initialisation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main(argv):
    if not argv:
        raise SystemExit("usage: python probes/invoke.py PROBE.py [arguments...]")
    root = Path(__file__).resolve().parent / "results"
    run_id = str(uuid.uuid4())
    out = root / "invocations" / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run([sys.executable, *argv], env={**os.environ, "PROBE_RUN_ID": run_id})
    statuses = []
    for path in root.glob(f"*/{run_id}/*.json"):
        record = json.loads(path.read_text())
        if isinstance(record, dict) and "probe" in record:
            statuses.append(record.get("verdict", "INVALID"))
    # A nonzero exit can mean a measured difference; absence of a measurement
    # or an explicit runtime failure is recorded separately.
    if not statuses or any(s in ("ERROR", "INVALID") for s in statuses):
        status = "ERROR"
    elif result.returncode and all(s == "bitwise-identical" for s in statuses):
        status = "ERROR"
    elif result.returncode:
        status = "NONIDENTICAL_OR_REJECTED"
    else:
        status = "COMPLETED"
    record = {"invocation_id": run_id, "started_at_utc": started, "argv": argv,
              "returncode": result.returncode, "status": status, "measurement_statuses": statuses}
    with (out / "status.json").open("x") as f:
        json.dump(record, f, indent=2)
    return result.returncode or int(status != "COMPLETED")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
