#!/bin/sh
set -eu

evidence_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$evidence_dir/../../../.." && pwd)
audit_tmp=$(mktemp -d "${TMPDIR:-/tmp}/moe_compile_audit.XXXXXX")
trap 'rm -rf "$audit_tmp"' EXIT HUP INT TERM

# source_digest() emits repository-relative paths when run from the repository root.
# This keeps the fresh report portable and directly comparable to the retained one.
cd "$repo_root"

python3 "$evidence_dir/source/moe_compile_audit.py" \
  --input "cu129_v2=$evidence_dir/cu129_v2_graphs_off_nodump" \
  --input "cu130_v2=$evidence_dir/cu130_v2_graphs_off_nodump" \
  --input "cu130_v1=$evidence_dir/cu130_v1_graphs_off_nodump" \
  --input "cu130_v2_repeat=$evidence_dir/cu130_v2_graphs_off_cold_repeat" \
  --input "cu129_v2_repeat=$evidence_dir/cu129_v2_graphs_off_cold_repeat" \
  --json-out "$audit_tmp/moe_compile_final.audit.json" \
  --markdown-out "$audit_tmp/moe_compile_final.audit.md" \
  --historical-e4-root "$repo_root/probes/shape/results/e4" \
  --historical-out "$audit_tmp/historical_e4_output_comparison.json" \
  >/dev/null

python3 "$evidence_dir/source/moe_compile_audit.py" \
  --input "cu130_debugdump_failed=$evidence_dir/cu130_v2_graphs_off_debugdump_failed" \
  --json-out "$audit_tmp/cu130_debugdump_failures.audit.json" \
  --markdown-out "$audit_tmp/cu130_debugdump_failures.audit.md" \
  >/dev/null

python3 - \
  "$evidence_dir/moe_compile_final.audit.json" \
  "$audit_tmp/moe_compile_final.audit.json" \
  "$evidence_dir/cu130_debugdump_failures.audit.json" \
  "$audit_tmp/cu130_debugdump_failures.audit.json" <<'PY'
import json
import sys
from pathlib import Path

pairs = zip(sys.argv[1::2], sys.argv[2::2], strict=True)
for retained_name, fresh_name in pairs:
    retained = json.loads(Path(retained_name).read_text())
    fresh = json.loads(Path(fresh_name).read_text())
    retained.pop("created_at", None)
    fresh.pop("created_at", None)
    if retained != fresh:
        raise SystemExit(
            f"audit mismatch after ignoring only top-level created_at: {retained_name}"
        )
PY

for name in \
  moe_compile_final.audit.md \
  historical_e4_output_comparison.json \
  cu130_debugdump_failures.audit.md
do
  if ! cmp -s "$evidence_dir/$name" "$audit_tmp/$name"; then
    echo "audit mismatch: $name" >&2
    exit 1
  fi
done

python3 - "$audit_tmp/moe_compile_final.audit.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
print(
    "verified retained audit: "
    f"state={report['evidence_state']} "
    f"cells={len(report['cells'])} "
    f"comparisons={len(report['comparisons'])}"
)
PY
