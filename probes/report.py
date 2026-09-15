"""Tabulate recorded runtime observations using the inventory's shared mapping."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triage.attach_runtime import campaign_rows, load_reports, related_rows, summarise


def main(argv: list[str]) -> int:
    root = ROOT / "probes/results"
    chosen = {Path(a).name for a in argv}
    for stack, probes in sorted(load_reports(root).items()):
        if chosen and stack not in chosen:
            continue
        print(f"\n### {stack}\n")
        print("Legacy fresh-process comparisons lack complete input and kernel provenance. Fresh-process pairs are formed within one campaign (reports sharing a probe source digest and recorded package versions; legacy reports have none). Evaluations include baselines; neither count implies statistical power.\n")
        print("| Probe | Campaign | In process | Fresh process | Evaluations | Evidence relation and related rows |")
        print("|---|---|---|---|---|---|")
        for name, campaign, _, reports in campaign_rows(probes):
            inside, fresh, count = summarise(reports)
            links = "; ".join(f"{rid} ({relation})" for rid, relation, _ in related_rows(name)) or "no source-site attribution"
            print(f"| {name} | {campaign} | {inside} | {fresh} | {count} | {links} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
