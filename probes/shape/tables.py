"""Regenerate the batch-shape tables from probes/shape/results/.

    python probes/shape/tables.py [probes/shape/results]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent / "results"
    for exp in ("e2", "e3", "e4"):
        d = root / exp
        if not d.exists():
            continue
        print(f"\n### {exp.upper()}\n")
        if exp == "e2":
            print("| Arm | Repeats | Hook steps | Distinct shape vectors | Slots with >1 output hash | Histories mapping to >1 hash | P1 |")
            print("|---|---|---|---|---|---|---|")
        elif exp == "e3":
            print("| Arm | Steps | All steps identical | Outputs identical | Differing steps | P2 |")
            print("|---|---|---|---|---|---|")
        else:
            print("| Arm | Repeats | Shapes equal | Target hidden identical across kinds | Target tokens identical | MoE counts changed | First divergent step | P3 |")
            print("|---|---|---|---|---|---|---|---|")
        for s in sorted(d.glob("*/summary.json")):
            j = json.loads(s.read_text())
            if exp == "e2":
                n_multi = sum(1 for v in j["distinct_output_hashes_per_slot"].values() if v > 1)
                print(f"| {j['arm']} | {j['repeats']} | {j['hook_steps']} | {j['distinct_step_shape_vectors']} | {n_multi} | {len(j['history_to_multiple_hashes'])} | {j['verdict_P1']} |")
            elif exp == "e3":
                print(f"| {j['arm']} | {j['steps_compared']} | {j['all_steps_identical']} | {j['outputs_identical']} | {j['differing_steps'][:5]} | {j['verdict_P2']} |")
            else:
                a = j["target_across_kinds"]
                print(f"| {j['arm']} | {j['repeats']} | {j['shape_histories_equal_across_all_runs']} | {a['hidden']} | {a['tokens']} | {j['moe_first_layer_expert_counts_changed']} | {j['first_divergent_step']} | {j['verdict_P3']} |")
    for f in sorted(root.glob("e3/*/e5_vs_*.json")):
        j = json.loads(f.read_text())
        print(f"\nE5 {f.parent.name}: {j['a']['gpu']} vs {j['b']['gpu']}: hidden identical {j['hidden_identical_steps']}/{j['steps']} steps, argmax identical {j['argmax_identical_steps']}/{j['steps']}: {j['verdict_P5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
