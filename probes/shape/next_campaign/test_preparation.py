#!/usr/bin/env python3
"""CPU-only checks for the C2 follow-up preparation."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from preflight import weight_index
from validate_moe_routing import real_rows, validate_rows


REPO = HERE.parents[2]
HISTORICAL_MOE = (
    REPO
    / "probes/shape/campaigns/2026-09-15-h100-e6-modern/results_m1/e6"
    / "Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs1_prefix0/record/hook/rank0.jsonl"
)


class RoutingAuditTest(unittest.TestCase):
    def test_historical_full_graph_nulls_fail(self):
        records = [json.loads(line) for line in HISTORICAL_MOE.read_text().splitlines() if line.strip()]
        rows = real_rows(records)
        errors, sources = validate_rows(rows, experts=60, top_k=4, allow_estimated=False)
        self.assertEqual(len(rows), 37)
        self.assertEqual(sources, {None: 37})
        self.assertTrue(any("resolved compile/graphs" in error for error in errors))
        self.assertTrue(any("count vector" in error for error in errors))

    def test_synthetic_eager_actual_counts_pass(self):
        rows = []
        for _ in range(37):
            rows.append(
                {
                    "resolved_compile": "0",
                    "resolved_cudagraph": "NONE",
                    "dispatch": {"cudagraph_mode": "NONE"},
                    "total_scheduled": 3,
                    "moe_counts_source": "actual_router",
                    "moe_expert_counts_first_layer": [4, 2],
                }
            )
        errors, sources = validate_rows(rows, experts=2, top_k=2, allow_estimated=False)
        self.assertEqual(errors, [])
        self.assertEqual(sources, {"actual_router": 37})

    def test_noninteger_counts_fail_without_crashing(self):
        rows = [
            {
                "resolved_compile": "0",
                "resolved_cudagraph": "NONE",
                "dispatch": None,
                "total_scheduled": 1,
                "moe_counts_source": "actual_router",
                "moe_expert_counts_first_layer": [1, "bad"],
            }
        ]
        errors, _ = validate_rows(rows, experts=2, top_k=1, allow_estimated=False)
        self.assertTrue(any("non-negative-integer" in error for error in errors))


class WeightAuditTest(unittest.TestCase):
    A = ("org/a", "rev-a")
    B = ("org/b", "rev-b")

    def test_missing_required_model_fails(self):
        rows = [
            {
                "repo": self.A[0],
                "revision": self.A[1],
                "file": "model.safetensors",
                "size": 1,
                "sha256": "a" * 64,
            }
        ]
        _, errors = weight_index(rows, {self.A, self.B}, "fixture")
        self.assertTrue(any("no rows for required model" in error and "org/b" in error for error in errors))

    def test_duplicate_and_bad_hash_fail(self):
        row = {
            "repo": self.A[0],
            "revision": self.A[1],
            "file": "model.safetensors",
            "size": 1,
            "sha256": "not-a-sha256",
        }
        _, errors = weight_index([row, dict(row)], {self.A}, "fixture")
        self.assertTrue(any("invalid SHA-256" in error for error in errors))
        self.assertTrue(any("duplicate row" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
