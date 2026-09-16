#!/usr/bin/env python3
"""CPU-only tests for graph-safe routing telemetry evidence handling."""
from __future__ import annotations

import hashlib
import unittest
import json
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from routing_harness import (
    _handoff_manifest,
    _label,
    _shutdown_llm,
    _wait_ack,
    canonical_json,
    run_condition,
    write_json,
)
from routing_telemetry import (
    RoutingTelemetryError,
    compare_hook_hashes,
    compare_matched_samples,
    hook_evidence,
    summarise_selected_experts,
    validate_graph_dispatch,
    validate_dynamic_inputs,
)


class RoutingTelemetryTest(unittest.TestCase):
    def test_condition_forces_and_resolves_synchronous_scheduling(self):
        observed = {}

        class Core:
            def shutdown(self):
                observed["shutdown"] = True

        class FakeLLM:
            def __init__(self, **kwargs):
                observed["kwargs"] = kwargs
                scheduler = SimpleNamespace(async_scheduling=kwargs["async_scheduling"])
                config = SimpleNamespace(scheduler_config=scheduler)
                self.llm_engine = SimpleNamespace(
                    vllm_config=config, engine_core=Core()
                )

            def generate(self, prompts, params, use_tqdm):
                del params, use_tqdm
                return [
                    SimpleNamespace(
                        prompt_token_ids=[index + 1],
                        outputs=[SimpleNamespace(token_ids=[index + 10])],
                    )
                    for index, _ in enumerate(prompts)
                ]

        fake = ModuleType("vllm")
        fake.LLM = FakeLLM
        fake.SamplingParams = lambda **kwargs: kwargs
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "sys.modules", {"vllm": fake}
        ):
            output = Path(temporary) / "condition.json"
            result = run_condition(
                condition="graph-control",
                model="model",
                revision="revision",
                prompts=["a", "b"],
                max_tokens=1,
                cudagraph_mode="FULL",
                enable_telemetry=False,
                num_experts=60,
                num_layers=24,
                top_k=4,
                shape_hook_root=None,
                condition_output=output,
                source={"attested": True},
            )
        self.assertIs(observed["kwargs"]["async_scheduling"], False)
        self.assertIs(result["engine_controls"]["async_scheduling"], False)
        self.assertTrue(observed["shutdown"])

    def test_shutdown_uses_vllm_029_engine_core_api(self):
        calls = []

        class Core:
            def shutdown(self):
                calls.append("core")

        class Engine:
            engine_core = Core()

        class LLM:
            llm_engine = Engine()

        self.assertEqual(
            _shutdown_llm(LLM()),
            "LLM.llm_engine.engine_core.shutdown",
        )
        self.assertEqual(calls, ["core"])

    def test_shutdown_prefers_public_api_and_tolerates_no_api(self):
        calls = []

        class PublicLLM:
            def shutdown(self):
                calls.append("public")

        self.assertEqual(_shutdown_llm(PublicLLM()), "LLM.shutdown")
        self.assertIsNone(_shutdown_llm(object()))
        self.assertEqual(calls, ["public"])

    def test_atomic_json_writer_persists_exact_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "condition.json"
            value = {"status": "complete", "samples": [1, 2]}
            write_json(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_handoff_manifest_and_exact_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acks").mkdir()
            condition = root / "report.conditions" / "graph-control.json"
            write_json(condition, {"status": "complete"})
            hook = root / "hook" / "graph-control"
            hook.mkdir(parents=True)
            (hook / "rank0.jsonl").write_text("{}\n")
            label = _label(root / "report.json", "graph-control")
            self.assertEqual(label, "report_graph_control")
            manifest = _handoff_manifest(root, label, [condition, hook])
            value = json.loads(manifest.read_text())
            payload = {key: value[key] for key in ("schema", "label", "files")}
            self.assertEqual(value["sha256"], hashlib.sha256(canonical_json(payload)).hexdigest())
            self.assertEqual(
                [row["path"] for row in value["files"]],
                ["hook/graph-control/rank0.jsonl", "report.conditions/graph-control.json"],
            )
            ack = {
                "schema": 1,
                "label": label,
                "handoff_manifest_sha256": value["sha256"],
                "verified_manifest_sha256": value["sha256"],
            }
            write_json(root / "acks" / f"{label}.json", ack)
            _wait_ack(root, label, manifest)

    def test_trims_capture_padding_without_counting_it(self):
        routing = [
            [[0, 1], [2, 3]],
            [[1, 2], [3, 0]],
            [[3, 3], [3, 3]],  # static capture allocation tail, not a real token
        ]
        summary = summarise_selected_experts(routing, num_experts=4, valid_tokens=2)
        self.assertEqual(summary["capture_padding_trimmed"], 1)
        self.assertEqual(summary["per_layer_expert_counts"], [[1, 2, 1, 0], [1, 0, 1, 2]])
        self.assertEqual(summary["selected_id_count"], 8)
        self.assertEqual(summary["per_layer_count_sums"], [4, 4])
        self.assertEqual(summary["per_layer_distinct_experts"], [3, 3])
        self.assertEqual(summary["selected_ids"], routing[:2])

    def test_rejects_out_of_range_actual_id(self):
        with self.assertRaises(RoutingTelemetryError):
            summarise_selected_experts([[[4]]], num_experts=4, valid_tokens=1)

    def test_dynamic_validation_requires_changed_output_routing_and_counts(self):
        rows = [
            {"prompt_index": 0, "prompt_token_count": 1, "generated_token_count": 3, "output_token_ids_sha256": "a", "routing": {"selected_ids_sha256": "ra", "per_layer_expert_counts": [[2, 0]], "selected_ids": [[[0]], [[0]], [[1]]]}},
            {"prompt_index": 1, "prompt_token_count": 1, "generated_token_count": 3, "output_token_ids_sha256": "b", "routing": {"selected_ids_sha256": "rb", "per_layer_expert_counts": [[1, 1]], "selected_ids": [[[1]], [[1]], [[0]]]}},
        ]
        self.assertTrue(validate_dynamic_inputs(rows)["valid"])
        rows[1]["routing"]["per_layer_expert_counts"] = [[2, 0]]
        self.assertFalse(validate_dynamic_inputs(rows)["valid"])

    def test_dynamic_validation_rejects_differences_from_length_alone(self):
        rows = [
            {"prompt_index": 0, "prompt_token_count": 1, "generated_token_count": 2, "output_token_ids_sha256": "a", "routing": {"selected_ids_sha256": "ra", "per_layer_expert_counts": [[2, 0]], "selected_ids": [[[0]], [[0]]]}},
            {"prompt_index": 1, "prompt_token_count": 2, "generated_token_count": 2, "output_token_ids_sha256": "b", "routing": {"selected_ids_sha256": "rb", "per_layer_expert_counts": [[3, 0]], "selected_ids": [[[0]], [[0]], [[0]]]}},
        ]
        report = validate_dynamic_inputs(rows)
        self.assertEqual(report["overlap_route_differences"], 0)
        self.assertFalse(report["valid"])

    def test_matched_comparison_detects_instrumentation_change(self):
        baseline = [{"prompt_index": 0, "prompt_token_ids_sha256": "p", "output_token_ids": [1], "output_token_ids_sha256": "o", "routing": None}]
        observed = [{"prompt_index": 0, "prompt_token_ids_sha256": "p", "output_token_ids": [1], "output_token_ids_sha256": "o", "routing": {"selected_ids_sha256": "r", "per_layer_expert_counts": [[1]]}}]
        # Control has no routing observation; token preservation remains
        # meaningful without pretending to compare absent routing data.
        self.assertTrue(compare_matched_samples(baseline, observed, compare_routing=False)["all_equal"])
        with self.assertRaises(RoutingTelemetryError):
            compare_matched_samples(baseline, observed, compare_routing=True)

    def test_hook_evidence_retains_resolved_dispatch_and_hashes(self):
        records = [{
            "event": "forward",
            "resolved_compile": "0",
            "resolved_cudagraph": "FULL_AND_PIECEWISE",
            "dispatch": {"cudagraph_mode": "FULL", "padded_num_tokens": 8},
            "requests": [{
                "req": "random-runtime-id",
                "prompt_sha256": "prompt",
                "q": 1,
                "computed": 4,
                "phase": "decode",
                "new_token_ids": [12],
                "h": "hidden-hash",
                "logits_h": "logits-hash",
            }],
        }]
        evidence = hook_evidence(records, require_hashes=True)
        self.assertEqual(evidence["dispatch_modes"], ["FULL"])
        self.assertTrue(validate_graph_dispatch(evidence, expect_graph=True, expect_compile="0")["valid"])
        self.assertFalse(validate_graph_dispatch(evidence, expect_graph=True, expect_compile="3")["valid"])
        self.assertTrue(compare_hook_hashes(evidence, evidence)["all_equal"])

    def test_hook_evidence_rejects_missing_hash_or_stale_padding_row(self):
        records = [{
            "event": "forward",
            "resolved_compile": "3",
            "resolved_cudagraph": "FULL_AND_PIECEWISE",
            "dispatch": {"cudagraph_mode": "FULL"},
            "requests": [{
                "req": "live",
                "prompt_sha256": "prompt",
                "q": 1,
                "computed": 1,
                "phase": "decode",
                "new_token_ids": [3],
                "h": None,
                "logits_h": "logits-hash",
            }],
        }]
        with self.assertRaises(RoutingTelemetryError):
            hook_evidence(records, require_hashes=True)


if __name__ == "__main__":
    unittest.main()
