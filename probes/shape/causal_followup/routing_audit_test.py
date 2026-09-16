from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from routing_audit import (
    CONDITIONS,
    EXPECTED_HOOK_SHA256,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    EXPECTED_SHAPE,
    EXPECTED_SOURCE,
    EXPECTED_VLLM,
    EXPECTED_VLLM_FILES,
    AuditError,
    audit,
    canonical_json,
    sha256_file,
)
from routing_telemetry import (
    compare_hook_hashes,
    compare_matched_samples,
    digest_token_ids,
    hook_evidence,
    summarise_selected_experts,
    validate_dynamic_inputs,
    validate_graph_dispatch,
)

HERE = Path(__file__).resolve().parent


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def hook_records(mode: str) -> list[dict]:
    return [
        {
            "event": "forward",
            "resolved_compile": "0",
            "resolved_cudagraph": f"CUDAGraphMode.{mode}",
            "dispatch": {"cudagraph_mode": mode, "padded_num_tokens": 2},
            "requests": [
                {"req": "runtime-0", "prompt_sha256": "prompt-0", "q": 1, "computed": 3, "phase": "decode", "new_token_ids": [11], "h": "hidden-0", "logits_h": "logits-0"},
                {"req": "runtime-1", "prompt_sha256": "prompt-1", "q": 1, "computed": 4, "phase": "decode", "new_token_ids": [12], "h": "hidden-1", "logits_h": "logits-1"},
            ],
        }
    ]


def telemetry_samples() -> list[dict]:
    selected = []
    for prompt in range(2):
        rows = []
        for token in range(3):
            rows.append(
                [
                    [
                        (prompt * 11 + layer + slot + token * (prompt + 1)) % 60
                        for slot in range(4)
                    ]
                    for layer in range(24)
                ]
            )
        selected.append(rows)
    samples = []
    for index, ids in enumerate(selected):
        outputs = [index + 1, index + 2, index + 3]
        samples.append(
            {
                "prompt_index": index,
                "prompt_token_count": 1,
                "generated_token_count": 3,
                "prompt_token_ids_sha256": digest_token_ids([10 + index]),
                "output_token_ids": outputs,
                "output_token_ids_sha256": digest_token_ids(outputs),
                "routing": summarise_selected_experts(ids, num_experts=60, valid_tokens=3),
            }
        )
    return samples


def control_samples(observed: list[dict]) -> list[dict]:
    return [{**{key: value for key, value in sample.items() if key != "routing"}, "routing": None} for sample in observed]


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.report_path = root / "report.json"
        self.shape = EXPECTED_SHAPE
        self.attestation = {
            "source_commit": EXPECTED_SOURCE,
            "reviewed_vllm_commit": EXPECTED_VLLM,
            "vllm_version": "0.29.0",
            "vllm_files": EXPECTED_VLLM_FILES,
            "shape_hook_sha256": EXPECTED_HOOK_SHA256,
            "installed_modules": {"vllm": "/opt/c2/vllm/__init__.py", "shape_hook": "/root/c2-source/probes/shape/shape_hook_pkg/shape_hook/__init__.py"},
        }
        source = root / "source_snapshot"
        source.mkdir(parents=True)
        (source / "routing_harness.py").write_bytes((HERE / "routing_harness.py").read_bytes())
        (source / "routing_telemetry.py").write_bytes((HERE / "routing_telemetry.py").read_bytes())
        frozen_hook = subprocess.check_output(
            [
                "git",
                "show",
                f"{EXPECTED_SOURCE}:probes/shape/shape_hook_pkg/shape_hook/__init__.py",
            ],
            cwd=HERE.parents[2],
        )
        (source / "shape_hook_init.py").write_bytes(frozen_hook)
        assert sha256_file(source / "shape_hook_init.py") == EXPECTED_HOOK_SHA256
        self.conditions = self._conditions()
        self._write_conditions()
        self._write_report()
        for condition in CONDITIONS:
            self.refresh_handoff(condition)

    def _conditions(self) -> dict[str, dict]:
        observed = telemetry_samples()
        out = {}
        for condition in CONDITIONS:
            telemetry = condition != "graph-control"
            mode = "NONE" if condition == "eager-telemetry" else "FULL"
            records = hook_records(mode)
            out[condition] = {
                "schema": 1,
                "condition": condition,
                "status": "complete",
                "source_attestation": self.attestation,
                "engine_controls": {
                    "seed": 0,
                    "max_model_len": 2048,
                    "enable_prefix_caching": False,
                    "async_scheduling": False,
                    "enforce_eager": False,
                    "compilation_config": {"mode": 0, "cudagraph_mode": mode},
                    "enable_return_routed_experts": telemetry,
                },
                "samples": observed if telemetry else control_samples(observed),
                "hook_evidence": hook_evidence(records, require_hashes=True),
            }
        return out

    def _write_conditions(self) -> None:
        for condition, value in self.conditions.items():
            base = self.root / "report.conditions"
            write_json(base / f"{condition}.json", value)
            (base / f"{condition}.log").write_text(f"completed {condition}\n")
            records = hook_records("NONE" if condition == "eager-telemetry" else "FULL")
            hook = self.root / "hook" / condition / "rank0.jsonl"
            hook.parent.mkdir(parents=True, exist_ok=True)
            hook.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))

    def _write_report(self) -> None:
        graph_control = self.conditions["graph-control"]
        graph = self.conditions["graph-telemetry"]
        eager = self.conditions["eager-telemetry"]
        fields = {
            "instrumentation_preserves_graph_tokens": compare_matched_samples(graph_control["samples"], graph["samples"], compare_routing=False),
            "instrumentation_preserves_graph_hidden_logits": compare_hook_hashes(graph_control["hook_evidence"], graph["hook_evidence"]),
            "graph_telemetry_dispatch": validate_graph_dispatch(graph["hook_evidence"], expect_graph=True, expect_compile="0"),
            "eager_telemetry_dispatch": validate_graph_dispatch(eager["hook_evidence"], expect_graph=False, expect_compile="0"),
            "graph_matches_eager": compare_matched_samples(graph["samples"], eager["samples"], compare_routing=True),
            "dynamic_graph_inputs": validate_dynamic_inputs(graph["samples"]),
        }
        valid = all(
            [
                fields["instrumentation_preserves_graph_tokens"]["all_equal"],
                fields["instrumentation_preserves_graph_hidden_logits"]["all_equal"],
                fields["graph_telemetry_dispatch"]["valid"],
                fields["eager_telemetry_dispatch"]["valid"],
                fields["graph_matches_eager"]["all_equal"],
                fields["dynamic_graph_inputs"]["valid"],
            ]
        )
        report = {
            "schema": 1,
            "source": "vllm 0.29 enable_return_routed_experts",
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "expected_routing_shape": self.shape,
            "source_attestation": self.attestation,
            "requested_graph_mode": True,
            "conditions": {"graph_control": graph_control, "graph_telemetry": graph, "eager_telemetry": eager},
            **fields,
            "valid": valid,
        }
        write_json(self.report_path, report)
        write_json(
            self.root / "COMPLETE.json",
            {"schema": 1, "status": "valid" if valid else "invalid", "valid": valid, "output": "report.json", "acked_labels": [f"report_{condition.replace('-', '_')}" for condition in CONDITIONS]},
        )

    def refresh_handoff(self, condition: str) -> None:
        label = f"report_{condition.replace('-', '_')}"
        relatives = [
            f"report.conditions/{condition}.json",
            f"report.conditions/{condition}.log",
            f"hook/{condition}/rank0.jsonl",
            "source_snapshot/routing_harness.py",
            "source_snapshot/routing_telemetry.py",
            "source_snapshot/shape_hook_init.py",
        ]
        files = []
        for relative in sorted(relatives):
            path = self.root / relative
            files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
        payload = {"schema": 1, "label": label, "files": files}
        digest = hashlib.sha256(canonical_json(payload)).hexdigest()
        write_json(self.root / "handoff" / label / "manifest.json", {**payload, "sha256": digest})
        write_json(self.root / "acks" / f"{label}.json", {"schema": 1, "label": label, "handoff_manifest_sha256": digest, "verified_manifest_sha256": digest})


class RoutingAuditTests(unittest.TestCase):
    def test_complete_bundle_recomputes_from_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = audit(fixture.root)
            self.assertTrue(result["audit_valid"])
            self.assertTrue(result["scientific_valid"])
            self.assertEqual(len(result["verified_handoff_manifests"]), 3)

    def test_selected_id_tamper_fails_even_with_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            condition = fixture.conditions["graph-telemetry"]
            condition["samples"][0]["routing"]["selected_ids"][0][0][0] = 2
            write_json(fixture.root / "report.conditions/graph-telemetry.json", condition)
            fixture.refresh_handoff("graph-telemetry")
            with self.assertRaisesRegex(AuditError, "summary differs"):
                audit(fixture.root)

    def test_ack_digest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            ack = fixture.root / "acks/report_graph_control.json"
            value = json.loads(ack.read_text())
            value["verified_manifest_sha256"] = "0" * 64
            write_json(ack, value)
            with self.assertRaisesRegex(AuditError, "ACK mismatch"):
                audit(fixture.root)

    def test_raw_hook_tamper_fails_even_with_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            hook = fixture.root / "hook/graph-telemetry/rank0.jsonl"
            row = json.loads(hook.read_text().splitlines()[0])
            row["requests"][0]["h"] = "tampered-hidden-state"
            hook.write_text(json.dumps(row, sort_keys=True) + "\n")
            fixture.refresh_handoff("graph-telemetry")
            with self.assertRaisesRegex(AuditError, "derived hook evidence differs"):
                audit(fixture.root)


if __name__ == "__main__":
    unittest.main()
