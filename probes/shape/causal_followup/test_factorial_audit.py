from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "factorial_audit", Path(__file__).with_name("factorial_audit.py")
)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit_module.canonical_json(value))


def make_manifest(results: Path, ack_dir: Path, label: str, files: list[Path]) -> None:
    rows = [
        {
            "path": file.relative_to(results).as_posix(),
            "size": file.stat().st_size,
            "sha256": audit_module.sha256_file(file),
        }
        for file in files
    ]
    payload = {"schema": 1, "label": label, "files": rows}
    manifest = {**payload, "sha256": hashlib.sha256(audit_module.canonical_json(payload)).hexdigest()}
    write_json(results / "handoff" / label / "manifest.json", manifest)
    write_json(
        ack_dir / f"{label}.json",
        {
            "schema": 1,
            "label": label,
            "handoff_manifest_sha256": manifest["sha256"],
            "verified_manifest_sha256": manifest["sha256"],
        },
    )


def make_results(tmp_path: Path) -> tuple[Path, Path]:
    results, acks = tmp_path / "raw", tmp_path / "acks"
    cell = audit_module.cells()[0]
    write_json(results / "plan.json", {"schema": 1, "cells": audit_module.cells(), "cold_repeats": 3})
    arm = results / "e6" / f"Qwen_factorial_{cell['label']}_mixed"
    record = arm / "record"
    replay = arm / "replay_warm"
    common = {
        "status": "configured",
        "resolved_compile": "3",
        "resolved_cudagraph": "FULL_AND_PIECEWISE",
        "resolved_config": "{'combo_kernels': False, 'benchmark_combo_kernel': False}",
        "args": {"mixed": True, "prefix_caching": 0},
    }
    write_json(record / "run.json", common)
    write_json(record / "env.json", {"env": {"TORCHINDUCTOR_DETERMINISTIC": "0"}})
    write_json(record / "artefacts.json", {"best_configs": {}, "triton_kernel_families": {}, "extern_calls": {}})
    replay_run = {**common, "recorded_run_id": "record-id"}
    write_json(replay / "run.json", replay_run)
    write_json(replay / "env.json", {"env": {"TORCHINDUCTOR_DETERMINISTIC": "0"}})
    write_json(replay / "artefacts.json", {"best_configs": {}, "triton_kernel_families": {}, "extern_calls": {}})
    write_json(
        arm / "summary_replay_warm.json",
        {
            "requirements_met": True,
            "verdict_P2": "DIFFERS",
            "passes_replayed": 2,
            "rows_compared": 3,
            "forcing_log": {"calls": 2, "forced_rows": 3},
        },
    )
    make_manifest(results, acks, "d0_c0_b0_record", [record / "run.json", record / "env.json", record / "artefacts.json", results / "plan.json"])
    make_manifest(
        results,
        acks,
        "d0_c0_b0_warm",
        [replay / "run.json", replay / "env.json", replay / "artefacts.json", arm / "summary_replay_warm.json", results / "plan.json"],
    )
    return results, acks


class FactorialAuditTest(unittest.TestCase):
    def test_audit_admits_only_acknowledged_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, acks = make_results(Path(temporary))
            report = audit_module.audit(results, acks)
        cell = next(row for row in report["cells"] if row["label"] == "d0_c0_b0")
        self.assertEqual(cell["phases"]["record"]["state"], "verified_valid")
        self.assertEqual(cell["phases"]["warm"]["state"], "verified_valid")
        self.assertEqual(cell["phases"]["cold_0"]["state"], "pending_or_unverified")
        self.assertEqual(report["status"], "partial")
        self.assertEqual(cell["phases"]["warm"]["verdict_P2"], "DIFFERS")

    def test_audit_rejects_ack_with_wrong_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results, acks = make_results(Path(temporary))
            bad = acks / "d0_c0_b0_record.json"
            value = json.loads(bad.read_text())
            value["verified_manifest_sha256"] = "not-the-manifest"
            write_json(bad, value)
            report = audit_module.audit(results, acks)
        cell = next(row for row in report["cells"] if row["label"] == "d0_c0_b0")
        self.assertEqual(cell["phases"]["record"]["state"], "pending_or_unverified")
        self.assertIn("acknowledgement", cell["phases"]["record"]["reason"])

    def test_source_ir_filter_excludes_cache_members(self) -> None:
        self.assertTrue(audit_module.public_member("inductor/a.py"))
        self.assertTrue(audit_module.public_member("inductor/graph.ttir"))
        self.assertFalse(audit_module.public_member("cache/x.py"))
        self.assertFalse(audit_module.public_member("inductor/kernel.so"))

    def test_public_stage_uses_byte_exact_frozen_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, acks = make_results(root)
            report = audit_module.audit(results, acks)
            staging = root / "public"
            audit_module.stage_public(results, acks, report, staging, Path(__file__).resolve().parents[3])
            staged = staging / "frozen_source" / "probes" / "shape" / "run_e6.py"
            expected = subprocess.check_output(
                ["git", "show", f"{audit_module.FROZEN_SOURCE}:probes/shape/run_e6.py"],
                cwd=Path(__file__).resolve().parents[3],
            )
            self.assertEqual(staged.read_bytes(), expected)
            self.assertTrue((staging / "audit.json").is_file())
            self.assertTrue((staging / "SHA256SUMS.txt").is_file())
            self.assertTrue((staging / "handoff" / "d0_c0_b0_record" / "verified-ack.json").is_file())
            self.assertTrue((staging / "raw" / "e6" / "Qwen_factorial_d0_c0_b0_mixed" / "record" / "run.json.gz").is_file())
            self.assertTrue((staging / "private_caches.json").is_file())
            self.assertFalse((staging / "raw" / "archives").exists())

    def test_public_sums_reject_tamper_before_code_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, acks = make_results(root)
            report = audit_module.audit(results, acks)
            staging = root / "public"
            audit_module.stage_public(results, acks, report, staging, Path(__file__).resolve().parents[3])
            self.assertGreater(audit_module.verify_public_sums(staging), 0)
            audit_json = staging / "audit.json"
            audit_json.write_bytes(audit_json.read_bytes() + b" ")
            with self.assertRaisesRegex(audit_module.AuditError, "digest mismatch"):
                audit_module.audit_public(staging)

    def test_public_sums_reject_unlisted_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, acks = make_results(root)
            report = audit_module.audit(results, acks)
            staging = root / "public"
            audit_module.stage_public(results, acks, report, staging, Path(__file__).resolve().parents[3])
            (staging / "unlisted.txt").write_text("extra\n")
            with self.assertRaisesRegex(audit_module.AuditError, "coverage mismatch"):
                audit_module.audit_public(staging)

    def test_public_comparator_digest_must_match_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, acks = make_results(root)
            report = audit_module.audit(results, acks)
            staging = root / "public"
            audit_module.stage_public(results, acks, report, staging, Path(__file__).resolve().parents[3])
            audit_json = staging / "audit.json"
            value = json.loads(audit_json.read_text())
            value["frozen_comparator_sha256"] = "0" * 64
            write_json(audit_json, value)
            sums = staging / "SHA256SUMS.txt"
            sums.write_text(
                "".join(
                    f"{audit_module.sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                    for path in sorted(staging.rglob("*"))
                    if path.is_file() and path != sums
                )
            )
            with self.assertRaisesRegex(audit_module.AuditError, "comparator differs"):
                audit_module.audit_public(staging)


if __name__ == "__main__":
    unittest.main()
