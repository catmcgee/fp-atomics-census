#!/usr/bin/env python3
"""CPU-only checks for the C2 follow-up preparation."""
from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from preflight import weight_index
from validate_moe_routing import real_rows, validate_rows
from c2_constraints import LOCAL_EDITABLE, requirement_lines
from run_followup import (
    GateError,
    cache_directories_gate,
    cache_environment,
    cache_is_empty,
    cache_snapshot,
    handoff_manifest,
    prepare_cold_cache,
    parse_jobs,
    reviewed_job_file,
    run_process,
    snapshot_state_for_handoff,
    tree_manifest,
    validate_replay_summary,
    verify_ack,
    verify_runner_copy,
)


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


class FollowupRunnerPreparationTest(unittest.TestCase):
    def test_queue_parser_preserves_json_inductor_config(self):
        reviewed_job_file(HERE / "jobs_tp1.txt")
        jobs = parse_jobs(HERE / "jobs_tp1.txt")
        record = next(job for job in jobs if job.label == "DET_MIX_record")
        self.assertEqual(record.env, {"TORCHINDUCTOR_DETERMINISTIC": "1"})
        self.assertIn('{"combo_kernels":false,"benchmark_combo_kernel":false}', record.rest)

    def test_cache_snapshot_is_explicit_and_empty_when_new(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "isolated"
            root.mkdir()
            snapshot = cache_snapshot(root)
            self.assertTrue(cache_is_empty(snapshot))
            self.assertIn("TMPDIR", cache_environment(root))
            self.assertTrue(all(str(root) in row["path"] for row in snapshot["roots"]))

    def test_prepare_cold_cache_creates_every_environment_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cold"
            snapshot = prepare_cold_cache(root)
            self.assertTrue(cache_is_empty(snapshot))
            self.assertTrue(all(Path(path).is_dir() for path in cache_environment(root).values()))
            cache_directories_gate(root)
            Path(cache_environment(root)["TMPDIR"]).rmdir()
            with self.assertRaises(GateError):
                cache_directories_gate(root)

    def test_process_timeout_is_not_blocked_by_partial_line(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "partial.log"
            command = [
                sys.executable,
                "-c",
                "import subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print(child.pid,flush=True); sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(30)",
            ]
            started = time.monotonic()
            rc = run_process(command, cwd=HERE, env=os.environ.copy(), log=log, timeout=1)
            self.assertEqual(rc, 124)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("partial", log.read_text())
            self.assertIn("RUNNER_TIMEOUT", log.read_text())
            descendant = int(log.read_text().splitlines()[1])
            with self.assertRaises(ProcessLookupError):
                os.kill(descendant, 0)

    def test_process_exit_cleans_worker_left_by_command_leader(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "leader-exit.log"
            command = [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print(child.pid,flush=True)",
            ]
            started = time.monotonic()
            rc = run_process(command, cwd=HERE, env=os.environ.copy(), log=log, timeout=5)
            self.assertEqual(rc, 0)
            self.assertLess(time.monotonic() - started, 2)
            descendant = int(log.read_text().splitlines()[1])
            with self.assertRaises(ProcessLookupError):
                os.kill(descendant, 0)

    def test_runner_copy_and_executing_digest_are_both_checked(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retained = root / "runner" / "run_followup.py"
            retained.parent.mkdir()
            retained.write_text("reviewed")
            running = root / "running.py"
            running.write_text("reviewed")
            import hashlib

            digest = hashlib.sha256(b"reviewed").hexdigest()
            state = {"runner_copy": "runner/run_followup.py", "runner_sha256": digest}
            verify_runner_copy(root, state, running)
            running.write_text("changed")
            with self.assertRaises(GateError):
                verify_runner_copy(root, state, running)

    def test_replay_summary_uses_frozen_comparator_filename(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = root / "e6" / "arm"
            arm.mkdir(parents=True)
            summary = arm / "summary.json"
            summary.write_text(json.dumps({"verdict_P2": "IDENTICAL", "requirements_met": True}))
            self.assertEqual(validate_replay_summary(root, "arm", "replay"), summary)

    def test_completed_ack_must_match_retained_manifest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"sha256": "manifest-digest"}))
            ack = root / "done.json"
            ack.write_text(json.dumps({
                "schema": 1,
                "label": "done",
                "handoff_manifest_sha256": "manifest-digest",
                "verified_manifest_sha256": "manifest-digest",
            }))
            verify_ack(ack, "done", manifest)
            ack.write_text(json.dumps({"schema": 1, "label": "done"}))
            with self.assertRaises(GateError):
                verify_ack(ack, "done", manifest)

    def test_malformed_ack_is_a_gate_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"sha256": "manifest-digest"}))
            ack = root / "done.json"
            ack.write_text('{"schema": 1, "label": "done"')
            with self.assertRaisesRegex(GateError, "cannot read handoff acknowledgement"):
                verify_ack(ack, "done", manifest)

    def test_handoff_state_snapshot_does_not_follow_later_state_updates(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm, logs, archives = root / "e6" / "arm", root / "logs", root / "archives"
            arm.mkdir(parents=True)
            logs.mkdir()
            archives.mkdir()
            (arm / "result.json").write_text("{}")
            state = {"schema": 1, "completed_labels": []}
            snapshot = snapshot_state_for_handoff(root, "arm_a", state)
            manifest = handoff_manifest(root, "arm_a", arm, logs, archives, [snapshot])
            state["completed_labels"].append("arm_a")
            self.assertEqual(json.loads(snapshot.read_text()), {"schema": 1, "completed_labels": []})
            paths = {row["path"] for row in json.loads(manifest.read_text())["files"]}
            self.assertIn("handoff/arm_a/state_before_ack.json", paths)
            self.assertNotIn("runner_state.json", paths)
            with self.assertRaises(GateError):
                snapshot_state_for_handoff(root, "arm_a", state)

    def test_tree_manifest_changes_when_evidence_changes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = root / "item.txt"
            item.write_text("before")
            before = tree_manifest(root)
            item.write_text("after")
            self.assertNotEqual(before, tree_manifest(root))

    def test_constraints_exclude_only_local_hook(self):
        requirements = requirement_lines({"vllm": "0.29.0", LOCAL_EDITABLE: "0.1.0", "torch": "2.13.0"})
        self.assertEqual(requirements, ["torch==2.13.0", "vllm==0.29.0"])


if __name__ == "__main__":
    unittest.main()
