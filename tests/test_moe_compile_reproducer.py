"""CPU-only checks for the issue #56900 diagnostic controller."""

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "probes/diagnostics/moe_compile/moe_compile_reproducer.py"
SPEC = importlib.util.spec_from_file_location("moe_compile_reproducer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MoeCompileReproducerTests(unittest.TestCase):
    def test_requested_graph_modes_match_the_posted_script(self) -> None:
        self.assertEqual(MODULE.requested_graph_mode("on", "on"), "FULL_AND_PIECEWISE")
        self.assertEqual(MODULE.requested_graph_mode("off", "on"), "FULL")
        self.assertEqual(MODULE.requested_graph_mode("on", "off"), "NONE")
        self.assertEqual(MODULE.requested_graph_mode("off", "off"), "NONE")

    def test_explicit_mode_overrides_legacy_compile_spelling(self) -> None:
        self.assertEqual(MODULE.requested_compile_mode(Namespace(mode=1, compile="off")), 1)
        self.assertEqual(MODULE.requested_compile_mode(Namespace(mode=2, compile="on")), 2)
        self.assertEqual(MODULE.requested_compile_mode(Namespace(mode=None, compile="off")), 0)
        self.assertEqual(MODULE.requested_compile_mode(Namespace(mode=None, compile="on")), 3)

    def test_vllm_version_check_allows_only_an_unspecified_wheel_suffix(self) -> None:
        self.assertTrue(MODULE.vllm_version_matches("0.28.0+cu129", "0.28.0"))
        self.assertTrue(MODULE.vllm_version_matches("0.28.0+cu129", "0.28.0+cu129"))
        self.assertFalse(MODULE.vllm_version_matches("0.28.0+cu130", "0.28.0+cu129"))
        self.assertFalse(MODULE.vllm_version_matches("0.29.0", "0.28.0"))

    def test_failed_cell_is_not_used_for_a_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = []
            for compile_mode, returncode, failure in (
                ("on", 1, True),
                ("off", 0, False),
            ):
                name = f"compile-{compile_mode}_graphs-off_runner-v2"
                run = root / name
                run.mkdir()
                (run / "result.json").write_text(json.dumps({"repeats": [[]]}))
                if failure:
                    (run / "failure.json").write_text("{}")
                completed.append(
                    {
                        "name": name,
                        "compile": compile_mode,
                        "graphs": "off",
                        "runner": "v2",
                        "returncode": returncode,
                    }
                )
            comparison = MODULE.compare_runs(root, completed)
            self.assertEqual(comparison["compiled_vs_eager"], [])

    def test_backend_lines_drop_log_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "worker.log"
            log.write_text(
                "INFO 01:02:03 [gpu_worker.py:396] Using V2 Model Runner\n"
                "INFO 01:02:04 [flash_attn.py:866] Using FlashAttention version 3\n"
            )
            self.assertEqual(
                MODULE.backend_lines(log),
                ["Using V2 Model Runner", "Using FlashAttention version 3"],
            )

    def test_timeout_has_distinct_nonzero_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            returncode, timed_out = MODULE.run_worker(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                root / "stdout.log",
                root / "stderr.log",
                timeout_seconds=1,
            )
            self.assertEqual(returncode, 124)
            self.assertTrue(timed_out)

    def test_sigterm_handler_turns_supervisor_stop_into_controller_cleanup(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            MODULE.controller_sigterm_handler(15, None)
        self.assertEqual(raised.exception.code, 143)

    def test_mode_matrix_retains_successes_and_startup_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = []
            for mode, returncode in ((0, 0), (1, 1)):
                name = f"mode-{mode}_graphs-off_runner-v2"
                run = root / name
                run.mkdir()
                if returncode:
                    (run / "failure.json").write_text(
                        json.dumps({"type": "RuntimeError", "message": "unsupported"})
                    )
                else:
                    tokens = [[mode, index] for index in range(16)]
                    (run / "result.json").write_text(
                        json.dumps(
                            {
                                "metrics": {"duplicate_pairs_agree": 8, "short_cycles": 0},
                                "repeats": [
                                    [
                                        {"token_ids": token_ids}
                                        for token_ids in tokens
                                    ]
                                ],
                            }
                        )
                    )
                completed.append(
                    {
                        "name": name,
                        "compile": "off" if mode == 0 else "on",
                        "mode": mode,
                        "graphs": "off",
                        "runner": "v2",
                        "returncode": returncode,
                        "timed_out": False,
                    }
                )
            comparison = MODULE.compare_runs(root, completed)
            self.assertEqual(comparison["mode_matrix"]["0"]["metrics"]["duplicate_pairs_agree"], 8)
            self.assertEqual(comparison["mode_matrix"]["1"]["failure"]["message"], "unsupported")

    def test_handoff_requires_exact_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = "mode-0_graphs-off_runner-v2"
            (root / cell).mkdir(parents=True)
            (root / cell / "result.json").write_text("{}")
            (root / f"{cell}.stdout.log").write_text("")
            (root / f"{cell}.stderr.log").write_text("")
            (root / "manifest.json").write_text("{}")
            (root / "comparison.json").write_text("{}")
            manifest_path = MODULE.handoff_manifest(root, cell)
            verified = MODULE.verify_handoff_manifest(root, manifest_path)
            (root / "acks").mkdir()
            MODULE.write_json(
                root / "acks" / f"{verified['label']}.json",
                {
                    "schema": 1,
                    "label": verified["label"],
                    "handoff_manifest_sha256": verified["sha256"],
                    "verified_manifest_sha256": verified["sha256"],
                },
            )
            self.assertEqual(MODULE.audit_handoffs(root)["handoffs_verified"], 1)


if __name__ == "__main__":
    unittest.main()
