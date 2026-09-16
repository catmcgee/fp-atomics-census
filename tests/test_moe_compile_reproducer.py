"""CPU-only checks for the issue #56900 diagnostic controller."""

import importlib.util
import json
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
