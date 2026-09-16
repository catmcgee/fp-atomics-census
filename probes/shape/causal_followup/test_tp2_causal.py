from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tp2_collective_hook"))

from tp2_validate_trace import PINNED, validate_trace
from tp2_runner import cache_env, cache_snapshot, prepare_cold_cache, tree_manifest, verify_plugin

hook = importlib.import_module("tp2_collective_hook")


class FakeTensor:
    shape = (4, 8)
    dtype = "torch.bfloat16"
    device = "cuda:0"
    nbytes = 64

    def is_contiguous(self):
        return True

    def numel(self):
        return 32


class Backend:
    disabled = False

    def all_reduce(self, input_):
        return object()


class Communicator:
    def __init__(self):
        self.unique_name = "tp:0"
        self.world_size = 2
        self.fi_ar_comm = Backend()
        self.qr_comm = self.aiter_ar_comm = self.ca_comm = None
        self.symm_mem_comm = self.pynccl_comm = None

    def all_reduce(self, input_):
        return self.fi_ar_comm.all_reduce(input_)


class InstrumentationTest(unittest.TestCase):
    def test_dispatch_wrapper_observes_actual_nested_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = os.environ.copy()
            os.environ.update(
                {
                    "TP2_COLLECTIVE_TRACE_DIR": temporary,
                    "TP2_COLLECTIVE_RUN_ID": "run",
                    "RANK": "0",
                }
            )
            try:
                module = types.SimpleNamespace(
                    CudaCommunicator=Communicator,
                    should_nccl_symm_mem_allreduce=lambda *_: False,
                )
                hook._patch_dispatch(module)
                Communicator().all_reduce(FakeTensor())
                rows = [json.loads(line) for line in (Path(temporary) / "rank0.jsonl").read_text().splitlines()]
                self.assertEqual(rows[-1]["backend"], "FLASHINFER")
                self.assertEqual(rows[-1]["tensor"]["shape"], [4, 8])
            finally:
                os.environ.clear()
                os.environ.update(old)

    def test_workspace_wrapper_records_requested_and_actual(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = os.environ.copy()
            os.environ.update({"TP2_COLLECTIVE_TRACE_DIR": temporary, "TP2_COLLECTIVE_RUN_ID": "run", "RANK": "1"})
            try:
                module = types.SimpleNamespace(
                    _create_workspace=lambda backend, *_: types.SimpleNamespace(backend=backend)
                )
                hook._patch_workspace(module)
                module._create_workspace("trtllm", 2, 1, 32, 8, "bf16", object())
                row = json.loads((Path(temporary) / "rank1.jsonl").read_text().splitlines()[-1])
                self.assertTrue(row["created"])
                self.assertEqual(row["actual_backend"], "trtllm")
            finally:
                os.environ.clear()
                os.environ.update(old)


class ValidatorTest(unittest.TestCase):
    def fixture(self, root: Path, backend: str = "FLASHINFER") -> tuple[Path, Path]:
        trace, shapes = root / "trace", root / "hook"
        trace.mkdir()
        shapes.mkdir()
        for rank in range(2):
            events = [
                {"schema": 1, "rank": rank, "run_id": "run", "event": "registered", "expected_backend": "trtllm", "configured_backend": "trtllm", "source_sha256": PINNED},
                {"schema": 1, "rank": rank, "run_id": "run", "event": "workspace", "created": True, "requested_backend": "trtllm", "actual_backend": "trtllm"},
                {"schema": 1, "rank": rank, "run_id": "run", "event": "dispatch", "backend": backend, "group": "tp:0", "world_size": 2, "tensor": {"shape": [4, 8], "dtype": "torch.bfloat16", "nbytes": 64}},
            ]
            (trace / f"rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
            forwards = [
                {"event": "forward", "resolved_compile": "3", "resolved_cudagraph": "FULL_AND_PIECEWISE", "dispatch": {"cudagraph_mode": "PIECEWISE" if i < 2 else "FULL"}}
                for i in range(37)
            ]
            (shapes / f"rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in forwards))
        return trace, shapes

    def test_valid_two_rank_fixed_backend_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace, shapes = self.fixture(Path(temporary))
            report = validate_trace(trace, "trtllm", "run", hook_dir=shapes)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["graph_dispatch"][0]["graph_dispatch_counts"], {"FULL": 35, "PIECEWISE": 2})

    def test_non_flashinfer_tp_dispatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace, shapes = self.fixture(Path(temporary), backend="PYNCCL")
            report = validate_trace(trace, "trtllm", "run", hook_dir=shapes)
            self.assertFalse(report["valid"])
            self.assertTrue(any("other than FLASHINFER" in error for error in report["errors"]))


class RunnerHelpersTest(unittest.TestCase):
    def test_static_entry_point_resolves_to_retained_plugin(self):
        verify_plugin(HERE)

    def test_new_cache_has_all_named_empty_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            prepare_cold_cache(root)
            report = cache_snapshot(root)
            self.assertEqual({row["name"] for row in report["roots"]}, set(cache_env(root)))
            self.assertTrue(all(row["file_count"] == 0 for row in report["roots"]))

    def test_tree_manifest_changes_with_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "x"
            path.write_text("a")
            before = tree_manifest(root)
            path.write_text("b")
            self.assertNotEqual(before, tree_manifest(root))


if __name__ == "__main__":
    unittest.main()
