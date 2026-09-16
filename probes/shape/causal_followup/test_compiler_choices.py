#!/usr/bin/env python3
"""CPU regressions for per-kernel compiler-choice association reports."""
from __future__ import annotations

import gzip
import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from compiler_choices import analyse_public, changed_kernels, normalise_choice, row_classes, source_kernel_name


class CompilerChoiceTests(unittest.TestCase):
    def test_normalise_omits_non_choice_fields(self) -> None:
        self.assertEqual(
            normalise_choice({"R0_BLOCK": 4096, "time_taken_ms": 3.2, "triton_cache_hash": "private"}),
            {"R0_BLOCK": 4096},
        )

    def test_changed_kernels_is_per_cache_key_with_source_name(self) -> None:
        record = {"best_configs": {"inductor_env/o2/key.best_config": {"R0_BLOCK": 2048, "time_taken_ms": 2.0}}}
        replay = {"best_configs": {"inductor_env/o2/key.best_config": {"R0_BLOCK": 4096, "time_taken_ms": 4.0}}}
        changes = changed_kernels(record, replay, {"o2": {"source_path": "inductor/o2/kernel.py", "kernel_name": "triton_red_x"}})
        self.assertEqual(changes, [{
            "best_config_path": "inductor_env/o2/key.best_config",
            "cache_key_prefix": "o2",
            "record_choice": {"R0_BLOCK": 2048},
            "replay_choice": {"R0_BLOCK": 4096},
            "changed_fields": ["R0_BLOCK"],
            "source_path": "inductor/o2/kernel.py",
            "kernel_name": "triton_red_x",
        }])

    def test_row_classes_are_digest_classes_not_aggregate_counts(self) -> None:
        recorded = [{"event": "forward", "requests": [{"req": "a", "h": "1", "logits_h": "2", "argmax": 3}]}]
        replayed = [{"event": "forward", "requests": [{"req": "a", "h": "x", "logits_h": "2", "argmax": 4}]}]
        result = row_classes(recorded, replayed)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["classes"], {"h+argmax": 1})
        self.assertEqual(result["record_rows"], 1)

    def test_source_kernel_name_prefers_inductor_metadata(self) -> None:
        source = "inductor_meta={'kernel_name': 'triton_red_real'}\ndef triton_red_other(x): pass\n"
        self.assertEqual(source_kernel_name(source), "triton_red_real")

    def test_public_gzip_bundle_uses_sources_ir_without_private_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            arm = "synthetic_factorial_d0_c1_b1_mixed"
            base = bundle / "raw" / "e6" / arm

            def write_json(relative: Path, value: object) -> None:
                path = base / relative.with_name(relative.name + ".gz")
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt") as handle:
                    json.dump(value, handle)

            choice = "inductor_env/aa/k.best_config"
            write_json(Path("record/artefacts.json"), {"best_configs": {choice: {"R0_BLOCK": 1}}})
            (base / "record" / "hook").mkdir(parents=True)
            with gzip.open(base / "record/hook/rank0.jsonl.gz", "wt") as handle:
                handle.write('{"requests":[{"req":"a","h":"h","logits_h":"l","argmax":1}]}\n')
            for phase in ("warm", "cold_0", "cold_1", "cold_2"):
                value = 2 if phase == "cold_0" else 1
                write_json(Path(f"replay_{phase}/artefacts.json"), {"best_configs": {choice: {"R0_BLOCK": value}}})
                write_json(Path(f"summary_replay_{phase}.json"), {"verdict_P2": "DIFFERS" if value == 2 else "IDENTICAL", "rows_compared": 1})
                path = base / f"replay_{phase}/hook/rank0.jsonl.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt") as handle:
                    handle.write('{"requests":[{"req":"a","h":"h","logits_h":"l","argmax":1}]}\n')
            source_dir = bundle / "sources_ir"
            source_dir.mkdir()
            with tarfile.open(source_dir / "d0_c1_b1_cold_0.sources.tar.gz", "w:gz") as archive:
                content = b"inductor_meta={'kernel_name': 'triton_red_synthetic'}\n"
                info = tarfile.TarInfo("synthetic/inductor/aa/kernel.py")
                info.size = len(content)
                archive.addfile(info, BytesIO(content))

            report = analyse_public(bundle, "d0_c1_b1")
            changed = report["comparisons"][1]["changed_kernels"]
            self.assertEqual(changed[0]["kernel_name"], "triton_red_synthetic")
            self.assertEqual(changed[0]["changed_fields"], ["R0_BLOCK"])


if __name__ == "__main__":
    unittest.main()
