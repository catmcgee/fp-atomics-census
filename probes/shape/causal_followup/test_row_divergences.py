#!/usr/bin/env python3
from __future__ import annotations

import unittest

from row_divergences import compare, shape


class RowDivergencesTests(unittest.TestCase):
    def test_compare_records_pass_shape_and_exact_digest_fields(self) -> None:
        record = [{"step": 7, "shape_vector": '{"total":128}', "dispatch": {"padded_num_tokens": 128}, "requests": [{"req": "0-a", "phase": "decode", "computed": 2, "kv": 3, "q": 1, "h": "a", "logits_h": "b", "argmax": 4}]}]
        replay = [{"step": 7, "shape_vector": '{"total":128}', "dispatch": {"padded_num_tokens": 128}, "requests": [{"req": "0-b", "phase": "decode", "computed": 2, "kv": 3, "q": 1, "h": "x", "logits_h": "b", "argmax": 4}]}]
        result = compare(record, replay)
        self.assertEqual(result["row_digest_classes"], {"h": 1})
        self.assertEqual(result["divergent_rows"][0]["slot"], "0")
        self.assertEqual(result["divergent_rows"][0]["pass_shape"], {"total": 128})
        self.assertEqual(result["divergent_rows"][0]["record"]["argmax"], 4)

    def test_shape_retains_non_json_text(self) -> None:
        self.assertEqual(shape({"shape_vector": "not-json"}), "not-json")


if __name__ == "__main__":
    unittest.main()
