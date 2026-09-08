"""Actual CPU tensor tests for bit comparisons, diagnostics and report integrity."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import common


def test_tensor_count_dtype_shape_and_signed_zero_are_part_of_equality():
    x = torch.tensor([1.0], dtype=torch.float32)
    assert not common.bitwise_equal([x], [x, x])[0]
    assert not common.bitwise_equal([x], [x.view(torch.int32)])[0]
    assert common.tensor_hash([x]) != common.tensor_hash([x.view(torch.int32)])
    assert not common.bitwise_equal([x], [x.reshape(1, 1)])[0]
    assert not common.bitwise_equal([torch.tensor([0.0])], [torch.tensor([-0.0])])[0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ulp_measurement_includes_adjacent_positive_and_negative_values(dtype):
    a = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
    b = torch.nextafter(a, torch.full_like(a, float("inf")))
    diagnostics = common.ulp_diagnostics(a, b)
    assert diagnostics["max_ulp"] == 1
    assert diagnostics["ulp_counts"]["1"] == 3


def test_reports_record_rejections_nonfinite_outputs_and_actual_evaluation_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    assert common.run_twice("identity", lambda: [torch.tensor([1.0])], repeats=2)
    identity = json.loads(next(tmp_path.rglob("identity.json")).read_text())
    assert identity["evaluations"] == 3
    assert len(identity["runs"]) == 2
    assert len(identity["first_hash"]) == 64
    assert common.run_twice("bad", lambda: [torch.tensor([float("nan")])]) is False
    assert json.loads(next(tmp_path.rglob("bad.json")).read_text())["verdict"] == "INVALID"
    original = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    def rejected():
        raise RuntimeError("does not have a deterministic implementation")
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        assert common.run_twice("reject", rejected) is False
    finally:
        torch.use_deterministic_algorithms(original, warn_only=warn)
    assert json.loads(next(tmp_path.rglob("reject.json")).read_text())["verdict"] == "REJECTED"
    with pytest.raises(FileExistsError):
        common.run_twice("identity", lambda: [torch.tensor([1.0])])


def test_hook_hashes_masked_logits_and_rejects_nan(monkeypatch):
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes/shape/shape_hook_pkg"))
    import shape_hook
    monkeypatch.setattr(shape_hook, "_state", {})
    logits = torch.tensor([[1.0, float("-inf")]])
    model = SimpleNamespace(compute_logits=lambda hidden: logits)
    shape_hook._wrap_compute_logits(model)
    model.compute_logits(torch.tensor([[1.0, 2.0]]))
    assert len(shape_hook._state["hashes"]["logit_rows"][0]) == 64
    logits[0, 0] = float("nan")
    model.compute_logits(torch.tensor([[1.0, 2.0]]))
    assert "error" in shape_hook._state["hashes"]


def test_hidden_hash_includes_dtype_and_shape():
    import shape_hook
    x = torch.tensor([1.0], dtype=torch.float32)
    assert shape_hook._row_hash(x) != shape_hook._row_hash(x.view(torch.int32))
    assert shape_hook._row_hash(x) != shape_hook._row_hash(x.reshape(1, 1))
