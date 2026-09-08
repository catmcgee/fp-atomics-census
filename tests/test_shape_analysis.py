"""Regression cases for conclusions that previously passed on incomplete data."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes/shape"))
sys.path.insert(0, str(ROOT / "probes/shape/shape_hook_pkg"))
from comparison import compare_traces
from shape_common import engine_kwargs, mixed_prompts, outputs_record, real_steps, steps_by_request
import run_e2
import run_e4
import shape_hook


def step(i=0, rid="0-a", h="abc", **kw):
    return {"step": i, "total_scheduled": 1, "num_reqs": 1, "shape_vector": "same",
            "requests": [{"req": rid, "q": 1, "kv": i + 2, "computed": i + 1, "phase": "decode", "h": h, "argmax": 1}], **kw}


def test_zero_calls_are_not_forward_passes():
    zero = step(total_scheduled=0)  # stale legacy rows must be ignored
    assert real_steps([zero, step()]) == [step()]
    assert len(steps_by_request([zero, step()])["0"]) == 1


@pytest.mark.parametrize("mutation", ["tail", "null", "request_set", "duplicate", "step_alignment", "count"])
def test_comparison_rejects_incomplete_or_misjoined_traces(mutation):
    a = [step(), step(1)]
    b = copy.deepcopy(a)
    if mutation == "tail": b.pop()
    if mutation == "null": b[0]["requests"][0]["h"] = None
    if mutation == "request_set": b[0]["requests"][0]["req"] = "3-other"
    if mutation == "duplicate": b[1]["step"] = 0
    if mutation == "step_alignment": b[1]["step"] = 4
    if mutation == "count": b[0]["total_scheduled"] = 7
    result = compare_traces(a, b)
    assert result["verdict"] == "INVALID"
    assert result["validation_errors"]


def test_shape_mismatch_is_not_a_kernel_difference():
    a, b = step(), step(shape_vector="other")
    assert compare_traces([a], [b])["verdict"] == "NOT COMPARABLE"


def test_request_suffixes_normalise_but_order_is_retained():
    assert compare_traces([step(rid="0-a")], [step(rid="0-b")])["verdict"] == "IDENTICAL"
    a = step(total_scheduled=2, num_reqs=2)
    second = {**a["requests"][0], "req": "1-a", "kv": 10, "computed": 9}
    a["requests"].append(second)
    b = {**a, "requests": list(reversed(a["requests"]))}
    assert compare_traces([a], [b])["verdict"] == "NOT COMPARABLE"
    assert steps_by_request([a])["0"][0][1] != steps_by_request([b])["0"][0][1]


def fake_output(probs):
    return NS(request_id="0", outputs=[NS(token_ids=[1], logprobs=[{k: NS(logprob=v) for k, v in probs.items()}])])


def test_logprob_hash_commits_vocabulary_and_signed_zero():
    a = outputs_record([fake_output({1: -0.1, 2: -1.0, 3: -2.0})])["0"]
    b = outputs_record([fake_output({1: -0.1, 2: -2.0, 3: -1.0})])["0"]
    assert len(a["hash"]) == 64
    assert a["hash"] != b["hash"]
    assert outputs_record([fake_output({1: 0.0})]) != outputs_record([fake_output({1: -0.0})])


def test_mixed_prompts_include_long_inputs_and_exact_requested_length():
    prompts = mixed_prompts()
    assert len(prompts) == 16
    assert max(map(len, prompts)) > 500
    assert len(mixed_prompts(41)) == 41


@pytest.mark.parametrize("graphs,compilation,expected", [(0, True, (3, "NONE")), (1, True, (3, "FULL_AND_PIECEWISE")), (0, False, (0, "NONE")), (1, False, (0, "FULL"))])
def test_graph_compile_controls_are_independent(monkeypatch, graphs, compilation, expected):
    # Environment is used by worker assertions; restore it between tests.
    monkeypatch.setenv("SHAPE_EXPECT_COMPILE", "")
    monkeypatch.setenv("SHAPE_EXPECT_GRAPHS", "")
    args = NS(model="model", tp=1, cudagraph=graphs, prefix_caching=0, no_compile=not compilation)
    kw = engine_kwargs(args)
    assert kw["enforce_eager"] is False
    assert (kw["compilation_config"]["mode"], kw["compilation_config"]["cudagraph_mode"]) == expected


@pytest.mark.parametrize("bad", ["null", "missing_join", "missing_output", "singleton_only"])
def test_e2_never_confirms_missing_evidence(tmp_path, bad):
    (tmp_path / "hook").mkdir()
    records = [{"repeat": i, "request_ids": [f"{i}-a"], "outputs": {f"{i}-a": {"hash": "same", "tokens": [1]}}} for i in range(2)]
    traces = [step(i, f"{i}-a") for i in range(2)]
    if bad == "null": traces[0]["requests"][0]["h"] = None
    if bad == "missing_join": traces.pop()
    if bad == "missing_output": records[0]["outputs"] = {}
    if bad == "singleton_only": traces[0]["shape_vector"] = "different-history"
    (tmp_path / "outputs.json").write_text(json.dumps(records))
    (tmp_path / "hook/rank0.jsonl").write_text("\n".join(map(json.dumps, traces)))
    run_e2.analyse(tmp_path, "synthetic", 2)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verdict_P1"] == ("NOT EVALUATED" if bad == "singleton_only" else "INVALID")


def test_empty_execute_flushes_prior_hash_before_clearing_batch(tmp_path, monkeypatch):
    path = tmp_path / "rank0.jsonl"
    monkeypatch.setattr(shape_hook, "_out_path", lambda: str(path))
    monkeypatch.setattr(shape_hook, "_state", {"pending": step(h=None), "hashes": {"hidden_rows": ["saved"], "argmax": [1]}, "batch": {"stale": True}, "step": 1})
    shape_hook._execute(NS(), lambda *a: "done", NS(total_num_scheduled_tokens=0))
    records = list(map(json.loads, path.read_text().splitlines()))
    assert records[0]["requests"][0]["h"] == "saved"
    assert records[1]["event"] == "empty_scheduler_call"
    assert records[1]["requests"] == []
    assert shape_hook._state["batch"] is None


def test_e4_does_not_truncate_or_mutate_raw_runs(tmp_path):
    (tmp_path / "hook").mkdir()
    runs = [{"repeat": i // 2, "kind": "original" if i % 2 == 0 else "dummy", "target_rid": f"{i}-a",
             "request_ids": [f"{i}-a"], "target": {"tokens": [1, 1], "logprobs": [], "hash": "same"}} for i in range(4)]
    traces = [step(2*i+j, f"{i}-a") for i in range(4) for j in range(2) if (i,j)!=(3,1)]
    raw = json.dumps(runs)
    (tmp_path / "runs.json").write_text(raw)
    (tmp_path / "hook/rank0.jsonl").write_text("\n".join(map(json.dumps, traces)))
    assert run_e4.analyse(tmp_path, "synthetic", 2, 0) == 1
    assert (tmp_path / "runs.json").read_text() == raw
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verdict_P3"] == "INVALID"
    assert summary["runs_missing_final_record"] == 1


def test_two_equally_truncated_traces_fail_terminal_validation():
    from comparison import output_trace_errors
    outputs = {"0": {"hash": "recorded", "tokens": [1, 1, 1]}}
    assert output_trace_errors([step(), step(1)], outputs)
    a = step()
    a["requests"][0]["prompt"] = 2
    assert not output_trace_errors([a], {"0": {"hash": "recorded", "tokens": [1]}})
    a["requests"][0]["kv"] = 3
    assert output_trace_errors([a], {"0": {"hash": "recorded", "tokens": [1]}})


def test_analysis_digests_resolve_symlinks(tmp_path):
    from shape_common import analysis_metadata
    real = tmp_path / "real"
    real.mkdir()
    (real / "raw.json").write_text("{}")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert analysis_metadata(real, [alias / "raw.json"])["input_sha256"].keys() == {"raw.json"}
