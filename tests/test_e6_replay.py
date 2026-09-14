"""E6 reconstruction: schema 3 records, replay planning, fail-closed comparison and teacher forcing (CPU only)."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes/shape"))
sys.path.insert(0, str(ROOT / "probes/shape/shape_hook_pkg"))
import reanalyse
import run_e6
import shape_hook
import tables
import teacher_forcing as tf
from shape_common import prompt_sha256, real_steps, trace_errors

PROMPTS = {"0": [10, 11, 12], "1": [20, 21], "2": [30, 31]}
OUTPUTS = {"0": [100, 101], "1": [200, 201], "2": [300, 301]}


def h(tag: str) -> str:
    return (tag * 64)[:64]


def row(slot, q, computed, prompt, new, argmax, suffix="a", cache_hit=None, phase=None, tag=None):
    tag = tag or f"{slot}{computed}"
    return {"req": f"{slot}-{suffix}", "q": q, "computed": computed, "kv": computed + q, "prompt": prompt,
            "phase": phase or ("prefill" if computed < prompt else "decode"), "cache_hit": cache_hit,
            "h": h(tag), "argmax": argmax, "logits_h": h("L" + tag), "new_token_ids": list(new), "prompt_sha256": prompt_sha256(PROMPTS[slot])}


def admitted(slot, suffix="a"):
    return {"req": f"{slot}-{suffix}", "prompt_token_ids": PROMPTS[slot], "prompt_len": len(PROMPTS[slot]), "num_computed_tokens": 0,
            "prompt_sha256": prompt_sha256(PROMPTS[slot])}


def pass_record(i, rows, admitted_rows=(), finished=(), suffix="a", **kw):
    rec = {"schema": 3, "event": "forward", "step": 5 + i, "rank": 0, "total_scheduled": sum(r["q"] for r in rows), "num_reqs": len(rows),
           "dispatch": {"cudagraph_mode": "FULL", "padded_num_tokens": 8, "padded_num_reqs": 4, "uniform_decode": False},
           "attention_backend": "FLASH_ATTN", "parallel": {"tp": 1, "pp": 1, "dp": 1, "ep": False},
           "resolved_compile": "3", "resolved_cudagraph": "FULL_AND_PIECEWISE", "model_runner": "v1",
           "scheduler": {"max_num_seqs": 16, "max_num_batched_tokens": 2048, "enable_chunked_prefill": True, "async_scheduling": False},
           "admitted": [admitted(s, suffix) for s in admitted_rows], "resumed": [], "preempted": [], "finished": [f"{s}-{suffix}" for s in finished],
           "requests": rows, "shape_vector": f"sv{i}"}
    rec.update(kw)
    return rec


def make_record(suffix="a"):
    """Three requests, max_tokens 2: slots 0 and 1 admitted at pass 0, slot 2 at pass 1."""
    return [
        pass_record(0, [row("0", 3, 0, 3, PROMPTS["0"], 100, suffix, cache_hit=0), row("1", 2, 0, 2, PROMPTS["1"], 200, suffix, cache_hit=0)], admitted_rows=("0", "1"), suffix=suffix),
        pass_record(1, [row("0", 1, 3, 3, [100], 101, suffix), row("1", 1, 2, 2, [200], 201, suffix), row("2", 2, 0, 2, PROMPTS["2"], 300, suffix, cache_hit=0)], admitted_rows=("2",), suffix=suffix),
        pass_record(2, [row("2", 1, 2, 2, [300], 301, suffix)], finished=("0", "1"), suffix=suffix),
    ]


def outputs(suffix="a"):
    return {f"{s}-{suffix}": {"hash": h("o" + s), "tokens": toks} for s, toks in OUTPUTS.items()}


RUN_META = {"run_id": "rec-1", "status": "configured", "resolved_model_runner": "v1",
            "resolved_scheduler": {"max_num_seqs": 16, "max_num_batched_tokens": 2048, "enable_chunked_prefill": True, "async_scheduling": False},
            "args": {"max_tokens": 2, "prefix_caching": 0, "mixed": False}}


def test_record_is_valid_and_replay_plan_is_complete():
    rec = make_record()
    assert run_e6.record_errors(rec) == []
    plan = run_e6.replay_plan(rec, RUN_META, run_e6.normalise_outputs(outputs()))
    assert plan["errors"] == []
    assert [p["admitted"] for p in plan["passes"]] == [["0", "1"], ["2"], []]
    assert plan["passes"][2]["finished"] == ["0", "1"]
    assert {s: v["forced"] for s, v in plan["requests"].items()} == OUTPUTS
    assert plan["requests"]["2"]["prompt_token_ids"] == [30, 31]


def test_identical_replay_reports_identical_with_no_free_running_divergence():
    result = run_e6.compare_replay(make_record("a"), make_record("b"))
    assert result["verdict_P2"] == "IDENTICAL"
    assert result["rows_compared"] == 6 and result["rows_differing"] == 0
    assert result["free_running_divergence"] == {"rows": 0, "passes": 0, "first": None}


def test_differs_names_first_pass_and_slot_and_counts_free_running_divergence():
    rep = make_record("b")
    rep[1]["requests"][1]["h"] = h("x")  # slot 1 hidden state at pass 1
    rep[2]["requests"][0]["argmax"] = 999  # free-running argmax at pass 2 differs from the forced token
    result = run_e6.compare_replay(make_record("a"), rep)
    assert result["verdict_P2"] == "DIFFERS"
    assert result["first_divergence"] == {"pass": 1, "step_recorded": 6, "slot": "1", "fields": ["h"]}
    assert result["rows_differing"] == 2 and result["passes_differing"] == 2
    assert result["free_running_divergence"] == {"rows": 1, "passes": 1, "first": {"pass": 2, "slot": "2"}}


@pytest.mark.parametrize("mutation,expected", [
    ("trace_length", {"pass": 2, "slot": None, "field": "trace_length"}),
    ("request_set", {"pass": 2, "slot": None, "field": "request_order"}),
    ("request_order", {"pass": 1, "slot": None, "field": "request_order"}),
    ("shape_row_prompt", {"pass": 0, "slot": "0", "field": "prompt"}),
    ("shape_row_cache_hit", {"pass": 1, "slot": "2", "field": "cache_hit"}),
    ("shape_dispatch", {"pass": 0, "slot": None, "field": "dispatch"}),
    ("attention_backend", {"pass": 1, "slot": None, "field": "attention_backend"}),
    ("finished", {"pass": 2, "slot": None, "field": "finished"}),
    ("model_runner", {"pass": 0, "slot": None, "field": "model_runner"}),
    ("scheduler", {"pass": 1, "slot": None, "field": "scheduler"}),
    ("preempted", {"pass": 0, "slot": None, "field": "preempted"}),
    ("new_token_ids", {"pass": 1, "slot": "0", "field": "new_token_ids"}),
])
def test_requirement_violations_are_not_comparable(mutation, expected):
    rep = make_record("b")
    if mutation == "trace_length":
        rep.pop()
    if mutation == "request_set":
        rep[2]["requests"][0]["req"] = "3-b"
        rep[2]["requests"][0]["prompt_sha256"] = None
    if mutation == "request_order":
        rep[1]["requests"].reverse()
    if mutation == "shape_row_prompt":
        rep[0]["requests"][0]["prompt"] = 4
    if mutation == "shape_row_cache_hit":
        rep[1]["requests"][2]["cache_hit"] = 1
    if mutation == "shape_dispatch":
        rep[0]["dispatch"]["padded_num_tokens"] = 16
    if mutation == "attention_backend":
        rep[1]["attention_backend"] = "FLASHINFER"
    if mutation == "finished":
        rep[2]["finished"] = ["0-b"]
    if mutation == "model_runner":
        rep[0]["model_runner"] = "v2"
    if mutation == "scheduler":
        rep[1]["scheduler"] = {**rep[1]["scheduler"], "max_num_batched_tokens": 128}
    if mutation == "preempted":
        rep[0]["preempted"] = ["1-b"]
    if mutation == "new_token_ids":
        rep[1]["requests"][0]["new_token_ids"] = [999]
    result = run_e6.compare_replay(make_record("a"), rep)
    assert result["verdict_P2"] == "NOT COMPARABLE", result
    assert {k: result["first_mismatch"][k] for k in ("pass", "slot", "field")} == expected
    assert result["rows_compared"] == 0


def strip_to_schema2(steps):
    out = copy.deepcopy(steps)
    for s in out:
        s["schema"] = 2
        for k in ("admitted", "resumed", "preempted", "finished", "model_runner", "scheduler"):
            s.pop(k, None)
        for r in s["requests"]:
            r.pop("new_token_ids", None)
            r.pop("prompt_sha256", None)
    return out


def test_schema2_records_are_rejected_with_a_reason_not_skipped():
    legacy = strip_to_schema2(make_record())
    assert trace_errors(legacy) == []  # still a valid schema 2 trace for E3/E5
    result = run_e6.compare_replay(legacy, make_record("b"))
    assert result["verdict_P2"] == "INVALID"
    assert any("schema 2 records" in e and "cannot drive or validate a replay" in e for e in result["validation_errors"])
    plan = run_e6.replay_plan(legacy, RUN_META, None)
    assert plan["passes"] == [] and any("schema 2" in e for e in plan["errors"])
    plan = run_e6.boundary_plan(legacy, 1, RUN_META["resolved_scheduler"])
    assert plan["eligible"] is False and "schema 2" in plan["reason"]


def test_row_with_wrong_token_count_is_invalid_with_a_reason():
    assert shape_hook.row_validity([1, 2], 3, 0, [1, 2, 3]) == "len(new_token_ids) 2 != q 3"
    assert "unavailable" in shape_hook.row_validity(None, 1, 0, [1])
    assert "placeholder" in shape_hook.row_validity([-1], 1, 3, [1, 2, 3])
    assert "disagree" in shape_hook.row_validity([9, 2], 2, 0, [1, 2, 3])
    assert "prompt token ids unavailable" in shape_hook.row_validity([5], 1, 3, None)
    assert shape_hook.row_validity([3, 7], 2, 2, [1, 2, 3]) is None  # crosses the prompt boundary correctly
    rec = make_record()
    rec[1]["requests"][2]["new_token_ids"] = [30]  # q is 2
    errors = trace_errors(rec)
    assert any("lacks 2 recorded token ids" in e for e in errors)
    rec[1]["requests"][2]["invalid"] = "len(new_token_ids) 1 != q 2"
    assert any("invalid row 2" in e for e in trace_errors(rec))
    assert run_e6.compare_replay(rec, make_record("b"))["verdict_P2"] == "INVALID"


def test_schedule_extraction_fails_closed():
    rec = make_record()
    rec[1]["admitted"][0]["num_computed_tokens"] = 2
    assert any("cache hit" in e for e in run_e6.extract_schedule(rec)["errors"])
    rec = make_record()
    rec[0]["admitted"][0]["prompt_sha256"] = h("z")
    assert any("prompt_sha256" in e for e in run_e6.extract_schedule(rec)["errors"])
    rec = make_record()
    rec[1]["admitted"] = []
    assert any("before any admission" in e for e in run_e6.extract_schedule(rec)["errors"])
    rec = make_record()
    rec[0]["admitted"][0]["prompt_token_ids"] = None
    assert any("without prompt token ids" in e for e in run_e6.extract_schedule(rec)["errors"])


def test_forced_continuations_check_greedy_consistency_and_outputs():
    rec = make_record()
    requests = run_e6.extract_schedule(rec)["requests"]
    forced, errors = run_e6.forced_continuations(rec, requests, run_e6.normalise_outputs(outputs()))
    assert errors == [] and forced == OUTPUTS
    bad = copy.deepcopy(rec)
    bad[0]["requests"][0]["argmax"] = 555  # the recorded free run consumed 100 next, so 555 cannot be its greedy choice
    _, errors = run_e6.forced_continuations(bad, requests, None)
    assert any("not a greedy free run" in e for e in errors)
    gap = copy.deepcopy(rec)
    gap[1]["requests"][0]["computed"] = 4
    gap[1]["requests"][0]["kv"] = 5
    _, errors = run_e6.forced_continuations(gap, requests, None)
    assert any("gap" in e for e in errors)
    wrong_outputs = run_e6.normalise_outputs(outputs())
    wrong_outputs["1"]["tokens"] = [200, 999]
    _, errors = run_e6.forced_continuations(rec, requests, wrong_outputs)
    assert any("outputs.json tokens differ" in e for e in errors)


def test_replay_plan_requires_a_configured_greedy_record():
    rec = make_record()
    meta = copy.deepcopy(RUN_META)
    meta["args"]["prefix_caching"] = 1
    assert any("prefix caching" in e for e in run_e6.replay_plan(rec, meta, None)["errors"])
    meta = copy.deepcopy(RUN_META)
    meta["status"] = "INVALID"
    assert any("status" in e for e in run_e6.replay_plan(rec, meta, None)["errors"])
    meta = copy.deepcopy(RUN_META)
    meta["args"]["max_tokens"] = 3
    assert any("max_tokens is 3" in e for e in run_e6.replay_plan(rec, meta, None)["errors"])
    meta = copy.deepcopy(RUN_META)
    del meta["args"]["max_tokens"]
    assert any("lacks args.max_tokens" in e for e in run_e6.replay_plan(rec, meta, None)["errors"])


def test_hook_and_analysis_prompt_digests_agree():
    ids = [0, 1, 2**31, 2**32 - 1]
    assert shape_hook.prompt_sha256(ids) == prompt_sha256(ids)
    assert shape_hook.prompt_sha256([1, 2]) != shape_hook.prompt_sha256([2, 1])
    with pytest.raises(ValueError):
        shape_hook.prompt_sha256([-1])


class FakeRequestState:
    def __init__(self, prompt, outputs):
        self.prompt_token_ids, self.output_token_ids, self.num_prompt_tokens = prompt, outputs, len(prompt)

    def get_token_id(self, idx):  # gpu_input_batch.py:79-89
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        if idx - self.num_prompt_tokens < len(self.output_token_ids):
            return self.output_token_ids[idx - self.num_prompt_tokens]
        return -1


def fake_runner(requests, req_ids, computed, prompts):
    return NS(requests=requests,
              input_batch=NS(req_ids=req_ids, num_reqs=len(req_ids), num_computed_tokens_cpu=computed, num_prompt_tokens=prompts),
              vllm_config=NS(compilation_config=NS(mode=3, cudagraph_mode=NS(name="FULL_AND_PIECEWISE")),
                             parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False),
                             scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=2048, enable_chunked_prefill=True, async_scheduling=False)),
              attn_groups=[[NS(backend=NS(get_name=lambda: "FLASH_ATTN"))]])


def fake_scheduler_output(new, sched, finished=(), preempted=(), resumed=()):
    return NS(scheduled_new_reqs=[NS(req_id=r, prompt_token_ids=p, num_computed_tokens=0) for r, p in new],
              scheduled_cached_reqs=NS(resumed_req_ids=set(resumed)), num_scheduled_tokens=sched,
              total_num_scheduled_tokens=sum(sched.values()), finished_req_ids=set(finished), preempted_req_ids=set(preempted))


def test_hook_records_schema3_fields_on_the_v1_runner_layout(monkeypatch):
    monkeypatch.setattr(shape_hook, "_state", {**shape_hook._state, "seen": set(), "prompts": {}, "hashes": None, "batch": None, "dispatch": {"cudagraph_mode": "FULL"}, "step": 7})
    monkeypatch.delenv("SHAPE_EXPECT_COMPILE", raising=False)
    monkeypatch.delenv("SHAPE_EXPECT_GRAPHS", raising=False)
    requests = {"0-x": FakeRequestState([10, 11, 12], []), "1-x": FakeRequestState([20, 21], [])}
    runner = fake_runner(requests, ["0-x", "1-x"], [0, 0], [3, 2])
    rec = shape_hook._record(runner, fake_scheduler_output([("0-x", [10, 11, 12]), ("1-x", [20, 21])], {"0-x": 3, "1-x": 2}))
    assert rec["schema"] == 3 and rec["model_runner"] == "v1"
    assert [a["req"] for a in rec["admitted"]] == ["0-x", "1-x"]
    assert rec["admitted"][0]["prompt_token_ids"] == [10, 11, 12] and rec["admitted"][0]["prompt_sha256"] == prompt_sha256([10, 11, 12])
    assert [r["new_token_ids"] for r in rec["requests"]] == [[10, 11, 12], [20, 21]]
    assert all("invalid" not in r for r in rec["requests"])
    assert rec["finished"] == [] and rec["preempted"] == [] and rec["resumed"] == []
    assert rec["scheduler"]["async_scheduling"] is False
    for k in ("q", "computed", "kv", "prompt", "phase", "cache_hit", "h", "argmax"):
        assert k in rec["requests"][0]
    # decode pass: the sampled token is in output_token_ids; a request whose id is unknown to the host is invalid
    requests["0-x"].output_token_ids.append(100)
    runner = fake_runner(requests, ["0-x", "1-x"], [3, 2], [3, 2])
    rec = shape_hook._record(runner, fake_scheduler_output([], {"0-x": 1, "1-x": 1}, finished=["9-x"], preempted=["8-x"], resumed=["1-x"]))
    assert rec["admitted"] == [] and rec["finished"] == ["9-x"] and rec["preempted"] == ["8-x"] and rec["resumed"] == ["1-x"]
    assert rec["requests"][0]["new_token_ids"] == [100] and "invalid" not in rec["requests"][0]
    assert rec["requests"][1]["new_token_ids"] == [-1] and "placeholder" in rec["requests"][1]["invalid"]
    assert real_steps([rec]) == [rec]
    assert any("invalid row 1" in e for e in trace_errors([rec]))


def params(forced=None, max_tokens=None, temperature=0.0):
    extra = {tf.EXTRA_ARGS_KEY: forced} if forced is not None else None
    return NS(extra_args=extra, max_tokens=len(forced) if forced is not None and max_tokens is None else max_tokens, temperature=temperature)


def test_teacher_forcing_processor_masks_every_logit_but_the_forced_token(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SHAPE_HOOK_OUT", str(tmp_path))
    proc = tf.TeacherForcingLogitsProcessor(None, torch.device("cpu"), False)
    assert proc.is_argmax_invariant() is False
    out0, out1 = [], []
    proc.update_state(tf.BatchUpdate(batch_size=2, removed=[], added=[(0, params([5, 7]), None, out0), (1, params(None), None, out1)], moved=[]))
    logits = torch.randn(2, 11)
    logits[0, 3] = 100.0
    logits[1, 4] = 100.0
    original = logits.clone()
    out = proc.apply(logits)
    assert out is logits
    assert int(torch.argmax(out[0])) == 5 and float(out[0, 5]) == float(original[0, 5])
    masked = torch.ones(11, dtype=torch.bool)
    masked[5] = False
    assert bool((out[0][masked] == torch.finfo(torch.float32).min).all())
    assert torch.equal(out[1], original[1])  # not forced, untouched
    out0.append(5)  # the engine appends the sampled (forced) token to the live output list
    out = proc.apply(original.clone())
    assert int(torch.argmax(out[0])) == 7
    out0.append(7)
    with pytest.raises(RuntimeError, match="only 2 forced"):
        proc.apply(original.clone())
    log = tf.read_forcing_log(tmp_path)
    assert [c["call"] for c in log] == [1, 2] and len(log[0]["rows"]) == 1
    assert log[0]["rows"][0] == {"index": 0, "generated": 0, "forced": 5, "argmax_before_forcing": 3, "equal": False}
    assert log[1]["rows"][0]["generated"] == 1 and log[1]["rows"][0]["forced"] == 7


def test_teacher_forcing_processor_fails_closed_on_placeholders_and_bad_params():
    torch = pytest.importorskip("torch")
    proc = tf.TeacherForcingLogitsProcessor(None, torch.device("cpu"), False)
    proc.update_state(tf.BatchUpdate(batch_size=1, removed=[], added=[(0, params([5]), None, [-1])], moved=[]))
    with pytest.raises(RuntimeError, match="placeholder"):
        proc.apply(torch.zeros(1, 8))
    proc = tf.TeacherForcingLogitsProcessor(None, torch.device("cpu"), False)
    proc.update_state(tf.BatchUpdate(batch_size=1, removed=[], added=[(0, params([5]), None, [])], moved=[]))
    bad = torch.zeros(1, 8)
    bad[0, 5] = float("-inf")
    with pytest.raises(RuntimeError, match="non-finite"):
        proc.apply(bad)
    with pytest.raises(ValueError, match="max_tokens"):
        tf.TeacherForcingLogitsProcessor.validate_params(params([1, 2], max_tokens=3))
    with pytest.raises(ValueError, match="greedy"):
        tf.TeacherForcingLogitsProcessor.validate_params(params([1], temperature=0.7))
    with pytest.raises(ValueError, match="non-negative"):
        tf.TeacherForcingLogitsProcessor.validate_params(params([1, -2]))
    assert tf.TeacherForcingLogitsProcessor.validate_params(params(None)) is None


def test_teacher_forcing_state_follows_batch_moves_and_removals():
    torch = pytest.importorskip("torch")
    proc = tf.TeacherForcingLogitsProcessor(None, torch.device("cpu"), False)
    proc.update_state(tf.BatchUpdate(batch_size=2, removed=[], added=[(0, params([5]), None, []), (1, params([6]), None, [])], moved=[]))
    proc.update_state(tf.BatchUpdate(batch_size=1, removed=[0], added=[], moved=[(1, 0, tf.MoveDirectionality.UNIDIRECTIONAL)]))
    assert proc.plan() == [(0, 0, 6)]
    proc.update_state(tf.BatchUpdate(batch_size=2, removed=[], added=[(1, params([9]), None, [])], moved=[(0, 1, tf.MoveDirectionality.SWAP)]))
    assert proc.plan() == [(0, 0, 9), (1, 0, 6)]
    proc.update_state(None)
    assert proc.plan() == [(0, 0, 9), (1, 0, 6)]


def forcing_log_for(steps):
    return [{"call": i + 1, "rank": 0, "rows": [{"index": j, "generated": 0, "forced": r["argmax"], "argmax_before_forcing": r["argmax"], "equal": True}
                                                 for j, r in enumerate(s["requests"])]} for i, s in enumerate(steps)]


def test_forcing_log_must_agree_with_the_hook():
    rep = make_record("b")
    assert run_e6.forcing_consistency(rep, forcing_log_for(rep), expect_forcing=True)["errors"] == []
    assert run_e6.forcing_consistency(rep, [], expect_forcing=True)["errors"] == ["no teacher-forcing log; the processor was not active"]
    log = forcing_log_for(rep)
    log[1]["rows"][0]["argmax_before_forcing"] = 1
    assert any("differs from the processor" in e for e in run_e6.forcing_consistency(rep, log, expect_forcing=True)["errors"])
    assert any("forcing calls but" in e for e in run_e6.forcing_consistency(rep, log[:2], expect_forcing=True)["errors"])


def decode_record(suffix="a"):
    """Two requests, max_tokens 3; pass 2 is a uniform decode pass eligible for the boundary rebuild."""
    return [
        pass_record(0, [row("0", 3, 0, 3, PROMPTS["0"], 100, suffix, cache_hit=0), row("1", 2, 0, 2, PROMPTS["1"], 200, suffix, cache_hit=0)], admitted_rows=("0", "1"), suffix=suffix),
        pass_record(1, [row("0", 1, 3, 3, [100], 101, suffix), row("1", 1, 2, 2, [200], 201, suffix)], suffix=suffix),
        pass_record(2, [row("0", 1, 4, 3, [101], 102, suffix), row("1", 1, 3, 2, [201], 202, suffix)], suffix=suffix),
    ]


def rebuilt_boundary(suffix="r"):
    return [
        pass_record(0, [row("0", 4, 0, 4, [10, 11, 12, 100], 101, suffix, cache_hit=0, tag="p0"), row("1", 3, 0, 3, [20, 21, 200], 201, suffix, cache_hit=0, tag="p1")], admitted_rows=("0", "1"), suffix=suffix),
        pass_record(1, [row("0", 1, 4, 4, [101], 102, suffix, tag="04"), row("1", 1, 3, 3, [201], 202, suffix, tag="13")], suffix=suffix),
    ]


def test_boundary_plan_gates_on_all_decode_rows_and_scheduler_limits():
    rec = decode_record()
    limits = RUN_META["resolved_scheduler"]
    plan = run_e6.boundary_plan(rec, 2, limits)
    assert plan["eligible"] is True
    assert [(r["slot"], r["prefix"], r["forced"]) for r in plan["rows"]] == [("0", [10, 11, 12, 100], [101, 102]), ("1", [20, 21, 200], [201, 202])]
    assert plan["prefix_tokens_total"] == 7
    assert run_e6.boundary_plan(make_record(), 1, limits)["reason"].startswith("row 2 at pass 1 is a prefill row")
    assert "max_num_batched_tokens 5" in run_e6.boundary_plan(rec, 2, {**limits, "max_num_batched_tokens": 5})["reason"]
    assert "max_num_seqs" in run_e6.boundary_plan(rec, 2, {**limits, "max_num_seqs": 1})["reason"]
    assert "lacks resolved_scheduler" in run_e6.boundary_plan(rec, 2, {})["reason"]
    assert "outside" in run_e6.boundary_plan(rec, 7, limits)["reason"]


def test_boundary_comparison_excludes_prompt_but_requires_the_rest():
    rec = decode_record()
    result = run_e6.compare_boundary(rec, 2, rebuilt_boundary())
    assert result["verdict_P2_boundary"] == "IDENTICAL" and result["rows_compared"] == 2 and result["excluded_row_fields"] == ["prompt", "cache_hit"]
    differs = rebuilt_boundary()
    differs[1]["requests"][1]["logits_h"] = h("q")
    result = run_e6.compare_boundary(rec, 2, differs)
    assert result["verdict_P2_boundary"] == "DIFFERS" and result["differing_slots"] == ["1"]
    wrong = rebuilt_boundary()
    wrong[1]["requests"][0]["computed"] = 5
    wrong[1]["requests"][0]["kv"] = 6
    wrong[0]["requests"][0]["q"] = 5
    wrong[0]["requests"][0]["kv"] = 5
    wrong[0]["requests"][0]["new_token_ids"] = [10, 11, 12, 100, 7]
    wrong[0]["total_scheduled"] = 8
    result = run_e6.compare_boundary(rec, 2, wrong)
    assert result["verdict_P2_boundary"] == "NOT COMPARABLE" and result["first_mismatch"]["field"] == "rebuilt_prefill_shape"
    assert run_e6.compare_boundary(rec, 2, rebuilt_boundary()[:1])["first_mismatch"]["field"] == "trace_length"
    tampered = rebuilt_boundary()  # the prefix is derived from the record, never trusted from boundary.json
    tampered[0]["requests"][1]["new_token_ids"] = [20, 21, 999]
    result = run_e6.compare_boundary(rec, 2, tampered)
    assert result["verdict_P2_boundary"] == "NOT COMPARABLE"
    assert result["first_mismatch"] == {"pass": 0, "slot": "1", "field": "rebuilt_prefix_token_ids", "recorded": [20, 21, 200], "replayed": [20, 21, 999]}
    other_limits = decode_record()
    for s in other_limits:
        s["scheduler"] = {**s["scheduler"], "max_num_batched_tokens": 4096}
    assert run_e6.compare_boundary(other_limits, 2, rebuilt_boundary())["first_mismatch"]["field"] == "scheduler"


def write_arm(d: Path, steps, outs, meta, forcing=None):
    (d / "hook").mkdir(parents=True)
    (d / "hook/rank0.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n")
    (d / "outputs.json").write_text(json.dumps(outs))
    (d / "run.json").write_text(json.dumps(meta))
    if forcing is not None:
        (d / "hook/forcing_rank0.jsonl").write_text("\n".join(json.dumps(c) for c in forcing) + "\n")


def replay_meta(record_dir: Path, mode="replay", **kw):
    digests = run_e6._digests(run_e6._record_files(record_dir))
    return {**RUN_META, "run_id": "rep-1", "mode": mode, "recorded_run_id": "rec-1", "record_sha256": digests, **kw}


def test_compare_writes_summary_without_rewriting_raw_files(tmp_path):
    arm = tmp_path / "results/e6/arm"
    write_arm(arm / "record", make_record("a"), outputs("a"), RUN_META)
    rep = make_record("b")
    write_arm(arm / "replay", rep, outputs("b"), replay_meta(arm / "record"), forcing_log_for(rep))
    raw = {p: p.read_bytes() for p in arm.rglob("*") if p.is_file()}
    assert run_e6.compare(arm / "record", arm / "replay") == 0
    summary = json.loads((arm / "summary.json").read_text())
    assert summary["verdict_P2"] == "IDENTICAL" and summary["requirements_met"] is True and summary["outputs_identical"] is True
    assert summary["forcing_log"]["calls"] == 3 and summary["validation_errors"] == []
    # Arms are named relative to their parent, never by absolute path, so reanalyse.py --check can rebuild in a copy.
    assert (summary["recorded_arm"], summary["replay_arm"]) == ("arm/record", "arm/replay")
    assert set(summary["input_sha256"]) >= {"record/hook/rank0.jsonl", "replay/hook/forcing_rank0.jsonl", "replay/run.json"}
    assert {p: p.read_bytes() for p in arm.rglob("*") if p.is_file() and p.name != "summary.json"} == raw
    out = _tables(tmp_path / "results")
    assert "### E6" in out and "| arm | 3 | True | 6 | 0 | None | 0 | IDENTICAL |" in out


def test_compare_is_invalid_without_forcing_log_or_with_schema2_record(tmp_path):
    arm = tmp_path / "results/e6/arm"
    write_arm(arm / "record", make_record("a"), outputs("a"), RUN_META)
    write_arm(arm / "replay", make_record("b"), outputs("b"), replay_meta(arm / "record"))
    assert run_e6.compare(arm / "record", arm / "replay") == 1
    summary = json.loads((arm / "summary.json").read_text())
    assert summary["verdict_P2"] == "INVALID" and any("no teacher-forcing log" in e for e in summary["validation_errors"])
    legacy = tmp_path / "results/e6/legacy"
    write_arm(legacy / "record", strip_to_schema2(make_record("a")), outputs("a"), RUN_META)
    rep = make_record("b")
    write_arm(legacy / "replay", rep, outputs("b"), replay_meta(legacy / "record"), forcing_log_for(rep))
    assert run_e6.compare(legacy / "record", legacy / "replay") == 1
    summary = json.loads((legacy / "summary.json").read_text())
    assert summary["verdict_P2"] == "INVALID" and any("schema 2 records" in e for e in summary["validation_errors"])


def test_compare_is_invalid_when_provenance_fails(tmp_path):
    arm = tmp_path / "results/e6/arm"
    write_arm(arm / "record", make_record("a"), outputs("a"), RUN_META)
    rep = make_record("b")
    # A second replay directory gets its own summary name, so it does not overwrite the first's.
    write_arm(arm / "replay_tampered", rep, outputs("b"), {**replay_meta(arm / "record"), "record_sha256": {"run.json": "0" * 64}}, forcing_log_for(rep))
    assert run_e6.compare(arm / "record", arm / "replay_tampered") == 1
    summary = json.loads((arm / "summary_replay_tampered.json").read_text())
    assert summary["verdict_P2"] == "INVALID" and any("digests the replay run.json recorded" in e for e in summary["validation_errors"])
    # The digests were right when the replay ran; the record changed afterwards.
    write_arm(arm / "replay", rep, outputs("b"), replay_meta(arm / "record"), forcing_log_for(rep))
    (arm / "record/outputs.json").write_text(json.dumps(outputs("a")) + "\n")
    assert run_e6.compare(arm / "record", arm / "replay") == 1
    summary = json.loads((arm / "summary.json").read_text())
    assert summary["verdict_P2"] == "INVALID" and any("raw record files must not change" in e for e in summary["validation_errors"])
    (arm / "record/outputs.json").write_text(json.dumps(outputs("a")))
    write_arm(arm / "replay_wrong_run", rep, outputs("b"), {**replay_meta(arm / "record"), "recorded_run_id": "other", "resolved_model_runner": "v2"}, forcing_log_for(rep))
    assert run_e6.compare(arm / "record", arm / "replay_wrong_run") == 1
    errors = json.loads((arm / "summary_replay_wrong_run.json").read_text())["validation_errors"]
    assert any("recorded run id" in e for e in errors) and any("different model runners" in e for e in errors)
    out = _tables(tmp_path / "results")
    assert out.count("| arm |") == 3 and out.count("INVALID") == 3
    assert set(reanalyse.derived(tmp_path)) == {"results/e6/arm/summary.json", "results/e6/arm/summary_replay_tampered.json", "results/e6/arm/summary_replay_wrong_run.json"}


def test_compare_boundary_arm_writes_boundary_summary(tmp_path):
    arm = tmp_path / "results/e6/arm"
    write_arm(arm / "record", decode_record("a"), {f"{s}-a": {"hash": h("o" + s), "tokens": t} for s, t in {"0": [100, 101, 102], "1": [200, 201, 202]}.items()},
              {**RUN_META, "args": {**RUN_META["args"], "max_tokens": 3}})
    reb = rebuilt_boundary()
    write_arm(arm / "boundary_2", reb, {"0-r": {"hash": h("b0"), "tokens": [101, 102]}, "1-r": {"hash": h("b1"), "tokens": [201, 202]}}, replay_meta(arm / "record", "boundary"), forcing_log_for(reb))
    plan = run_e6.boundary_plan(decode_record("a"), 2, RUN_META["resolved_scheduler"])
    (arm / "boundary_2/boundary.json").write_text(json.dumps(plan))
    assert run_e6.compare(arm / "record", arm / "boundary_2") == 0
    summary = json.loads((arm / "boundary_2.json").read_text())
    assert summary["verdict_P2_boundary"] == "IDENTICAL" and summary["pass"] == 2 and summary["rows_compared"] == 2
    assert "E6 boundary arm pass 2: 2 rows compared, 0 differing: IDENTICAL" in _tables(tmp_path / "results")
    assert reanalyse.rebuild(tmp_path) == 0
    assert set(reanalyse.derived(tmp_path)) == {"results/e6/arm/boundary_2.json"}


def _tables(root: Path) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tables.main([str(root)])
    return buf.getvalue()


def test_tables_print_nothing_about_e6_without_results(tmp_path):
    (tmp_path / "e3").mkdir()
    assert "E6" not in _tables(tmp_path)


def test_reanalyse_check_is_clean_on_committed_results():
    proc = subprocess.run([sys.executable, str(ROOT / "probes/shape/reanalyse.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "match the raw observations" in proc.stdout
