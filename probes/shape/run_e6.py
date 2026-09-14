"""E6, reconstruction of a recorded execution in a fresh process (prediction P2).

    python probes/shape/run_e6.py --record --model Qwen/Qwen2.5-7B-Instruct --revision MODEL_COMMIT --out results_new/e6 [--mixed]
    python probes/shape/run_e6.py --replay results_new/e6/ARM/record --out results_new/e6/ARM/replay
    python probes/shape/run_e6.py --replay results_new/e6/ARM/record --out results_new/e6/ARM/boundary_12 --boundary 12
    python probes/shape/run_e6.py --compare results_new/e6/ARM/record results_new/e6/ARM/replay

``--record`` runs the E3 arrival script (eight requests, five scheduler steps,
eight more) on the stock in-process engine core with the shape hook writing
schema 3 records: per row the consumed token ids, per pass the admitted
requests with their prompt token ids, and the finished, resumed and
preempted request ids. Prefix caching is off, sampling is greedy with
``ignore_eos`` and a fixed ``max_tokens``.

``--replay`` rebuilds the schedule from the record alone. Every recorded pass
index gets exactly the requests the record admitted there, as
``TokensPrompt`` with the recorded ``prompt_token_ids``
(vllm/inputs/llm.py:106-110) and the same sampling settings, added through
``LLMEngine.add_request`` (vllm/v1/engine/llm_engine.py:218-296) before the
``step`` (llm_engine.py:298) that produces that pass. The forward passes are
counted at the scheduler: ``EngineCore.step`` schedules once per call
(vllm/v1/engine/core.py:583-620) and the in-process client exposes the core
(vllm/v1/engine/core_client.py:306-320), so the driver wraps
``Scheduler.schedule`` (vllm/v1/core/sched/scheduler.py:476) and counts
outputs with scheduled tokens. The recorded continuation is teacher-forced
through ``teacher_forcing.TeacherForcingLogitsProcessor`` so the replayed
inputs equal the recorded ones even after a numerical divergence; the hook's
per-row ``argmax`` in the replay is the free-running choice, and comparing it
with the recorded ``argmax`` (the forced token) says, per pass and row,
whether free running would have diverged.

``--compare`` fails closed: equal trace lengths, request slot order per pass,
record-level dispatch and configuration fields, every row's recorded shape
and every row's ``new_token_ids`` are requirements; the first offending pass,
slot and field are named and the verdict is NOT COMPARABLE. Only when all of
that holds are ``h``, ``argmax`` and ``logits_h`` compared per row and pass.
Schema 2 records are rejected with a reason; they carry no token ids or
admission events.

``--boundary T`` rebuilds recorded pass T (index among forward passes) with
the stock scheduler when every row of T is a single-token decode row and the
rows' prefixes fit one prefill pass; see ``boundary_plan`` for the conditions
and the RUNBOOK for what such a comparison can and cannot show.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Custom logits processors are unsupported on the V2 runner (vllm/config/vllm.py:2458-2467);
# both arms run the V1 runner and the hook records which one ran.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from comparison import load_steps, normalise_outputs, output_trace_errors
from shape_common import (add_common_args, analysis_metadata, arm_name, engine_kwargs, env_with_hook, mixed_prompts, outputs_record,
                          prompt_sha256, read_hook, real_steps, record_run, record_schema, slot_of, trace_errors, write_json)
from teacher_forcing import EXTRA_ARGS_KEY, TeacherForcingLogitsProcessor, read_forcing_log

RECORD_FIELDS = ("num_reqs", "total_scheduled", "dispatch", "attention_backend", "parallel", "resolved_compile", "resolved_cudagraph",
                 "model_runner", "finished", "resumed", "preempted")
ROW_FIELDS = ("q", "computed", "prompt", "kv", "cache_hit")
BOUNDARY_ROW_FIELDS = ("q", "computed", "kv", "phase")  # prompt is the rebuilt prefix length by construction
HASH_FIELDS = ("h", "argmax", "logits_h")
ID_FIELDS = ("finished", "resumed", "preempted")


# ----------------------------------------------------------------------------
# Pure record analysis (CPU, tested)
# ----------------------------------------------------------------------------

def schema_errors(steps: list[dict]) -> list[str]:
    """Schema 2 records cannot drive or validate a replay; say so per pass."""
    errors = []
    for i, s in enumerate(steps):
        schema = record_schema(s)
        if schema < 3:
            errors.append(f"pass {i} (step {s.get('step')}) has schema {schema}: schema 2 records carry lengths and hashes "
                          "but no token ids or admission events, so they cannot drive or validate a replay")
            continue
        for key in ("admitted", *ID_FIELDS):
            if key not in s:
                errors.append(f"pass {i} lacks {key}")
        for r in s.get("requests", []):
            for key in ("new_token_ids", "prompt_sha256"):
                if key not in r:
                    errors.append(f"pass {i} row {slot_of(r['req'])} lacks {key}")
    return errors


def record_errors(steps: list[dict]) -> list[str]:
    return schema_errors(steps) + trace_errors(steps)


def _slots(values) -> list[str]:
    return sorted(slot_of(v) for v in (values or []))


def extract_schedule(steps: list[dict]) -> dict:
    """Admissions per recorded pass index, keyed by request slot, with fail-closed checks."""
    passes, requests, errors = [], {}, []
    for i, s in enumerate(steps):
        admitted = []
        for a in s.get("admitted", []):
            slot = slot_of(a["req"])
            ids = a.get("prompt_token_ids")
            if slot in requests:
                errors.append(f"slot {slot} admitted twice (passes {requests[slot]['admitted_at']} and {i})")
            if ids is None:
                errors.append(f"slot {slot} admitted without prompt token ids at pass {i}")
            elif a.get("prompt_sha256") != prompt_sha256(ids):
                errors.append(f"slot {slot}: recorded prompt_sha256 does not match its prompt token ids")
            if a.get("num_computed_tokens"):
                errors.append(f"slot {slot} admitted with a {a['num_computed_tokens']}-token cache hit; prefix caching must be off for a replay")
            requests[slot] = {"prompt_token_ids": ids, "admitted_at": i, "prompt_sha256": a.get("prompt_sha256")}
            admitted.append(slot)
        for r in s.get("requests", []):
            slot = slot_of(r["req"])
            if slot not in requests:
                errors.append(f"slot {slot} appears in pass {i} before any admission record")
            elif r.get("prompt_sha256") != requests[slot]["prompt_sha256"]:
                errors.append(f"slot {slot}: row prompt_sha256 differs from its admission at pass {i}")
        passes.append({"index": i, "step": s.get("step"), "admitted": admitted, "rows": [slot_of(r["req"]) for r in s.get("requests", [])],
                       **{k: _slots(s.get(k)) for k in ID_FIELDS}})
    return {"passes": passes, "requests": requests, "errors": errors}


def token_sequences(steps: list[dict], requests: dict) -> tuple[dict[str, list[int]], list[str]]:
    """Per slot, the token sequence the record shows being consumed: prompt then generated positions in order."""
    rows_by_slot: dict[str, list] = defaultdict(list)
    for i, s in enumerate(steps):
        for r in s.get("requests", []):
            rows_by_slot[slot_of(r["req"])].append((i, r))
    sequences, errors = {}, []
    for slot, info in requests.items():
        prompt = list(info.get("prompt_token_ids") or [])
        tokens = list(prompt)
        for i, r in rows_by_slot.get(slot, []):
            c = int(r["computed"])
            for offset, t in enumerate(r.get("new_token_ids") or []):
                p = c + offset
                if p < len(prompt):
                    if int(t) != int(prompt[p]):
                        errors.append(f"slot {slot} pass {i}: consumed id at prompt position {p} differs from the prompt")
                        break
                elif p == len(tokens):
                    tokens.append(int(t))
                elif p < len(tokens):
                    if tokens[p] != int(t):
                        errors.append(f"slot {slot} pass {i}: position {p} consumed twice with different ids")
                        break
                else:
                    errors.append(f"slot {slot} pass {i}: gap before position {p}; the record does not cover every consumed position")
                    break
        sequences[slot] = tokens
    return sequences, errors


def forced_continuations(steps: list[dict], requests: dict, outputs: dict | None = None) -> tuple[dict[str, list[int]], list[str]]:
    """Per slot, the greedy continuation the record shows: consumed generated ids plus the final row's argmax."""
    sequences, errors = token_sequences(steps, requests)
    rows_by_slot: dict[str, list] = defaultdict(list)
    for i, s in enumerate(steps):
        for r in s.get("requests", []):
            rows_by_slot[slot_of(r["req"])].append((i, r))
    forced = {}
    for slot, info in requests.items():
        prompt = list(info.get("prompt_token_ids") or [])
        rows = rows_by_slot.get(slot, [])
        tokens = sequences[slot]
        if not rows:
            errors.append(f"slot {slot} admitted but never scheduled")
            continue
        for i, r in rows[:-1]:
            kv = int(r["kv"])
            if kv >= len(prompt) and kv < len(tokens) and r.get("argmax") != tokens[kv]:
                errors.append(f"slot {slot} pass {i}: recorded argmax {r.get('argmax')} is not the id consumed next at position {kv} "
                              f"({tokens[kv]}); the record is not a greedy free run")
        last = rows[-1][1]
        if last.get("argmax") is None:
            errors.append(f"slot {slot}: final row lacks an argmax")
            continue
        if int(last["kv"]) != len(tokens):
            errors.append(f"slot {slot}: final row ends at position {last['kv']} but {len(tokens)} positions are known")
        forced[slot] = tokens[len(prompt):] + [int(last["argmax"])]
        if outputs is not None and outputs.get(slot, {}).get("tokens") != forced[slot]:
            errors.append(f"slot {slot}: outputs.json tokens differ from the continuation derived from the hook record")
    return forced, errors


def replay_plan(steps: list[dict], run_meta: dict, outputs: dict | None) -> dict:
    """Everything a fresh process needs to replay the record, or the reasons it cannot."""
    errors = record_errors(steps)
    if errors:
        return {"passes": [], "requests": {}, "max_tokens": None, "errors": errors, "notes": []}
    schedule = extract_schedule(steps)
    errors = list(schedule["errors"])
    forced, more = forced_continuations(steps, schedule["requests"], outputs)
    errors += more
    args = run_meta.get("args") or {}
    max_tokens = args.get("max_tokens")
    if max_tokens is None:
        errors.append("run.json lacks args.max_tokens")
    else:
        for slot, f in forced.items():
            if len(f) != int(max_tokens):
                errors.append(f"slot {slot}: derived continuation has {len(f)} tokens, max_tokens is {max_tokens}")
    if int(args.get("prefix_caching", 1) or 0) != 0:
        errors.append("recorded arm ran with prefix caching on; E6 replays require it off")
    if run_meta.get("status") != "configured":
        errors.append(f"recorded run.json status is {run_meta.get('status')!r}, not 'configured'")
    if outputs is not None and set(outputs) != set(schedule["requests"]):
        errors.append("outputs.json request set differs from the admitted requests")
    notes = []
    if any(p["preempted"] or p["resumed"] for p in schedule["passes"]):
        notes.append("the record contains preemptions; the replay relies on the stock scheduler reproducing them and the comparison checks it")
    return {"passes": schedule["passes"],
            "requests": {slot: {"prompt_token_ids": v["prompt_token_ids"], "forced": forced.get(slot)} for slot, v in schedule["requests"].items()},
            "max_tokens": max_tokens, "errors": errors, "notes": notes}


def _record_field(s: dict, key: str):
    return _slots(s.get(key)) if key in ID_FIELDS else s.get(key)


def compare_replay(recorded: list[dict], replayed: list[dict]) -> dict:
    """Pass-by-pass comparison of a record with its replay; requirements first, observations only when they all hold."""
    errors = [f"recorded: {e}" for e in record_errors(recorded)] + [f"replayed: {e}" for e in record_errors(replayed)]
    result = {"passes_recorded": len(recorded), "passes_replayed": len(replayed), "validation_errors": errors,
              "first_mismatch": None, "first_divergence": None, "rows_compared": 0, "rows_differing": 0, "passes_differing": 0,
              "hidden_identical_rows": 0, "argmax_identical_rows": 0, "logits_identical_rows": 0,
              "free_running_divergence": {"rows": 0, "passes": 0, "first": None}}
    if errors:
        result["verdict_P2"] = "INVALID"
        return result
    mismatch = None
    if len(recorded) != len(replayed):
        mismatch = {"pass": min(len(recorded), len(replayed)), "slot": None, "field": "trace_length", "recorded": len(recorded), "replayed": len(replayed)}
    for i, (a, b) in enumerate(zip(recorded, replayed)):
        if mismatch:
            break
        sa, sb = [slot_of(r["req"]) for r in a["requests"]], [slot_of(r["req"]) for r in b["requests"]]
        if sa != sb:
            mismatch = {"pass": i, "slot": None, "field": "request_order", "recorded": sa, "replayed": sb}
            break
        for key in RECORD_FIELDS:
            if _record_field(a, key) != _record_field(b, key):
                mismatch = {"pass": i, "slot": None, "field": key, "recorded": a.get(key), "replayed": b.get(key)}
                break
        if mismatch:
            break
        for ra, rb in zip(a["requests"], b["requests"]):
            for key in (*ROW_FIELDS, "new_token_ids"):
                if ra.get(key) != rb.get(key):
                    mismatch = {"pass": i, "slot": slot_of(ra["req"]), "field": key, "recorded": ra.get(key), "replayed": rb.get(key)}
                    break
            if mismatch:
                break
    if mismatch:
        result["first_mismatch"] = mismatch
        result["verdict_P2"] = "NOT COMPARABLE"
        return result
    for i, (a, b) in enumerate(zip(recorded, replayed)):
        pass_differs = pass_free = False
        for ra, rb in zip(a["requests"], b["requests"]):
            result["rows_compared"] += 1
            differing = [k for k in HASH_FIELDS if ra.get(k) != rb.get(k)]
            result["hidden_identical_rows"] += "h" not in differing
            result["argmax_identical_rows"] += "argmax" not in differing
            result["logits_identical_rows"] += "logits_h" not in differing
            if differing:
                result["rows_differing"] += 1
                pass_differs = True
                if result["first_divergence"] is None:
                    result["first_divergence"] = {"pass": i, "step_recorded": a.get("step"), "slot": slot_of(ra["req"]), "fields": differing}
            if "argmax" in differing:  # recorded argmax is the forced token; replayed argmax is the free-running choice
                result["free_running_divergence"]["rows"] += 1
                pass_free = True
                if result["free_running_divergence"]["first"] is None:
                    result["free_running_divergence"]["first"] = {"pass": i, "slot": slot_of(ra["req"])}
        result["passes_differing"] += pass_differs
        result["free_running_divergence"]["passes"] += pass_free
    result["verdict_P2"] = "IDENTICAL" if result["rows_differing"] == 0 else "DIFFERS"
    return result


def forcing_consistency(replayed: list[dict], log: list[dict], expect_forcing: bool) -> dict:
    """The processor's own log must agree with the hook: one call per forward pass, and per forced row the
    argmax it saw before masking equals the hook's argmax for that batch index."""
    result = {"calls": len(log), "forced_rows": sum(len(c.get("rows", [])) for c in log),
              "rows_where_forced_equalled_argmax": sum(bool(r.get("equal")) for c in log for r in c.get("rows", [])), "errors": []}
    if expect_forcing and not log:
        result["errors"].append("no teacher-forcing log; the processor was not active")
        return result
    if len(log) != len(replayed):
        result["errors"].append(f"{len(log)} forcing calls but {len(replayed)} forward passes")
    for i, (s, call) in enumerate(zip(replayed, log)):
        rows = s.get("requests", [])
        if expect_forcing and len(call.get("rows", [])) != len(rows):
            result["errors"].append(f"pass {i}: {len(call.get('rows', []))} forced rows but {len(rows)} batch rows")
        for r in call.get("rows", []):
            idx = int(r["index"])
            if idx >= len(rows):
                result["errors"].append(f"pass {i}: forced batch index {idx} outside {len(rows)} rows")
            elif rows[idx].get("argmax") != r.get("argmax_before_forcing"):
                result["errors"].append(f"pass {i} row {idx}: hook argmax {rows[idx].get('argmax')} differs from the processor's {r.get('argmax_before_forcing')}")
    return result


def replay_output_errors(steps: list[dict], outputs: dict, forced: dict[str, list[int]]) -> list[str]:
    """Replay outputs must be the forced continuation and the trace must reach its terminal state."""
    errors = []
    if set(outputs) != set(forced):
        errors.append("replay output request set differs from the plan")
    by_request: dict[str, list] = defaultdict(list)
    for s in steps:
        for r in s["requests"]:
            by_request[slot_of(r["req"])].append(r)
    for slot, f in forced.items():
        output = outputs.get(slot, {})
        if output.get("tokens") != f:
            errors.append(f"slot {slot}: replayed output tokens are not the forced continuation")
        observations = by_request.get(slot, [])
        if not output.get("hash") or len(observations) < len(f):
            errors.append(f"incomplete output or trace for request {slot}")
            continue
        last = observations[-1]
        if last.get("prompt") is not None and last.get("kv") != last["prompt"] + len(f) - 1:
            errors.append(f"terminal token state mismatch for request {slot}")
    return errors


def boundary_plan(steps: list[dict], T: int, limits: dict) -> dict:
    """Rebuild plan for recorded pass T with the stock scheduler, or the reason it is ineligible.

    Sound only when every row of T is a single-token decode row: the rebuilt run
    adds one request per row, in recorded order, whose prompt is the row's
    recorded prefix (positions 0..computed-1). If all prefixes fit one pass
    (``sum(computed) <= max_num_batched_tokens`` and ``rows <= max_num_seqs``)
    the stock scheduler prefills them all in the first pass and the second pass
    is a uniform decode with every row at its recorded ``computed`` and ``kv``,
    in admission order (V1 persistent batch appends in ``scheduled_new_reqs``
    order, gpu_model_runner.py:1560-1562 and gpu_input_batch.py:330-345; V2
    sorts decodes by token count with a stable sort, gpu/model_runner.py:2006-2018).
    Prefill chunk rows cannot be rebuilt because the scheduler, not the caller,
    fixes chunk boundaries (scheduler.py:751 and following). The ``prompt``
    field of the rebuilt rows is the prefix length, not the original prompt
    length; the comparison excludes it and says so.
    """
    if not 0 <= T < len(steps):
        return {"eligible": False, "reason": f"pass {T} is outside the {len(steps)} recorded forward passes"}
    errors = record_errors(steps[:T + 1])
    if errors:
        return {"eligible": False, "reason": "record invalid: " + "; ".join(errors[:3])}
    schedule = extract_schedule(steps[:T + 1])
    if schedule["errors"]:
        return {"eligible": False, "reason": "schedule invalid: " + "; ".join(schedule["errors"][:3])}
    sequences, errors = token_sequences(steps[:T + 1], schedule["requests"])
    if errors:
        return {"eligible": False, "reason": "token sequences invalid: " + "; ".join(errors[:3])}
    rows = []
    for r in steps[T]["requests"]:
        slot = slot_of(r["req"])
        if r.get("phase") != "decode" or int(r["q"]) != 1:
            return {"eligible": False, "reason": f"row {slot} at pass {T} is a {r.get('phase')} row with q={r.get('q')}; the stock "
                                                 "scheduler fixes chunk boundaries, so only all-decode passes can be rebuilt"}
        c = int(r["computed"])
        tokens = sequences[slot]
        if c < 1 or len(tokens) < c + 1:
            return {"eligible": False, "reason": f"row {slot}: {len(tokens)} positions known, {c + 1} needed to rebuild pass {T}"}
        if r.get("argmax") is None:
            return {"eligible": False, "reason": f"row {slot} at pass {T} lacks an argmax"}
        if tokens[c] != (r.get("new_token_ids") or [None])[0]:
            return {"eligible": False, "reason": f"row {slot}: consumed id at pass {T} disagrees with the token sequence"}
        rows.append({"slot": slot, "prefix": tokens[:c], "forced": [tokens[c], int(r["argmax"])], "computed": c, "kv": int(r["kv"])})
    total = sum(len(x["prefix"]) for x in rows)
    budget, seqs = limits.get("max_num_batched_tokens"), limits.get("max_num_seqs")
    if budget is None or seqs is None:
        return {"eligible": False, "reason": "recorded run.json lacks resolved_scheduler limits (max_num_batched_tokens, max_num_seqs)"}
    if total > int(budget):
        return {"eligible": False, "reason": f"the {len(rows)} prefixes total {total} tokens, above max_num_batched_tokens {budget}; "
                                             "they could not be prefilled in one pass, so pass T cannot be rebuilt with the stock scheduler"}
    if len(rows) > int(seqs):
        return {"eligible": False, "reason": f"{len(rows)} rows exceed max_num_seqs {seqs}"}
    return {"eligible": True, "pass": T, "step": steps[T].get("step"), "rows": rows, "prefix_tokens_total": total, "limits": limits,
            "reason": None}


def compare_boundary(recorded: list[dict], T: int, rebuilt: list[dict]) -> dict:
    """Compare recorded pass T with the rebuilt run's uniform-decode pass (its second forward pass)."""
    errors = [f"recorded: {e}" for e in record_errors(recorded)] + [f"rebuilt: {e}" for e in record_errors(rebuilt)]
    result = {"pass": T, "validation_errors": errors, "first_mismatch": None, "rows_compared": 0, "rows_differing": 0,
              "differing_slots": [], "excluded_row_fields": ["prompt", "cache_hit"],
              "rebuilt_passes": len(rebuilt), "verdict_P2_boundary": "INVALID"}
    if errors:
        return result
    if not 0 <= T < len(recorded):
        result["validation_errors"].append(f"pass {T} outside the recorded trace")
        return result
    target = recorded[T]
    if len(rebuilt) < 2:
        result["first_mismatch"] = {"pass": 1, "slot": None, "field": "trace_length", "recorded": 2, "replayed": len(rebuilt)}
        result["verdict_P2_boundary"] = "NOT COMPARABLE"
        return result
    prefill, decode = rebuilt[0], rebuilt[1]
    mismatch = None
    slots_recorded = [slot_of(r["req"]) for r in target["requests"]]
    for name, s in (("prefill", prefill), ("decode", decode)):
        got = [slot_of(r["req"]) for r in s["requests"]]
        if got != slots_recorded:
            mismatch = {"pass": 0 if name == "prefill" else 1, "slot": None, "field": "request_order", "recorded": slots_recorded, "replayed": got}
            break
    if mismatch is None:
        for ra, rb in zip(target["requests"], prefill["requests"]):
            if rb.get("computed") != 0 or rb.get("q") != ra.get("computed") or rb.get("phase") != "prefill":
                mismatch = {"pass": 0, "slot": slot_of(ra["req"]), "field": "rebuilt_prefill_shape",
                            "recorded": {"computed": ra.get("computed")}, "replayed": {"computed": rb.get("computed"), "q": rb.get("q"), "phase": rb.get("phase")}}
                break
    if mismatch is None:
        for key in [k for k in RECORD_FIELDS if k not in ID_FIELDS]:
            if target.get(key) != decode.get(key):
                mismatch = {"pass": 1, "slot": None, "field": key, "recorded": target.get(key), "replayed": decode.get(key)}
                break
    if mismatch is None:
        for ra, rb in zip(target["requests"], decode["requests"]):
            for key in (*BOUNDARY_ROW_FIELDS, "new_token_ids"):
                if ra.get(key) != rb.get(key):
                    mismatch = {"pass": 1, "slot": slot_of(ra["req"]), "field": key, "recorded": ra.get(key), "replayed": rb.get(key)}
                    break
            if mismatch:
                break
    if mismatch:
        result["first_mismatch"] = mismatch
        result["verdict_P2_boundary"] = "NOT COMPARABLE"
        return result
    for ra, rb in zip(target["requests"], decode["requests"]):
        result["rows_compared"] += 1
        if any(ra.get(k) != rb.get(k) for k in HASH_FIELDS):
            result["rows_differing"] += 1
            result["differing_slots"].append(slot_of(ra["req"]))
    result["verdict_P2_boundary"] = "IDENTICAL" if result["rows_differing"] == 0 else "DIFFERS"
    return result


# ----------------------------------------------------------------------------
# Offline comparison entry point (also used by reanalyse.py)
# ----------------------------------------------------------------------------

def _digests(paths: list[Path]) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _record_files(d: Path) -> list[Path]:
    return [d / "run.json", d / "outputs.json", d / "boundary.json", *sorted((d / "hook").glob("rank*.jsonl")), *sorted((d / "hook").glob("forcing_rank*.jsonl"))]


def _summary_path(record_dir: Path, replay_dir: Path, boundary: dict | None) -> Path:
    name = f"boundary_{boundary['pass']}.json" if boundary else "summary.json"
    return (replay_dir.parent / name) if replay_dir.parent.resolve() == record_dir.parent.resolve() else (replay_dir / name)


def _load_json(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def compare(record_dir: Path, replay_dir: Path) -> int:
    record_dir, replay_dir = Path(record_dir), Path(replay_dir)
    rec_meta = _load_json(record_dir / "run.json", {})
    rep_meta = _load_json(replay_dir / "run.json", {})
    boundary = _load_json(replay_dir / "boundary.json")
    try:
        recorded = load_steps(record_dir)
    except ValueError as e:
        recorded, load_error = [], f"recorded: {e}"
    else:
        load_error = None
    try:
        replayed = load_steps(replay_dir)
    except ValueError as e:
        replayed, load_error = [], (load_error + "; " if load_error else "") + f"replayed: {e}"
    provenance = []
    if rep_meta.get("recorded_run_id") != rec_meta.get("run_id") or not rec_meta.get("run_id"):
        provenance.append("replay run.json does not reference the recorded run id")
    if rep_meta.get("status") != "configured":
        provenance.append(f"replay run.json status is {rep_meta.get('status')!r}, not 'configured'")
    if rec_meta.get("resolved_model_runner") != rep_meta.get("resolved_model_runner"):
        provenance.append("recorded and replayed arms resolved different model runners")
    if load_error:
        provenance.append(load_error)
    forcing = forcing_consistency(replayed, read_forcing_log(replay_dir / "hook"), expect_forcing=True) if replayed else {"errors": [], "calls": 0, "forced_rows": 0, "rows_where_forced_equalled_argmax": 0}
    common = Path(os.path.commonpath([record_dir.resolve(), replay_dir.resolve()]))
    arm = record_dir.parent.name if record_dir.name == "record" else record_dir.name
    if boundary is not None:
        comparison = compare_boundary(recorded, int(boundary["pass"]), rebuilt=replayed) if not load_error else {"validation_errors": [], "verdict_P2_boundary": "INVALID", "rows_compared": 0, "rows_differing": 0, "first_mismatch": None, "differing_slots": []}
        errors = comparison["validation_errors"] + provenance + forcing["errors"]
        verdict = "INVALID" if errors else comparison["verdict_P2_boundary"]
        summary = {"experiment": "E6", "mode": "boundary", "arm": arm, "recorded_arm": str(record_dir), "replay_arm": str(replay_dir),
                   "pass": boundary["pass"], "recorded_step": boundary.get("step"), "rows": len(boundary.get("rows", [])),
                   "prefix_tokens_total": boundary.get("prefix_tokens_total"), **{k: v for k, v in comparison.items() if k != "validation_errors"},
                   "verdict_P2_boundary": verdict, "forcing_log": {k: v for k, v in forcing.items() if k != "errors"},
                   "validation_errors": sorted(set(errors)),
                   "claim_scope": "One recorded all-decode pass rebuilt by prefilling every row's recorded prefix in a single pass; the KV state "
                                  "comes from that prefill, not from the recorded sequence of passes, so DIFFERS does not separate a "
                                  "prefill-versus-decode KV difference from other causes. The rebuilt rows' prompt field is the prefix length and is excluded."}
    else:
        outputs_a = normalise_outputs(_load_json(record_dir / "outputs.json", {}))
        outputs_b = normalise_outputs(_load_json(replay_dir / "outputs.json", {}))
        plan = replay_plan(recorded, rec_meta, outputs_a) if recorded else {"errors": ["recorded trace empty or unreadable"], "requests": {}, "notes": []}
        comparison = compare_replay(recorded, replayed)
        errors = comparison["validation_errors"] + plan["errors"] + provenance + forcing["errors"]
        if recorded:
            errors += [f"recorded: {e}" for e in output_trace_errors(recorded, outputs_a)]
        if replayed and not plan["errors"]:
            errors += [f"replayed: {e}" for e in replay_output_errors(replayed, outputs_b, {s: v["forced"] for s, v in plan["requests"].items()})]
        verdict = "INVALID" if errors else comparison["verdict_P2"]
        summary = {"experiment": "E6", "mode": "replay", "arm": arm, "recorded_arm": str(record_dir), "replay_arm": str(replay_dir),
                   **{k: v for k, v in comparison.items() if k not in ("validation_errors", "verdict_P2")},
                   "requirements_met": comparison["first_mismatch"] is None and not errors,
                   "outputs_identical": bool(outputs_a) and outputs_a == outputs_b,
                   "forcing_log": {k: v for k, v in forcing.items() if k != "errors"}, "notes": plan.get("notes", []),
                   "verdict_P2": verdict, "validation_errors": sorted(set(errors)),
                   "claim_scope": "One recorded schedule replayed in one fresh process on the same stack with teacher forcing; identical means the "
                                  "recorded passes were reproduced from the record, not that every schedule or stack would be."}
    summary["record_sha256"] = _digests(_record_files(record_dir))
    summary["replay_sha256"] = _digests(_record_files(replay_dir))
    summary.update(analysis_metadata(common, [*_record_files(record_dir), *_record_files(replay_dir)]))
    write_json(_summary_path(record_dir, replay_dir, boundary), summary)
    label = f"boundary pass {boundary['pass']}" if boundary else "replay"
    print(f"E6 {arm} {label}: P2 {verdict}; first mismatch {summary.get('first_mismatch')}; rows compared {summary.get('rows_compared')}; "
          f"rows differing {summary.get('rows_differing')}; errors {len(summary['validation_errors'])}")
    return 1 if errors else 0


# ----------------------------------------------------------------------------
# GPU drivers (import vLLM lazily)
# ----------------------------------------------------------------------------

def _engine(args, mixed: bool):
    from vllm import LLM
    kw = engine_kwargs(args)
    kw["logits_processors"] = [TeacherForcingLogitsProcessor]
    kw["async_scheduling"] = False  # the V1 runner keeps sampled ids on the host only when scheduling is synchronous
    if mixed:
        kw["max_num_batched_tokens"] = 128
        kw["enable_chunked_prefill"] = True
    if getattr(args, "fp8_per_tensor", False):
        os.environ["SHAPE_FORCE_FP8_PER_TENSOR"] = "1"
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    if bool(cfg.use_v2_model_runner):
        raise RuntimeError("E6 requires the V1 model runner; custom logits processors are unsupported on the V2 runner")
    if bool(cfg.scheduler_config.async_scheduling):
        raise RuntimeError("E6 requires async scheduling off")
    return llm


def _e6_extra(llm, mode: str, **more) -> dict:
    cfg = llm.llm_engine.vllm_config
    sc = cfg.scheduler_config
    return {"experiment": "E6", "mode": mode, "resolved_model_runner": "v2" if cfg.use_v2_model_runner else "v1",
            "resolved_scheduler": {"max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens,
                                   "enable_chunked_prefill": sc.enable_chunked_prefill, "async_scheduling": sc.async_scheduling},
            "sampling": {"temperature": 0.0, "logprobs": 5, "ignore_eos": True}, "logits_processors": ["teacher_forcing:TeacherForcingLogitsProcessor"],
            **more}


def _count_forward_passes(llm) -> dict:
    """Wrap the in-process scheduler so the driver knows how many forward passes ran (core_client.py:306-320)."""
    counter = {"forward": 0, "calls": 0}
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    orig = scheduler.schedule

    def schedule(*a, **kw):
        out = orig(*a, **kw)
        counter["calls"] += 1
        if int(out.total_num_scheduled_tokens) > 0:
            counter["forward"] += 1
        return out

    scheduler.schedule = schedule
    return counter


def record(args) -> int:
    args.prefix_caching = 0
    out = args.out / (arm_name(args) + ("_mixed" if args.mixed else "")) / "record"
    out.mkdir(parents=True, exist_ok=False)
    env_with_hook(out / "hook")
    import shape_hook
    shape_hook.register()
    from vllm import SamplingParams
    llm = _engine(args, args.mixed)
    counter = _count_forward_passes(llm)
    eng = llm.llm_engine
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5, ignore_eos=True)
    prompts = mixed_prompts()
    if args.mixed:
        long = " ".join(["The verifier records the batch shape at every step and replays it later."] * 40)
        prompts = prompts[:8] + [long + f" Prompt {i}." for i in range(8)]
    record_run(out, args, llm, prompts, extra=_e6_extra(llm, "record", arrival_script="add 0-7; five scheduler steps; add 8-15; run to completion"))
    finished = []
    for i in range(8):
        eng.add_request(str(i), prompts[i], sp)
    for _ in range(5):
        finished.extend(o for o in eng.step() if o.finished)
    for i in range(8, 16):
        eng.add_request(str(i), prompts[i], sp)
    while eng.has_unfinished_requests():
        finished.extend(o for o in eng.step() if o.finished)
    shape_hook.flush()
    write_json(out / "outputs.json", outputs_record(finished))
    steps = real_steps(read_hook(out / "hook"))
    plan = replay_plan(steps, json.loads((out / "run.json").read_text()), normalise_outputs(json.loads((out / "outputs.json").read_text())))
    write_json(out / "schedule.json", {"forward_passes_at_scheduler": counter["forward"], "passes": plan["passes"], "errors": plan["errors"], "notes": plan["notes"]})
    status = "replayable" if not plan["errors"] else "NOT replayable: " + "; ".join(plan["errors"][:5])
    print(f"E6 {out.parent.name} record: {len(steps)} forward passes logged ({counter['forward']} at the scheduler); {status}")
    return 1 if plan["errors"] or counter["forward"] != len(steps) else 0


def _step_until(eng, counter: dict, target: int, finished: list, label: str) -> None:
    while counter["forward"] < target:
        if not eng.has_unfinished_requests():
            raise RuntimeError(f"engine idle before {label}: {counter['forward']} forward passes ran")
        finished.extend(o for o in eng.step() if o.finished)


def replay(record_dir: Path, out: Path, boundary: int | None) -> int:
    record_dir = Path(record_dir)
    meta = json.loads((record_dir / "run.json").read_text())
    recorded = load_steps(record_dir)
    outputs = normalise_outputs(_load_json(record_dir / "outputs.json", {})) or None
    args = NS(**meta["args"])
    args.out, args.recorded_from = str(out), str(record_dir)
    mixed = bool(meta["args"].get("mixed"))
    if boundary is None:
        plan = replay_plan(recorded, meta, outputs)
        if plan["errors"]:
            print("E6 replay refused; the record cannot drive a replay:\n  " + "\n  ".join(plan["errors"]))
            return 2
    else:
        plan = boundary_plan(recorded, boundary, meta.get("resolved_scheduler") or {})
        if not plan["eligible"]:
            print(f"E6 boundary replay refused for pass {boundary}: {plan['reason']}")
            return 2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    env_with_hook(out / "hook")
    import shape_hook
    shape_hook.register()
    from vllm import SamplingParams, TokensPrompt
    llm = _engine(args, mixed)
    counter = _count_forward_passes(llm)
    eng = llm.llm_engine
    extra = _e6_extra(llm, "replay" if boundary is None else "boundary", recorded_arm=str(record_dir), recorded_run_id=meta.get("run_id"),
                      record_sha256=_digests(_record_files(record_dir)))
    finished = []
    if boundary is None:
        slots = sorted(plan["requests"], key=lambda s: int(s) if s.isdigit() else s)
        record_run(out, args, llm, [plan["requests"][s]["prompt_token_ids"] for s in slots], extra=extra)
        for p in plan["passes"]:
            for slot in p["admitted"]:
                req = plan["requests"][slot]
                sp = SamplingParams(temperature=0.0, max_tokens=int(plan["max_tokens"]), logprobs=5, ignore_eos=True, extra_args={EXTRA_ARGS_KEY: list(req["forced"])})
                eng.add_request(slot, TokensPrompt(prompt_token_ids=list(req["prompt_token_ids"])), sp)
            _step_until(eng, counter, p["index"] + 1, finished, f"recorded pass {p['index']}")
        while eng.has_unfinished_requests():
            finished.extend(o for o in eng.step() if o.finished)
    else:
        record_run(out, args, llm, [r["prefix"] for r in plan["rows"]], extra={**extra, "boundary_pass": boundary})
        write_json(out / "boundary.json", plan)
        for r in plan["rows"]:
            sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=5, ignore_eos=True, extra_args={EXTRA_ARGS_KEY: list(r["forced"])})
            eng.add_request(r["slot"], TokensPrompt(prompt_token_ids=list(r["prefix"])), sp)
        while eng.has_unfinished_requests():
            finished.extend(o for o in eng.step() if o.finished)
    shape_hook.flush()
    write_json(out / "outputs.json", outputs_record(finished))
    steps = real_steps(read_hook(out / "hook"))
    print(f"E6 {'boundary' if boundary is not None else 'replay'} {out}: {len(steps)} forward passes logged ({counter['forward']} at the scheduler), "
          f"recorded {len(recorded)}; run --compare for the verdict")
    return 0 if counter["forward"] == len(steps) else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--compare" in argv:
        i = argv.index("--compare")
        return compare(Path(argv[i + 1]), Path(argv[i + 2]))
    if "--replay" in argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--replay", type=Path, required=True, help="recorded arm directory (the one holding run.json and hook/)")
        ap.add_argument("--out", type=Path, required=True, help="new directory for the replay arm")
        ap.add_argument("--boundary", type=int, default=None, help="rebuild only this recorded forward-pass index")
        a = ap.parse_args(argv)
        return replay(a.replay, a.out, a.boundary)
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--record", action="store_true", required=True)
    ap.add_argument("--mixed", action="store_true", help="chunked prefill of long second-wave prompts so mid-stream passes mix prefill and decode")
    return record(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
