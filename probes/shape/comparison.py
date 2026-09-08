"""Fail-closed offline comparisons for staged trajectories and recorded stacks."""
from __future__ import annotations

import json
from pathlib import Path

from shape_common import read_hook, real_steps, recorded_shape, slot_of, trace_errors


def load_steps(run: Path) -> list[dict]:
    raw = read_hook(run / "hook")
    if any("error" in s for s in raw):
        raise ValueError(f"hook recorded instrumentation errors in {run}")
    return real_steps(raw if raw else json.loads((run / "steps.json").read_text()))


def normalise_outputs(outputs: dict) -> dict:
    result = {slot_of(k): v for k, v in outputs.items()}
    if len(result) != len(outputs):
        raise ValueError("duplicate normalised output request IDs")
    return result


def compare_traces(a: list[dict], b: list[dict]) -> dict:
    errors = [f"a: {e}" for e in trace_errors(a)] + [f"b: {e}" for e in trace_errors(b)]
    if len(a) != len(b):
        errors.append(f"trace length mismatch: {len(a)} != {len(b)}")
    rows = []
    # Pairing is diagnostic only: any missing tail makes the whole comparison invalid.
    for i, (sa, sb) in enumerate(zip(a, b)):
        ha = {slot_of(r["req"]): (r.get("h"), r.get("argmax"), r.get("logits_h")) for r in sa["requests"]}
        hb = {slot_of(r["req"]): (r.get("h"), r.get("argmax"), r.get("logits_h")) for r in sb["requests"]}
        if set(ha) != set(hb):
            errors.append(f"request set mismatch at offset {i}")
        if sa["step"] - a[0]["step"] != sb["step"] - b[0]["step"]:
            errors.append(f"step alignment mismatch at offset {i}")
        rows.append({"offset": i, "step": sa["step"], "shape_identical": recorded_shape(sa) == recorded_shape(sb),
                     "hashes_identical": ha == hb, "hidden_identical": {k: v[0] for k, v in ha.items()} == {k: v[0] for k, v in hb.items()},
                     "argmax_identical": {k: v[1] for k, v in ha.items()} == {k: v[1] for k, v in hb.items()},
                     "num_reqs": len(ha), "phase_mix": sorted({r["phase"] for r in sa["requests"]})})
    shapes_equal = bool(rows) and all(r["shape_identical"] for r in rows)
    verdict = "INVALID" if errors else ("NOT COMPARABLE" if not shapes_equal else
                                         "IDENTICAL" if all(r["hashes_identical"] for r in rows) else "DIFFERS")
    return {"rows": rows, "validation_errors": errors, "verdict": verdict, "shape_histories_equal": shapes_equal}


def output_trace_errors(steps: list[dict], outputs: dict) -> list[str]:
    """Check terminal state, so two equally truncated traces cannot pass."""
    by_request = {}
    for step in steps:
        for r in step["requests"]:
            by_request.setdefault(slot_of(r["req"]), []).append(r)
    errors = []
    if set(outputs) != set(by_request):
        errors.append("hook/output request set mismatch")
    for rid, output in outputs.items():
        observations = by_request.get(rid, [])
        tokens = output.get("tokens", [])
        if not output.get("hash") or not tokens or len(observations) < len(tokens):
            errors.append(f"incomplete output or trace for request {rid}")
            continue
        last = observations[-1]
        if last.get("prompt") is not None and last.get("kv") != last["prompt"] + len(tokens) - 1:
            errors.append(f"terminal token state mismatch for request {rid}")
        if last.get("argmax") != tokens[-1]:
            errors.append(f"terminal greedy token disagrees for request {rid}")
    return errors
