"""E2, bucket attribution: stock repeats with the shape hook on, hashes joined to shape histories.

    python probes/shape/run_e2.py --model Qwen/Qwen2.5-7B-Instruct --cudagraph 1 --prefix-caching 1 --repeats 12 --out probes/shape/results/e2

Runs the engine as a stock deployment (multiprocess engine core; the hook is
loaded there as a vLLM plugin), generates the mixed request set ``repeats``
times, and joins every request's per-step hidden-state hash to the history
of shape vectors of the steps it took part in. P1 is falsified by any
(request slot, step index, shape history) that maps to more than one hash.
The per-request output hash is also joined to the request's full history.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from shape_common import analysis_metadata, saved_env, trace_errors, record_run, real_steps, add_common_args, arm_name, engine_kwargs, env_with_hook, environment, history_key, mixed_prompts, outputs_record, read_hook, slot_of, steps_by_request, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--analyse", type=Path, default=None, help="recompute summary.json for a saved arm directory")
    if "--analyse" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--analyse") + 1])
        rep = len(json.loads((d / "outputs.json").read_text()))
        return analyse(d, d.name, rep)
    args = ap.parse_args()
    out = args.out / arm_name(args)
    env_with_hook(out / "hook")
    if args.fp8_per_tensor:
        os.environ["SHAPE_FORCE_FP8_PER_TENSOR"] = "1"
    from vllm import LLM, SamplingParams

    llm = LLM(**engine_kwargs(args))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5, ignore_eos=True)
    prompts = mixed_prompts()
    record_run(out, args, llm, prompts)
    per_repeat = []
    for k in range(args.repeats):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        rec = outputs_record(outs)
        per_repeat.append({"repeat": k, "request_ids": [o.request_id for o in outs], "outputs": rec})
        print(f"repeat {k}: output hashes {[rec[o.request_id]['hash'][:8] for o in outs][:6]}...")
    write_json(out / "outputs.json", per_repeat)
    return analyse(out, arm_name(args), args.repeats)


def analyse(out: Path, arm: str, repeats: int) -> int:
    """Join hashes to shape histories; usable offline on a saved arm directory."""
    per_repeat = json.loads((out / "outputs.json").read_text())
    steps = real_steps(read_hook(out / "hook"))
    by_req = steps_by_request(steps)  # keyed by request slot (counter prefix of the engine's id)
    table: dict[tuple[int, int], dict[str, set]] = defaultdict(lambda: defaultdict(set))
    final: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    joined = 0
    errors = trace_errors(steps)
    if any("error" in s for s in read_hook(out / "hook")):
        errors.append("hook recorded instrumentation errors")
    observations = defaultdict(int)
    expected_ids = []
    if len(per_repeat) != repeats or repeats < 2:
        errors.append("repeat count mismatch or fewer than two repeats")
    for rep in per_repeat:
        if set(rep["request_ids"]) != set(rep["outputs"]):
            errors.append("output/request set mismatch")
        expected_ids.extend(slot_of(rid) for rid in rep["request_ids"])
        for slot, rid in enumerate(rep["request_ids"]):
            seq = by_req.get(slot_of(rid), [])
            joined += bool(seq)
            output = rep["outputs"].get(rid, {})
            if not seq or len(seq) != len(output.get("tokens", [])) or not output.get("hash"):
                errors.append(f"incomplete request {rid}")
            for j in range(len(seq)):
                hk = history_key(seq, j)
                if seq[j][2] is not None:
                    table[(slot, j)][hk].add(seq[j][2])
                observations[(slot, j, hk)] += 1
            if seq and output.get("hash"):
                final[slot][history_key(seq, len(seq) - 1)].add(rep["outputs"][rid]["hash"])
    if len(set(expected_ids)) != len(expected_ids):
        errors.append("duplicate request IDs across repeats")
    if set(expected_ids) != set(by_req):
        errors.append("hook/output request set mismatch")
    violations = [{"slot": s, "step": j, "history": hk, "hashes": sorted(hs)} for (s, j), m in table.items() for hk, hs in m.items() if len(hs) > 1]
    final_violations = [{"slot": s, "history": hk, "output_hashes": sorted(hs)} for s, m in final.items() for hk, hs in m.items() if len(hs) > 1]
    n_hist = {f"{s}/{j}": len(m) for (s, j), m in table.items()}
    n_hash = {f"{s}/{j}": len({h for hs in m.values() for h in hs}) for (s, j), m in table.items()}
    env = saved_env(out)
    summary = {"experiment": "E2", "arm": arm, "env": env, "repeats": repeats, "requests_joined_to_hook": joined,
               "hook_steps": len(steps), "distinct_step_shape_vectors": len({s.get("shape_vector") for s in steps if "shape_vector" in s}),
               "distinct_output_hashes_per_slot": {s: len({h for hs in m.values() for h in hs}) for s, m in final.items()},
               "distinct_histories_per_slot": {s: len(m) for s, m in final.items()},
               "history_to_multiple_hashes": violations, "final_history_to_multiple_outputs": final_violations,
               "verdict_P1": "DIFFERS" if (violations or final_violations) else "IDENTICAL",
               "per_step_counts": {"histories": n_hist, "hashes": n_hash}}
    summary.update(analysis_metadata(out, [out / "outputs.json", *sorted((out / "hook").glob("rank*.jsonl"))]))
    summary["validation_errors"] = sorted(set(errors))
    summary["repeated_history_groups"] = sum(n > 1 for n in observations.values())
    summary["singleton_history_groups"] = sum(n == 1 for n in observations.values())
    summary["scheduler_calls_excluded"] = len(read_hook(out / "hook")) - len(steps)
    if errors:
        summary["verdict_P1"] = "INVALID"
    elif not summary["repeated_history_groups"]:
        summary["verdict_P1"] = "NOT EVALUATED"
    summary["claim_scope"] = "Observed repeatability conditional on recorded histories; singleton groups provide no repeatability test."
    write_json(out / "summary.json", summary)
    print(f"E2 {arm}: P1 {summary['verdict_P1']}; steps {len(steps)}; joined {joined}; distinct shape vectors {summary['distinct_step_shape_vectors']}; violations {len(violations)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
