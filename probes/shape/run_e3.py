"""E3, scripted trajectory repeatability: a staged trajectory is logged, then re-run in a fresh process and compared step by step.

    RUN_TAG=a python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --out probes/shape/results/e3
    RUN_TAG=b python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --out probes/shape/results/e3
    python probes/shape/run_e3.py --compare probes/shape/results/e3/<arm>

The engine core runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the
scheduler can be forced to one composition: eight requests are added, five
steps run, eight more are added, and the rest runs to completion. The initial pass is a prefill; the second wave can produce mixed prefill
and decode. Chunking and the actual scheduler determine the recorded phases. The scheduler, not the model-runner entry point, is used,
because in-process synchronous arrival forces the composition exactly; the
hook confirms it by shape vector. ``--compare`` joins the two tags and
reports IDENTICAL or DIFFERS per selected step and for every step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from comparison import compare_traces, load_steps, normalise_outputs, output_trace_errors
from shape_common import analysis_metadata, record_run, real_steps, add_common_args, arm_name, engine_kwargs, env_with_hook, environment, mixed_prompts, outputs_record, read_hook, slot_of, write_json


def run(args) -> int:
    tag = os.environ.get("RUN_TAG", "a")
    out = args.out / (arm_name(args) + ("_mixed" if args.mixed else "")) / tag
    env_with_hook(out / "hook")
    import shape_hook
    shape_hook.register()
    from vllm import LLM, SamplingParams

    kw = engine_kwargs(args)
    if args.mixed:
        # chunked prefill: the second wave's long prompts are prefilled in 128-token
        # chunks over several steps while the first wave decodes, so those steps
        # mix decode rows with prefill chunks.
        kw["max_num_batched_tokens"] = 128
        kw["enable_chunked_prefill"] = True
    llm = LLM(**kw)
    eng = llm.llm_engine
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5, ignore_eos=True)
    prompts = mixed_prompts()
    if args.mixed:
        long = " ".join(["The verifier records the batch shape at every step and replays it later."] * 40)
        prompts = prompts[:8] + [long + f" Prompt {i}." for i in range(8)]
    record_run(out, args, llm, prompts)
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
    steps = real_steps(read_hook(out / "hook"))
    write_json(out / "outputs.json", outputs_record(finished))
    write_json(out / "steps.json", [{"step": s["step"], "shape_vector": s.get("shape_vector"), "requests": s.get("requests")} for s in steps if "step" in s])
    print(f"E3 {arm_name(args)}{'_mixed' if args.mixed else ''} tag {tag}: {len(steps)} steps logged")
    return 0


def compare(arm_dir: Path) -> int:
    a, b = load_steps(arm_dir / "a"), load_steps(arm_dir / "b")
    oa = normalise_outputs(json.loads((arm_dir / "a" / "outputs.json").read_text()))
    ob = normalise_outputs(json.loads((arm_dir / "b" / "outputs.json").read_text()))
    comparison = compare_traces(a, b)
    rows = comparison["rows"]
    errors = comparison["validation_errors"]
    for tag, steps, outputs in (("a", a, oa), ("b", b, ob)):
        errors.extend(f"{tag}: {error}" for error in output_trace_errors(steps, outputs))
    verdict = "INVALID" if errors else comparison["verdict"]
    if verdict == "IDENTICAL" and oa != ob:
        verdict = "DIFFERS"
    selected = [r for r in rows if r["offset"] in (0, 5, len(rows) // 2)]
    summary = {"experiment": "E3", "arm": arm_dir.name, "steps_compared": len(rows),
               "all_steps_identical": bool(rows) and all(r["hashes_identical"] for r in rows),
               "outputs_identical": oa == ob, "shape_histories_equal": comparison["shape_histories_equal"],
               "selected_steps": selected, "differing_steps": [r["step"] for r in rows if not r["hashes_identical"]],
               "verdict_repeatability": verdict, "verdict_P2": "NOT TESTED",
               "validation_errors": errors,
               "claim_scope": "Fresh-process reruns of the same arrival script; no recorded-schedule reconstruction, teacher forcing or midstream KV reconstruction."}
    summary.update(analysis_metadata(arm_dir, [p for tag in ("a", "b") for p in
                   [arm_dir / tag / "steps.json", arm_dir / tag / "outputs.json", *sorted((arm_dir / tag / "hook").glob("rank*.jsonl"))]]))
    write_json(arm_dir / "summary.json", summary)
    print(f"E3 {arm_dir.name}: scripted repeatability {verdict}; {len(rows)} steps; P2 NOT TESTED")
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--mixed", action="store_true", help="chunked prefill of long second-wave prompts so mid-stream steps mix prefill and decode")
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        return compare(Path(sys.argv[i + 1]))
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
