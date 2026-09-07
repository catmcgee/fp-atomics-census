"""E3, logged replay: a staged trajectory is logged, then re-run in a fresh process and compared step by step.

    RUN_TAG=a python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --out probes/shape/results/e3
    RUN_TAG=b python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --out probes/shape/results/e3
    python probes/shape/run_e3.py --compare probes/shape/results/e3/<arm>

The engine core runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the
scheduler can be forced to one composition: eight requests are added, five
steps run, eight more are added, and the rest runs to completion. Step 0 is
a pure prefill, step 5 mixes eight decodes with eight prefills, later steps
are pure decode. The scheduler, not the model-runner entry point, is used,
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

from shape_common import real_steps, add_common_args, arm_name, engine_kwargs, env_with_hook, environment, mixed_prompts, outputs_record, read_hook, slot_of, write_json


def run(args) -> int:
    tag = os.environ.get("RUN_TAG", "a")
    out = args.out / arm_name(args) / tag
    env_with_hook(out / "hook")
    import shape_hook
    shape_hook.register()
    from vllm import LLM, SamplingParams

    llm = LLM(**engine_kwargs(args))
    eng = llm.llm_engine
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5, ignore_eos=True)
    prompts = mixed_prompts()
    finished = []
    for i in range(8):
        eng.add_request(str(i), prompts[i], sp)
    for _ in range(5):
        finished.extend(o for o in eng.step() if o.finished)
    for i in range(8, 16):
        eng.add_request(str(i), prompts[i], sp)
    while eng.has_unfinished_requests():
        finished.extend(o for o in eng.step() if o.finished)
    steps = real_steps(read_hook(out / "hook"))
    write_json(out / "outputs.json", outputs_record(finished))
    write_json(out / "steps.json", [{"step": s["step"], "shape_vector": s.get("shape_vector"), "requests": s.get("requests")} for s in steps if "step" in s])
    write_json(out / "env.json", environment())
    print(f"E3 {arm_name(args)} tag {tag}: {len(steps)} steps logged")
    return 0


def compare(arm_dir: Path) -> int:
    a = json.loads((arm_dir / "a" / "steps.json").read_text())
    b = json.loads((arm_dir / "b" / "steps.json").read_text())
    oa = json.loads((arm_dir / "a" / "outputs.json").read_text())
    ob = json.loads((arm_dir / "b" / "outputs.json").read_text())
    rows = []
    oa = {slot_of(k): v for k, v in oa.items()}
    ob = {slot_of(k): v for k, v in ob.items()}
    for sa, sb in zip(a, b):
        same_shape = sa["shape_vector"] == sb["shape_vector"]
        ha = {slot_of(r["req"]): (r["h"], r["argmax"]) for r in sa["requests"]}
        hb = {slot_of(r["req"]): (r["h"], r["argmax"]) for r in sb["requests"]}
        same_hash = ha == hb
        rows.append({"step": sa["step"], "shape_identical": same_shape, "hashes_identical": same_hash, "num_reqs": len(ha),
                     "phase_mix": sorted({r["phase"] for r in sa["requests"]})})
    selected = [r for r in rows if r["step"] in (0, 5, max(x["step"] for x in rows) // 2)]
    verdict = "IDENTICAL" if all(r["hashes_identical"] and r["shape_identical"] for r in rows) and oa == ob else "DIFFERS"
    summary = {"experiment": "E3", "arm": arm_dir.name, "steps_compared": len(rows), "all_steps_identical": all(r["hashes_identical"] for r in rows),
               "outputs_identical": oa == ob, "selected_steps": selected, "differing_steps": [r["step"] for r in rows if not r["hashes_identical"]][:20], "verdict_P2": verdict}
    write_json(arm_dir / "summary.json", summary)
    print(f"E3 {arm_dir.name}: P2 {verdict}; {len(rows)} steps; selected {selected}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--compare", type=Path, default=None)
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        return compare(Path(sys.argv[i + 1]))
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
