"""E4, dummy-neighbour replay: the target request keeps its tokens, every neighbour becomes random tokens of the same lengths.

    python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --repeats 12 --out probes/shape/results/e4 [--quantization fp8] [--fp8-per-tensor] [--tp 2] [--cudagraph 0]

In-process engine core, synchronous arrival, ``ignore_eos`` and a fixed
``max_tokens`` for every request, so the shape vector of every step is the
same in the original and the dummy batch by construction; the hook verifies
this. Prefix caching is off in this experiment so that computed-token
counts cannot differ between original and dummy neighbours. Twelve
original and twelve dummy runs alternate. Reported per arm: whether the
target's per-step hidden hashes, argmax logits, sampled tokens and top
logprobs are identical across all runs of each kind and between the two
kinds, and, for MoE models, whether the first layer's per-expert counts
changed between kinds.
"""
from __future__ import annotations

import argparse
import json
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from shape_common import real_steps, add_common_args, arm_name, engine_kwargs, env_with_hook, environment, mixed_prompts, outputs_record, random_token_prompt, read_hook, slot_of, steps_by_request, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--analyse", type=Path, default=None, help="recompute summary.json for a saved arm directory")
    if "--analyse" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--analyse") + 1])
        runs = json.loads((d / "runs.json").read_text())
        return analyse(d, d.name, max(r["repeat"] for r in runs) + 1, 0)
    args = ap.parse_args()
    args.prefix_caching = 0
    out = args.out / arm_name(args)
    env_with_hook(out / "hook")
    if args.fp8_per_tensor:
        os.environ["SHAPE_FORCE_FP8_PER_TENSOR"] = "1"
    import shape_hook
    shape_hook.register()
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(**engine_kwargs(args))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5, ignore_eos=True)
    prompts = mixed_prompts()
    ids = [tok.encode(p) for p in prompts]
    original = [TokensPrompt(prompt_token_ids=i) for i in ids]
    runs = []
    for k in range(args.repeats):
        for kind in ("original", "dummy"):
            if kind == "original":
                batch = original
            else:
                batch = [TokensPrompt(prompt_token_ids=(ids[i] if i == args.target else random_token_prompt(tok, len(ids[i]), seed=1000 * k + i))) for i in range(len(ids))]
            outs = llm.generate(batch, sp, use_tqdm=False)
            rec = outputs_record(outs)
            target_rid = outs[args.target].request_id
            runs.append({"repeat": k, "kind": kind, "target_rid": target_rid, "target": rec[target_rid], "request_ids": [o.request_id for o in outs]})
            print(f"repeat {k} {kind}: target output hash {rec[target_rid]['hash']}")
    write_json(out / "runs.json", runs)
    return analyse(out, arm_name(args), args.repeats, args.target)


def analyse(out: Path, arm: str, repeats: int, target: int) -> int:
    runs = json.loads((out / "runs.json").read_text())
    steps = real_steps(read_hook(out / "hook"))
    by_req = steps_by_request(steps)
    step_moe = {s["step"]: s.get("moe_expert_counts_first_layer") for s in steps if "step" in s}
    step_fp8 = {s["step"]: s.get("fp8_scale_first_call") for s in steps if "step" in s}
    for r in runs:
        seq = by_req.get(slot_of(r["target_rid"]), [])
        r["fp8_scales"] = [step_fp8.get(st) for st, _, _, _ in seq]
        r["target_hidden_hashes"] = [h for _, _, h, _ in seq]
        r["target_argmax"] = [a for _, _, _, a in seq]
        r["shape_history"] = [sv for _, sv, _, _ in seq]
        r["moe_counts_first_step"] = step_moe.get(seq[0][0]) if seq else None
    by_kind = {k: [r for r in runs if r["kind"] == k] for k in ("original", "dummy")}
    def all_same(items):
        return len({str(x) for x in items}) == 1
    within = {k: {"hidden": all_same([r["target_hidden_hashes"] for r in rs]), "output": all_same([r["target"]["hash"] for r in rs]), "shape": all_same([r["shape_history"] for r in rs])} for k, rs in by_kind.items()}
    shapes_equal = all_same([r["shape_history"] for r in runs])
    across = {"hidden": all_same([r["target_hidden_hashes"] for r in runs]), "argmax": all_same([r["target_argmax"] for r in runs]),
              "tokens": all_same([r["target"]["tokens"] for r in runs]), "logprobs": all_same([r["target"]["logprobs"] for r in runs])}
    moe_changed = None
    if any(r["moe_counts_first_step"] for r in runs):
        moe_changed = not all_same([r["moe_counts_first_step"] for r in runs])
    joined = sum(1 for r in runs if r["target_hidden_hashes"])
    fp8_changed = None
    if any(any(v is not None for v in r["fp8_scales"]) for r in runs):
        fp8_changed = not all_same([r["fp8_scales"] for r in runs])
    verdict = "IDENTICAL" if shapes_equal and all(across.values()) else "DIFFERS"
    if joined == 0:
        verdict += " (outputs only: no hook record joined)"
    env = json.loads((out / "summary.json").read_text()).get("env") if (out / "summary.json").exists() else None
    summary = {"experiment": "E4", "arm": arm, "env": env or environment(), "repeats": repeats, "target_slot": target, "runs_joined_to_hook": joined,
               "shape_histories_equal_across_all_runs": shapes_equal, "within_kind": within, "target_across_kinds": across,
               "moe_first_layer_expert_counts_changed": moe_changed, "fp8_first_scale_changed_between_kinds": fp8_changed, "verdict_P3": verdict,
               "first_divergent_step": next((i for i, (x, y) in enumerate(zip(by_kind["original"][0]["target_hidden_hashes"], by_kind["dummy"][0]["target_hidden_hashes"])) if x != y), None)}
    write_json(out / "summary.json", summary)
    print(f"E4 {arm}: P3 {verdict}; joined {joined}; shapes equal {shapes_equal}; across kinds {across}; moe counts changed {moe_changed}; fp8 scale changed {fp8_changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
