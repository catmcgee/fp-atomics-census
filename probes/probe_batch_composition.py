"""Observe batch composition directly and tie it to output differences.

    python probes/probe_batch_composition.py --model Qwen/Qwen2.5-7B-Instruct --repeats 12

Runs vLLM with the engine core in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
so the scheduler can be instrumented: every call to Scheduler.schedule is
logged as the multiset of tokens scheduled per request that step, and a
repeat's composition signature is the hash of that log.

Two modes are run and compared:

* ``sync``: all prompts are added before the first step, as LLM.generate
  does in-process. Arrival order cannot vary, so every repeat should carry
  the same composition signature; if the stock kernels are deterministic
  for a fixed composition, the outputs are identical too.
* ``stagger``: the last prompt is added only after the first scheduling
  step, which forces a different first batch. Repeats of this mode should
  be identical to each other and differ from ``sync`` by whole requests,
  the signature seen in the multiprocess runs.

The report records, per repeat, the composition signature and whether the
outputs matched the first repeat of the same mode, and the cross-mode
comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

from common import bitwise_equal, environment, stack_dir, tensor_hash  # noqa: E402

PROMPTS = [
    "The verifier re-runs a sampled computation and compares hashes.",
    "Floating-point addition is not associative, so",
    "def fibonacci(n):",
    "In 1854 the",
] * 4


def find_scheduler(llm):
    eng = llm.llm_engine
    for path in ("engine_core.engine_core.scheduler", "engine_core.scheduler", "scheduler"):
        obj = eng
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("scheduler not reachable in-process; attributes: " + ", ".join(dir(eng)))


def instrument(scheduler, log: list):
    orig = scheduler.schedule

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        try:
            toks = sorted(out.num_scheduled_tokens.values())
            log.append((len(out.scheduled_new_reqs), tuple(toks)))
        except Exception as e:  # noqa: BLE001
            log.append(("?", repr(e)[:60]))
        return out

    scheduler.schedule = wrapped


def collect(outs) -> list[torch.Tensor]:
    outs = sorted(outs, key=lambda o: int(o.request_id))
    tokens, vals = [], []
    for o in outs:
        tokens.extend(o.outputs[0].token_ids)
        for pos in o.outputs[0].logprobs:
            vals.extend(sorted(lp.logprob for lp in pos.values()))
    return [torch.tensor(tokens, dtype=torch.int64), torch.tensor(vals, dtype=torch.float32)]


def run_sync(llm, sp):
    return collect(llm.generate(PROMPTS, sp, use_tqdm=False))


def run_stagger(llm, sp, hold: int = 1):
    eng = llm.llm_engine
    n = len(PROMPTS)
    for i in range(n - hold):
        eng.add_request(str(i), PROMPTS[i], sp)
    finished = []
    finished.extend(o for o in eng.step() if o.finished)
    for i in range(n - hold, n):
        eng.add_request(str(i), PROMPTS[i], sp)
    while eng.has_unfinished_requests():
        finished.extend(o for o in eng.step() if o.finished)
    return collect(finished)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, seed=0, enable_prefix_caching=False, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5)
    sched = find_scheduler(llm)
    log: list = []
    instrument(sched, log)
    report = {"probe": args.name or f"batch_composition_{args.model.replace('/', '_')}", "env": environment(), "modes": {}}
    firsts = {}
    verdict = True
    for mode, fn in (("sync", run_sync), ("stagger", run_stagger)):
        entries = []
        for k in range(args.repeats):
            log.clear()
            torch.manual_seed(0)
            out = fn(llm, sp)
            torch.cuda.synchronize()
            sig = hashlib.sha256(json.dumps(log).encode()).hexdigest()[:16]
            first = firsts.setdefault(mode, ([t.clone() for t in out], sig))
            same_out, diffs = bitwise_equal(first[0], out)
            entries.append({"repeat": k, "composition": sig, "steps": len(log), "first_step": log[0] if log else None,
                            "output_matches_first": same_out, "diffs": diffs, "output_hash": tensor_hash(out)[:16]})
            verdict &= same_out
            print(f"{mode} repeat {k}: composition={sig} steps={len(log)} first_step_new_reqs={log[0][0] if log else '?'} output_matches_first={same_out}")
        report["modes"][mode] = entries
    cross_ok, cross_diffs = bitwise_equal(firsts["sync"][0], firsts["stagger"][0])
    report["sync_vs_stagger"] = {"identical": cross_ok, "diffs": cross_diffs,
                                 "sync_composition": firsts["sync"][1], "stagger_composition": firsts["stagger"][1]}
    report["verdict"] = "bitwise-identical" if verdict else "DIFFERS"
    out_dir = stack_dir(report["env"])
    tag = os.environ.get("RUN_TAG")
    fname = f"{report['probe']}.{tag}.json" if tag else f"{report['probe']}.json"
    (out_dir / fname).write_text(json.dumps(report, indent=2, default=str))
    print(f"PROBE {report['probe']} {report['verdict']} sync_vs_stagger_identical={cross_ok} diffs={cross_diffs} -> {out_dir / fname}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
