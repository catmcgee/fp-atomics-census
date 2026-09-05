"""Run an engine twice on the same prompts and compare logprobs bit for bit.

    python probes/probe_engine_logits.py --engine vllm --model Qwen/Qwen3-8B \
        [--quantization gptq_marlin] [--tp 1] [--extra-arg key=value ...]
    python probes/probe_engine_logits.py --engine sglang --model ... --extra-arg moe-runner-backend=flashinfer_cutlass

The engine is started in-process with greedy sampling and a fixed batch.
The probe writes the top-k logprobs of every position and compares runs.
It is the generic instrument for every default-path class-A row: run it
once with the configuration that reaches the row (see the inventory's
path.entry_points) and once with the exclusion-list configuration in the README.
"""
from __future__ import annotations

import argparse
import sys

import torch

from common import maybe_print_hash, run_twice

PROMPTS = [
    "The verifier re-runs a sampled computation and compares hashes.",
    "Floating-point addition is not associative, so",
    "def fibonacci(n):",
    "In 1854 the",
] * 4


def vllm_fn(args):
    from vllm import LLM, SamplingParams

    kw = {}
    for kv in args.extra_arg:
        k, v = kv.split("=", 1)
        kw[k.replace("-", "_")] = v
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, quantization=args.quantization, seed=0,
              enable_prefix_caching=False, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5, prompt_logprobs=5)

    def run():
        outs = llm.generate(PROMPTS, sp)
        vals = []
        for o in outs:
            for tok in o.outputs[0].logprobs:
                vals.extend(sorted(lp.logprob for lp in tok.values()))
        return [torch.tensor(vals, dtype=torch.float32)]
    return run


def sglang_fn(args):
    import sglang as sgl

    kw = {}
    for kv in args.extra_arg:
        k, v = kv.split("=", 1)
        kw[k.replace("-", "_")] = v
    engine = sgl.Engine(model_path=args.model, tp_size=args.tp, quantization=args.quantization, **kw)

    def run():
        outs = engine.generate(PROMPTS, {"temperature": 0, "max_new_tokens": args.max_tokens}, return_logprob=True, top_logprobs_num=5)
        vals = []
        for o in outs:
            for pos in o["meta_info"]["output_top_logprobs"]:
                vals.extend(sorted(lp[0] for lp in pos))
        return [torch.tensor(vals, dtype=torch.float32)]
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--print-hash", action="store_true")
    args = ap.parse_args()
    fn = vllm_fn(args) if args.engine == "vllm" else sglang_fn(args)
    name = f"{args.engine}_{args.model.replace('/', '_')}_{args.quantization or 'none'}_tp{args.tp}"
    ok = run_twice(name, fn)
    maybe_print_hash(fn())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
