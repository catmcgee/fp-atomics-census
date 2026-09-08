"""Run an engine several times on the same prompts and compare logprobs bit for bit.

    python probes/probe_engine_logits.py --engine vllm --model Qwen/Qwen3-8B
    python probes/probe_engine_logits.py --engine vllm --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
    python probes/probe_engine_logits.py --engine vllm --model Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --quantization moe_wna16
    python probes/probe_engine_logits.py --engine vllm --model HuggingFaceH4/zephyr-7b-beta --lora typeof/zephyr-7b-beta-lora
    python probes/probe_engine_logits.py --engine sglang --model Qwen/Qwen3-8B --extra-arg moe-runner-backend=flashinfer_cutlass

The engine is started in-process with greedy sampling and a fixed batch of
16 prompts. The probe collects the sampled token ids and the top-5 logprobs
of every generated position, runs the batch 1 + repeats times, and compares
runs bit for bit. It is the generic instrument for every default-path
class-A and A3 row: run it once with the configuration that reaches the row
(the inventory's path.entry_points and default_path_condition) and once
with the exclusion-list configuration in the README.
"""
from __future__ import annotations

import argparse
import sys
import hashlib
import json

import torch

from common import maybe_print_hash, run_twice, record_status
from observations import canonical_bytes, sparse_output

PROMPTS = [
    "The verifier re-runs a sampled computation and compares hashes.",
    "Floating-point addition is not associative, so",
    "def fibonacci(n):",
    "In 1854 the",
] * 4


def _kw(extra: list[str]) -> dict:
    kw = {}
    for kv in extra:
        k, v = kv.split("=", 1)
        if v.startswith("{") or v.startswith("["):
            import json
            v = json.loads(v)
        elif v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif v.isdigit():
            v = int(v)
        kw[k.replace("-", "_")] = v
    return kw


def vllm_fn(args):
    from vllm import LLM, SamplingParams

    kw = _kw(args.extra_arg)
    if args.revision:
        kw.update(revision=args.revision, tokenizer_revision=args.revision)
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        kw["enable_lora"] = True
        lora_request = LoRARequest("probe", 1, args.lora)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, quantization=args.quantization, seed=0,
              enable_prefix_caching=False, max_model_len=args.max_model_len, **kw)
    args.resolved_config = str(llm.llm_engine.vllm_config)
    args.resolved_model_revision = getattr(llm.llm_engine.vllm_config.model_config.hf_config, "_commit_hash", None)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5)

    def run():
        outs = llm.generate(PROMPTS, sp, lora_request=lora_request, use_tqdm=False)
        observations = [sparse_output(o.outputs[0].token_ids,
                                       [((t, lp.logprob) for t, lp in pos.items())
                                        for pos in o.outputs[0].logprobs]) for o in outs]
        return [torch.tensor(list(canonical_bytes(observations)), dtype=torch.uint8)]
    return run


def sglang_fn(args):
    import sglang as sgl

    kw = _kw(args.extra_arg)
    if args.revision:
        kw["revision"] = args.revision
    if args.lora:
        kw["lora_paths"] = [args.lora]
    kw.setdefault("disable_radix_cache", True)  # prefix caching changes the batch shape between runs; not a kernel property
    engine = sgl.Engine(model_path=args.model, tp_size=args.tp, quantization=args.quantization, random_seed=0, **kw)

    args.resolved_config = str(getattr(engine, "server_args", None))
    args.resolved_model_revision = None  # must not substitute the requested revision for an observed hash

    def run():
        params = {"temperature": 0, "max_new_tokens": args.max_tokens}
        outs = engine.generate(PROMPTS, params, return_logprob=True, top_logprobs_num=5)
        observations = [sparse_output([tok[1] for tok in o["meta_info"]["output_token_logprobs"]],
                                       [[(lp[1], lp[0]) for lp in pos] for pos in o["meta_info"]["output_top_logprobs"]]) for o in outs]
        return [torch.tensor(list(canonical_bytes(observations)), dtype=torch.uint8)]
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--lora", default=None, help="LoRA adapter path or HF id")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--name", default=None, help="override the report name")
    ap.add_argument("--print-hash", action="store_true")
    args = ap.parse_args()
    name = args.name or f"{args.engine}_{args.model.replace('/', '_')}_{args.quantization or 'none'}{'_lora' if args.lora else ''}_tp{args.tp}"
    try:
        fn = vllm_fn(args) if args.engine == "vllm" else sglang_fn(args)
    except Exception as exc:
        record_status(name, "ERROR", f"engine initialisation: {type(exc).__name__}: {exc}", vars(args))
        return 2
    ok = run_twice(name, fn, repeats=args.repeats, extra={**vars(args), "prompts": PROMPTS, "input_sha256": hashlib.sha256(canonical_bytes(PROMPTS)).hexdigest(), "weight_content_hash": None, "tokenizer_content_hash": None, "observation_schema": "labelled-topk-v2"})
    if args.print_hash:
        maybe_print_hash(fn())  # explicitly requested diagnostic evaluation, outside the measured repeats
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
