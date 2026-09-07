"""Shared helpers for the batch-shape experiments (probes/shape/)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def environment() -> dict:
    """The environment record the other probes write (probes/common.py), loaded under a distinct name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("probes_common", HERE.parent / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.environment()

PROMPTS = [
    "The verifier re-runs a sampled computation and compares hashes.",
    "Floating-point addition is not associative, so",
    "def fibonacci(n):",
    "In 1854 the",
] * 4


def mixed_prompts(n: int = 16) -> list[str]:
    """The request set of the earlier probes plus longer prompts, so prefill lengths differ."""
    long = [" ".join(["The verifier records the batch shape at every step and replays it later."] * k) for k in (3, 6, 9, 12)]
    return (PROMPTS + long)[:n] if n <= 20 else (PROMPTS + long) * (n // 20 + 1)


def read_hook(out_dir: Path, rank: int = 0) -> list[dict]:
    p = Path(out_dir) / f"rank{rank}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def steps_by_request(steps: list[dict]) -> dict[str, list[tuple[int, str, str | None, int | None]]]:
    """req id -> [(step, shape_vector, hidden hash, argmax)] in step order."""
    out: dict[str, list] = defaultdict(list)
    for s in steps:
        if "requests" not in s:
            continue
        for r in s["requests"]:
            if str(r["req"]).startswith("_warmup"):
                continue
            out[r["req"]].append((s["step"], s["shape_vector"], r.get("h"), r.get("argmax")))
    return out


def real_steps(steps: list[dict]) -> list[dict]:
    """Drop warm-up and profiling passes (request ids starting with _warmup)."""
    return [s for s in steps if "requests" in s and not all(str(r["req"]).startswith("_warmup") for r in s["requests"])]


def history_key(seq: list[tuple], upto: int) -> str:
    """Hash of the shape vectors of a request's steps 0..upto inclusive."""
    h = hashlib.sha256()
    for _, sv, _, _ in seq[: upto + 1]:
        h.update(sv.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def random_token_prompt(tokenizer, length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = []
    while len(ids) < length:
        t = rng.randrange(0, vocab)
        if t not in special:
            ids.append(t)
    return ids


def slot_of(rid: str) -> str:
    """vLLM 0.28.0 request ids are '<counter>-<random>'; the counter is the slot, the suffix differs per process."""
    return str(rid).split("-")[0]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def outputs_record(outs) -> dict:
    """Per-request tokens and top logprobs from vLLM RequestOutputs, keyed by request id."""
    rec = {}
    for o in outs:
        vals = []
        for pos in o.outputs[0].logprobs or []:
            vals.extend(sorted(lp.logprob for lp in pos.values()))
        rec[o.request_id] = {"tokens": list(o.outputs[0].token_ids), "logprobs": vals,
                             "hash": hashlib.sha256(json.dumps([list(o.outputs[0].token_ids), vals]).encode()).hexdigest()[:16]}
    return rec


def env_with_hook(out_dir: Path) -> None:
    os.environ["SHAPE_HOOK_OUT"] = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def engine_kwargs(args) -> dict:
    kw = {"model": args.model, "tensor_parallel_size": args.tp, "seed": 0, "max_model_len": 2048,
          "enable_prefix_caching": bool(args.prefix_caching), "enforce_eager": not bool(args.cudagraph)}
    if getattr(args, "quantization", None):
        kw["quantization"] = args.quantization
    if getattr(args, "max_num_seqs", None):
        kw["max_num_seqs"] = args.max_num_seqs
    return kw


def add_common_args(ap) -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--cudagraph", type=int, default=1, help="1: CUDA graphs on (default); 0: enforce_eager")
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True, help="results directory for this arm")
    ap.add_argument("--fp8-per-tensor", action="store_true", help="force per-tensor dynamic FP8 activation scales (disables the CUTLASS FP8 path)")


def arm_name(args) -> str:
    return f"{args.model.replace('/', '_')}_tp{args.tp}_{args.quantization or 'none'}{'_pertensor' if getattr(args, 'fp8_per_tensor', False) else ''}_graphs{args.cudagraph}_prefix{args.prefix_caching}"
