"""Shared helpers for the batch-shape experiments (probes/shape/)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
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
    base = [p for pair in zip(PROMPTS[:4], long) for p in pair]
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def read_hook(out_dir: Path, rank: int = 0) -> list[dict]:
    p = Path(out_dir) / f"rank{rank}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def steps_by_request(steps: list[dict]) -> dict[str, list[tuple[int, str, str | None, int | None]]]:
    """req id -> [(step, shape_vector, hidden hash, argmax)] in step order."""
    out: dict[str, list] = defaultdict(list)
    for s in real_steps(steps):
        for batch_slot, r in enumerate(s["requests"]):
            if str(r["req"]).startswith("_warmup"):
                continue
            out[slot_of(r["req"])].append((s["step"], recorded_shape(s) + f"|target_slot={batch_slot}", r.get("h"), r.get("argmax")))
    return out


def real_steps(steps: list[dict]) -> list[dict]:
    """Exclude warmups and empty scheduler calls, including stale legacy rows."""
    return [s for s in steps if s.get("requests")
            and s.get("event", "forward") == "forward"
            and s.get("total_scheduled", sum(r.get("q", 0) for r in s["requests"])) > 0
            and not all(str(r["req"]).startswith("_warmup") for r in s["requests"])]


def recorded_shape(s: dict) -> str:
    """Ordered *recorded* state, not a claim to capture every kernel decision."""
    return json.dumps({"legacy_shape": s.get("shape_vector"), "dispatch": s.get("dispatch"),
                       "parallel": s.get("parallel"), "attention_backend": s.get("attention_backend"),
                       "rows": [[r.get(k) for k in ("q", "kv", "computed", "prompt", "phase")]
                                for r in s["requests"]]}, sort_keys=True, separators=(",", ":"))


def trace_errors(steps: list[dict]) -> list[str]:
    errors, seen = [], set()
    previous_by_request = {}
    previous_by_rank = {}
    for s in steps:
        key = (s.get("rank", 0), s.get("step"))
        if key in seen:
            errors.append(f"duplicate step {key}")
        seen.add(key)
        if key[0] in previous_by_rank and key[1] <= previous_by_rank[key[0]]:
            errors.append(f"out-of-order step {key}")
        previous_by_rank[key[0]] = key[1]
        reqs = s.get("requests", [])
        ids = [slot_of(r["req"]) for r in reqs]
        if len(set(ids)) != len(ids):
            errors.append(f"duplicate request in step {key}")
        if not reqs or any(r.get("h") is None or r.get("argmax") is None for r in reqs):
            errors.append(f"missing observation in step {key}")
        if s.get("total_scheduled", sum(r.get("q", 0) for r in reqs)) != sum(r.get("q", 0) for r in reqs):
            errors.append(f"scheduled token count disagrees in step {key}")
        if any(r.get("q", 0) <= 0 for r in reqs):
            errors.append(f"unscheduled request in step {key}")
        for r in reqs:
            rid = slot_of(r["req"])
            prior = previous_by_request.get(rid)
            if prior and prior.get("kv") is not None and r.get("computed") != prior["kv"]:
                errors.append(f"request {rid} has a missing or discontinuous forward observation")
            previous_by_request[rid] = r
            if s.get("schema") == 2 and (not r.get("logits_h") or len(r.get("h") or "") != 64):
                errors.append(f"missing version-2 full hashes in step {key}")
        if s.get("num_reqs", len(reqs)) != len(reqs):
            errors.append(f"request count disagrees in step {key}")
    if not steps:
        errors.append("empty forward trace")
    return errors


def analysis_metadata(out: Path, files: list[Path]) -> dict:
    """Content-address inputs and analyser; never replace raw observations."""
    code = list(HERE.glob("*.py")) + list((HERE / "shape_hook_pkg" / "shape_hook").glob("*.py"))
    digest = hashlib.sha256()
    for p in sorted(code):
        digest.update(p.name.encode() + b"\0" + p.read_bytes())
    return {"analysis_schema": 2, "analyser_sha256": digest.hexdigest(),
            "input_sha256": {str(p.resolve().relative_to(out.resolve())): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(files) if p.is_file()},
            "observation_limits": "Legacy runs use truncated hidden hashes and unlabelled sparse logprobs; no KV commitment or teacher-forced replay."}


def saved_env(out: Path):
    for p in (out / "env.json", out / "summary.json"):
        if p.exists():
            obj = json.loads(p.read_text())
            return obj.get("env", obj) if p.name == "summary.json" else obj
    return {"provenance_status": "missing; not replaced with analysis host"}


def history_key(seq: list[tuple], upto: int) -> str:
    """Hash of the shape vectors of a request's steps 0..upto inclusive."""
    h = hashlib.sha256()
    for _, sv, _, _ in seq[: upto + 1]:
        h.update(sv.encode())
        h.update(b"\n")
    return h.hexdigest()


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
    sys.path.insert(0, str(HERE.parent))
    from observations import sparse_output
    return {o.request_id: sparse_output(o.outputs[0].token_ids,
                                       [((t, lp.logprob) for t, lp in pos.items())
                                        for pos in o.outputs[0].logprobs or []]) for o in outs}


def env_with_hook(out_dir: Path) -> None:
    os.environ["SHAPE_HOOK_OUT"] = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=False)
    os.environ["SHAPE_RUN_ID"] = str(uuid.uuid4())


def engine_kwargs(args) -> dict:
    graphs = bool(args.cudagraph)
    compile_on = not getattr(args, "no_compile", False)
    # vLLM 0.28.0: FULL_AND_PIECEWISE requires compilation. FULL is the
    # explicit graph-only arm; fail on unsupported models instead of relabelling.
    graph_mode = ("FULL_AND_PIECEWISE" if compile_on else "FULL") if graphs else "NONE"
    kw = {"model": args.model, "tensor_parallel_size": args.tp, "seed": 0, "max_model_len": 2048,
          "enable_prefix_caching": bool(args.prefix_caching), "enforce_eager": False,
          "compilation_config": {"mode": 3 if compile_on else 0, "cudagraph_mode": graph_mode}}
    if getattr(args, "revision", None):
        kw["revision"] = args.revision
        kw["tokenizer_revision"] = args.revision
    if getattr(args, "quantization", None):
        kw["quantization"] = args.quantization
    if getattr(args, "max_num_seqs", None):
        kw["max_num_seqs"] = args.max_num_seqs
    os.environ["SHAPE_EXPECT_COMPILE"] = str(int(compile_on))
    os.environ["SHAPE_EXPECT_GRAPHS"] = str(int(graphs))
    return kw


def record_run(out: Path, args, llm, prompts) -> None:
    config = llm.llm_engine.vllm_config
    cc = config.compilation_config
    compile_on = int(cc.mode) != 0
    graphs_on = str(getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)) != "NONE"
    expected = (not args.no_compile, bool(args.cudagraph))
    actual = (compile_on, graphs_on)
    record = {"schema": 2, "run_id": os.environ["SHAPE_RUN_ID"],
              "args": vars(args), "dataset": "mixed-v2", "prompts": prompts,
              "input_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              "resolved_config": str(config), "resolved_compile": str(cc.mode),
              "resolved_cudagraph": str(cc.cudagraph_mode),
              "model_revision": getattr(config.model_config.hf_config, "_commit_hash", None),
              "tokenizer_revision": getattr(config.model_config, "tokenizer_revision", None),
              "weight_content_hash": None, "kernel_content_hash": None,
              "status": "configured" if actual == expected else "INVALID"}
    write_json(out / "run.json", record)
    env = environment()
    env["run_id"] = os.environ["SHAPE_RUN_ID"]
    write_json(out / "env.json", env)
    if actual != expected:
        raise RuntimeError(f"resolved compile/graphs {actual} != requested {expected}")


def add_common_args(ap) -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pin model and tokenizer revision; resolved commit is recorded")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--cudagraph", type=int, choices=(0, 1), default=1, help="explicit CUDA graph control, independent of --no-compile")
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True, help="results directory for this arm")
    ap.add_argument("--fp8-per-tensor", action="store_true", help="force per-tensor dynamic FP8 activation scales (disables the CUTLASS FP8 path)")
    ap.add_argument("--no-compile", action="store_true", help="disable compilation; combine with --cudagraph 0 for eager execution")


def arm_name(args) -> str:
    return f"{args.model.replace('/', '_')}_tp{args.tp}_{args.quantization or 'none'}{'_pertensor' if getattr(args, 'fp8_per_tensor', False) else ''}{'_nocompile' if getattr(args, 'no_compile', False) else '_compile'}_v2_graphs{args.cudagraph}_prefix{args.prefix_caching}"
