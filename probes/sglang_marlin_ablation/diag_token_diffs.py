"""Diagnostic, not a census probe: count sampled-token changes across in-process repeats.

Builds the engine exactly as probes/probe_engine_logits.py::sglang_fn does (same
prompts, greedy, 32 new tokens, radix cache off, random_seed 0, pinned revision),
runs a baseline plus N repeats, and reports how many of the 16*32 sampled tokens
and top-5 positions differ from the baseline in each repeat. Output: one JSON file.
"""
import json, os, sys, time
sys.path.insert(0, "/workspace/census/probes")
from probe_engine_logits import PROMPTS  # identical prompt batch

import sglang as sgl


def main():

    name, out, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])
    extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
    kw = {k.replace("-", "_"): (v.lower() == "true" if v.lower() in ("true", "false") else v) for k, v in extra.items()}
    kw.setdefault("disable_radix_cache", True)
    engine = sgl.Engine(model_path="Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", tp_size=1, quantization=None, random_seed=0,
                        revision="4f5a0e31008e2966e8f787964bcc4e1105e03dc3", **kw)
    params = {"temperature": 0, "max_new_tokens": 32}

    def evaluate():
        outs = engine.generate(PROMPTS, params, return_logprob=True, top_logprobs_num=5)
        toks = [[t[1] for t in o["meta_info"]["output_token_logprobs"]] for o in outs]
        top5 = [[sorted((lp[1], lp[0]) for lp in pos) for pos in o["meta_info"]["output_top_logprobs"]] for o in outs]
        return toks, top5

    record = {"diagnostic": name, "env_SGLANG_MARLIN_USE_ATOMIC_ADD": os.environ.get("SGLANG_MARLIN_USE_ATOMIC_ADD"),
              "extra_kwargs": kw, "prompts": len(PROMPTS), "max_new_tokens": 32, "repeats": []}
    base_t, base_l = evaluate()
    record["baseline_total_tokens"] = sum(len(t) for t in base_t)
    for k in range(1, repeats + 1):
        t, l = evaluate()
        tok_changed = req_changed = first_div = 0
        first_positions = []
        for a, b in zip(base_t, t):
            n = max(len(a), len(b))
            diff = [i for i in range(n) if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
            tok_changed += len(diff)
            if diff:
                req_changed += 1
                first_positions.append(diff[0])
        top5_changed = sum(1 for a, b in zip(base_l, l) for i in range(min(len(a), len(b))) if a[i] != b[i])
        top5_values_changed_same_tokens = sum(1 for a, b, ta, tb in ((a, b, ta, tb) for a, b, ta, tb in zip(base_l, l, base_t, t))
                                              for i in range(min(len(a), len(b), len(ta), len(tb)))
                                              if ta[:i + 1] == tb[:i + 1] and a[i] != b[i])
        record["repeats"].append({"repeat": k, "identical": t == base_t and l == base_l, "sampled_tokens_changed": tok_changed,
                                  "requests_changed": req_changed, "first_divergence_positions": first_positions,
                                  "top5_positions_changed": top5_changed,
                                  "top5_positions_changed_before_any_token_divergence": top5_values_changed_same_tokens,
                                  "total_generated_tokens": sum(len(x) for x in t)})
        print(json.dumps(record["repeats"][-1]), flush=True)
    json.dump(record, open(out, "w"), indent=2)
    print("DIAG_DONE", out, flush=True)
    engine.shutdown()


if __name__ == "__main__":  # the spawn-started scheduler re-imports this module; it must not build an engine
    main()
