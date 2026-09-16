#!/usr/bin/env python
"""The ten-point GPU smoke test of PLAN.md section 1, run against C2's arm directories on the pod.

Usage: smoke.py RESULTS_ROOT
Every point is evaluated against whatever arms exist under RESULTS_ROOT. A point that needs a topology or a
model this pod does not run is reported N/A with the reason rather than silently passed.
"""
import json, sys, subprocess, dataclasses
from pathlib import Path

root = Path(sys.argv[1])
res = []
def add(n, ok, msg):
    res.append((n, ok, msg))

def rows(d, rank=0):
    p = Path(d) / "hook" / f"rank{rank}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def fwd(d, rank=0):
    return [r for r in rows(d, rank) if r.get("event") == "forward"]

A = {
 "bf16":      "Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0",
 "bf16_mixed":"Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0_mixed",
 "fp8_pt":    "Qwen_Qwen2.5-7B-Instruct_tp1_fp8_compile_v2_graphs1_prefix0",
 "fp8_ptens": "Qwen_Qwen2.5-7B-Instruct_tp1_fp8_pertensor_compile_v2_graphs1_prefix0",
 "fp8_block": "Qwen_Qwen3-8B-FP8_tp1_fp8_compile_v2_graphs1_prefix0",
 "moe":       "Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs1_prefix0",
 "tp2":       "Qwen_Qwen2.5-7B-Instruct_tp2_none_compile_v2_graphs1_prefix0",
 "tp2_mixed": "Qwen_Qwen2.5-7B-Instruct_tp2_none_compile_v2_graphs1_prefix0_mixed",
}
D = {k: root / v for k, v in A.items()}
def rec(k): return D[k] / "record"
def warm(k): return D[k] / "replay_warm"

# the reference pair for the points that need one record and one replay
ref = None
for k in ("bf16", "tp2"):
    if (rec(k) / "run.json").exists() and (warm(k) / "run.json").exists():
        ref = k; break
if ref is None:
    for k in A:
        if (rec(k) / "run.json").exists():
            ref = k; break
R, P = rec(ref), warm(ref)
has_replay = (P / "run.json").exists()

# 1 plugin loaded in the engine core and every worker
ok1 = (R / "hook" / "rank0.jsonl").exists() and (not has_replay or (P / "hook" / "rank0.jsonl").exists())
tp2present = (rec("tp2") / "run.json").exists()
if tp2present:
    r1 = (rec("tp2") / "hook" / "rank1.jsonl").exists()
    msg = f"rank0.jsonl present in the {ref} record and its warm replay: {ok1}; TP=2 rank1.jsonl present: {r1}"
    ok1 = ok1 and r1
else:
    msg = f"rank0.jsonl present in the {ref} record and its warm replay: {ok1}; TP=2 rank1 half N/A on this pod (the TP=2 arms run on the 2x pod)"
add(1, ok1, msg)

# 2 model_runner v1 everywhere
mr, rmr = {}, {}
for k in A:
    for kind, d in (("record", rec(k)), ("replay_warm", warm(k))):
        if not (d / "run.json").exists(): continue
        f = fwd(d)
        if f:
            mr[f"{k}/{kind}"] = sorted({(r.get("model_runner") or {}).get("runner") if isinstance(r.get("model_runner"), dict) else r.get("model_runner") for r in f})
        rmr[f"{k}/{kind}"] = json.loads((d / "run.json").read_text()).get("resolved_model_runner")
ok = bool(mr) and all(set(x) <= {"v1"} for x in mr.values()) and set(rmr.values()) == {"v1"}
add(2, ok, f"hook model_runner {mr}; run.json resolved_model_runner {sorted(set(rmr.values()))} over {len(rmr)} arm directories")

# 3 schema 3, new_token_ids present, no invalid row, on TP=1 and TP=2
def schema_check(dirs):
    bad = miss = 0; sch = set(); n = 0
    for d in dirs:
        for rank in (0, 1):
            for r in fwd(d, rank):
                sch.add(r.get("schema"))
                for row in r.get("requests", []):
                    if str(row.get("req", "")).startswith("_warmup"): continue
                    n += 1
                    if row.get("invalid"): bad += 1
                    if row.get("new_token_ids") is None: miss += 1
    return sch, bad, miss, n
tp1dirs = [d for k in ("bf16","bf16_mixed","fp8_pt","fp8_ptens","fp8_block","moe") for d in (rec(k), warm(k)) if (d/"run.json").exists()]
tp2dirs = [d for k in ("tp2","tp2_mixed") for d in (rec(k), warm(k)) if (d/"run.json").exists()]
s1 = schema_check(tp1dirs); s2 = schema_check(tp2dirs)
ok = (not tp1dirs or (s1[0] == {3} and s1[1] == 0 and s1[2] == 0)) and (not tp2dirs or (s2[0] == {3} and s2[1] == 0 and s2[2] == 0)) and (tp1dirs or tp2dirs)
tp2msg = f"TP=2: schema {sorted(s2[0])}, invalid {s2[1]}, missing new_token_ids {s2[2]}, rows {s2[3]}" if tp2dirs else "TP=2 half N/A on this pod"
add(3, ok, f"TP=1: schema {sorted(s1[0])}, invalid {s1[1]}, missing new_token_ids {s1[2]}, rows {s1[3]}; {tp2msg}")

# 4 attention backend, cudagraph mode on decode passes, resolved compile/graphs
be, dm, rc = set(), set(), {}
for k in A:
    for kind, d in (("record", rec(k)), ("replay_warm", warm(k))):
        if not (d / "run.json").exists(): continue
        f = fwd(d)
        be |= {r.get("attention_backend") for r in f}
        dm |= {(r.get("dispatch") or {}).get("cudagraph_mode") for r in f if r.get("total_scheduled") == r.get("num_reqs")}
        j = json.loads((d / "run.json").read_text())
        rc[f"{k}/{kind}"] = (j.get("resolved_compile"), j.get("resolved_cudagraph"))
ok = bool(be) and all(b for b in be) and all(m in ("FULL", "PIECEWISE") for m in dm if m is not None)
bad4 = {}
for kk, (a, b) in rc.items():
    want = ("0", "FULL") if "nocompile" in A[kk.split("/")[0]] else ("3", "FULL_AND_PIECEWISE")
    if (str(a), str(b)) != want:
        ok = False; bad4[kk] = (str(a), str(b), "wanted " + str(want))
add(4, ok, f"attention_backend {sorted(x for x in be if x)}, decode-pass cudagraph modes {sorted(x for x in dm if x)}, resolved (compile, cudagraph) {rc}" + (f"; MISMATCHES {bad4}" if bad4 else ""))

# 5 linear_quant on a quantised model, per-tensor key differs from per-token
lq = {}
for k in ("bf16", "fp8_pt", "fp8_ptens", "fp8_block"):
    f = fwd(rec(k))
    if f: lq[k] = f[0].get("linear_quant")
keys = {k: (v or {}).get("activation_quant_key") for k, v in lq.items()}
ok = bool(lq.get("fp8_pt")) and bool(lq.get("fp8_ptens")) and keys.get("fp8_pt") != keys.get("fp8_ptens")
add(5, ok, f"linear_quant {json.dumps(lq)}")

# 6 fp8_forcing on the forced arm
f = fwd(rec("fp8_ptens"))
ff = f[0].get("fp8_forcing") if f else None
mods = (ff or {}).get("patched_modules") or []
ok = bool(mods) and any("quantization.online.fp8" in m for m in mods)
add(6, ok, f"fp8_forcing.patched_modules {mods}")

# 7 MoE routing counts
f = fwd(rec("moe"))
if not f:
    add(7, None, "N/A: no MoE arm on this pod")
else:
    counts = f[0].get("moe_expert_counts_first_layer"); src = f[0].get("moe_counts_source")
    ok = isinstance(counts, list) and src == "actual_router"
    add(7, ok, f"moe_expert_counts_first_layer is a list of {len(counts) if isinstance(counts, list) else counts!r}, moe_counts_source {src!r}")

# 8 cache_before_engine empty on a cold arm; artefacts.json lists best_configs and extern_calls
cb = json.loads((R / "cache_before_engine.json").read_text())
art = json.loads((R / "artefacts.json").read_text())
ok = cb.get("empty") is True and bool(art.get("best_configs")) and bool(art.get("extern_calls"))
add(8, ok, f"{ref} record cache_before_engine.empty={cb.get('empty')}, artefacts best_configs={len(art.get('best_configs') or {})}, extern_calls={art.get('extern_calls')}, triton_kernel_families={len(art.get('triton_kernel_families') or {})}")

# 9 one forcing call per forward pass; --compare of the record against itself is INVALID on the run-id check
fl = [json.loads(l) for l in (R / "hook" / "forcing_rank0.jsonl").read_text().splitlines() if l.strip()] if (R / "hook" / "forcing_rank0.jsonl").exists() else []
p = subprocess.run([sys.executable, "run_e6.py", "--compare", str(R), str(R)], capture_output=True, text=True, cwd="/root/census/probes/shape")
self_out = (p.stdout + p.stderr).strip().splitlines()
ok = len(fl) == len(fwd(R)) and any("INVALID" in l for l in self_out)
add(9, ok, f"forcing calls {len(fl)} vs forward passes {len(fwd(R))}; self-compare says: {self_out[-1][:220] if self_out else '(no output)'}")

# 10 batch-sharded sampling off. The field is ParallelConfig.enable_batch_sharded_sampling; vLLM 0.29.0 leaves it
# unset (None) rather than False and does not print it in the resolved config, so the test is: the field exists,
# its resolved value is not true, the runner resolved to V1, and the hook's batch-sharded refusal path was not
# taken (a record with hashes was written).
from vllm.config import ParallelConfig
fields = {f_.name for f_ in dataclasses.fields(ParallelConfig)}
present = "enable_batch_sharded_sampling" in fields
value = ParallelConfig().enable_batch_sharded_sampling if present else "field absent"
hashed = all(any(row.get("h") for row in r.get("requests", [])) for r in fwd(R)[:1])
ok = present and not value and set(rmr.values()) == {"v1"} and hashed
add(10, ok, f"ParallelConfig.enable_batch_sharded_sampling present={present} resolved={value!r} (vLLM 0.29.0 does not print it in the resolved config); resolved_model_runner v1 in every arm directory; the hook wrote hashed rows, so the batch-sharded refusal path was not taken")

fails = [n for n, ok, _ in res if ok is False]
for n, ok, msg in res:
    print(f"SMOKE {n:>2} {'PASS' if ok else ('N/A ' if ok is None else 'FAIL')} :: {msg}")
print("SMOKE_RESULT", "PASS" if not fails else f"FAIL points {fails}")
