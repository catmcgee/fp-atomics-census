#!/usr/bin/env python
"""The ten-point GPU smoke test of PLAN.md section 1, run against arm directories on the pod.

Usage: smoke.py RESULTS_ROOT
Points that need a topology or a model this campaign does not use (TP=2 for points 1 and 3, an MoE model
for point 7) are reported N/A with the reason rather than silently passed.
"""
import json, sys, subprocess
from pathlib import Path

root = Path(sys.argv[1])
res = []
def add(n, ok, msg):
    res.append((n, ok, msg))

def rows(d):
    p = Path(d) / "hook" / "rank0.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

base = root / "Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0"
rec, rep = base / "record", base / "replay_warm"
fp8pt = root / "Qwen_Qwen2.5-7B-Instruct_tp1_fp8_pertensor_compile_v2_graphs1_prefix0" / "record"
fp8pk = root / "Qwen_Qwen2.5-7B-Instruct_tp1_fp8_compile_v2_graphs1_prefix0" / "record"

# 1 plugin loaded in the engine core and every worker
ok = (rec / "hook" / "rank0.jsonl").exists() and (rep / "hook" / "rank0.jsonl").exists()
add(1, ok, f"rank0.jsonl present in record and warm replay: {ok}; TP=2 rank1 N/A (C1 is a one-GPU campaign, C2 covers TP=2)")

# 2 model_runner v1 everywhere
fwd = {"record": [r for r in rows(rec) if r.get("event") == "forward"], "replay": [r for r in rows(rep) if r.get("event") == "forward"]}
mr = {k: sorted({(r.get("model_runner") or {}).get("runner") if isinstance(r.get("model_runner"), dict) else r.get("model_runner") for r in v}) for k, v in fwd.items()}
runs = {k: json.loads((d / "run.json").read_text()) for k, d in (("record", rec), ("replay", rep))}
rmr = {k: v.get("resolved_model_runner") for k, v in runs.items()}
ok = all(set(x) <= {"v1"} for x in mr.values()) and set(rmr.values()) == {"v1"}
add(2, ok, f"hook model_runner {mr}, run.json resolved_model_runner {rmr}")

# 3 schema 3, new_token_ids present, no invalid row
bad, nomiss, sch = 0, 0, set()
for k, v in fwd.items():
    for r in v:
        sch.add(r.get("schema"))
        for row in r.get("requests", []):
            if str(row.get("req", "")).startswith("_warmup"):
                continue
            if row.get("invalid"): bad += 1
            if row.get("new_token_ids") is None: nomiss += 1
ok = sch == {3} and bad == 0 and nomiss == 0
add(3, ok, f"schema {sorted(sch)}, rows with an invalid reason {bad}, rows missing new_token_ids {nomiss}; TP=2 half N/A (one-GPU campaign)")

# 4 attention backend, cudagraph mode on decode passes, resolved compile/graphs
be = sorted({r.get("attention_backend") for r in fwd["record"]})
decode_modes = sorted({(r.get("dispatch") or {}).get("cudagraph_mode") for r in fwd["record"] if r.get("total_scheduled") == r.get("num_reqs")})
rc = {k: (v.get("resolved_compile"), v.get("resolved_cudagraph")) for k, v in runs.items()}
ok = all(b for b in be) and all(m in ("FULL", "PIECEWISE") for m in decode_modes if m is not None) and all("3" in str(a) and "FULL_AND_PIECEWISE" in str(b) for a, b in rc.values())
add(4, ok, f"attention_backend {be}, decode-pass cudagraph modes {decode_modes}, resolved (compile, cudagraph) {rc}")

# 5 linear_quant on a quantised model, per-tensor key differs from per-token
lq = {}
for name, d in (("bf16", rec), ("fp8_per_token", fp8pk), ("fp8_per_tensor", fp8pt)):
    if (d / "hook" / "rank0.jsonl").exists():
        f = [r for r in rows(d) if r.get("event") == "forward"]
        lq[name] = f[0].get("linear_quant") if f else None
keys = {k: (v or {}).get("activation_quant_key") for k, v in lq.items()}
ok = bool(lq.get("fp8_per_token")) and bool(lq.get("fp8_per_tensor")) and keys.get("fp8_per_token") != keys.get("fp8_per_tensor")
add(5, ok, f"linear_quant {json.dumps(lq)}")

# 6 fp8_forcing on the forced arm
ff = None
if (fp8pt / "hook" / "rank0.jsonl").exists():
    f = [r for r in rows(fp8pt) if r.get("event") == "forward"]
    ff = f[0].get("fp8_forcing") if f else None
mods = (ff or {}).get("patched_modules") or []
ok = bool(mods) and any("quantization.online.fp8" in m for m in mods)
add(6, ok, f"fp8_forcing.patched_modules {mods}")

# 7 MoE routing counts
add(7, None, "N/A: C1 runs one dense model (Qwen2.5-7B-Instruct); no MoE arm. C2 and C3 carry the MoE arms")

# 8 cache_before_engine empty on a cold arm; artefacts.json lists best_configs and extern_calls
cb = json.loads((rec / "cache_before_engine.json").read_text())
art = json.loads((rec / "artefacts.json").read_text())
ok = cb.get("empty") is True and bool(art.get("best_configs")) and bool(art.get("extern_calls"))
add(8, ok, f"record cache_before_engine.empty={cb.get('empty')}, artefacts best_configs={len(art.get('best_configs') or {})}, extern_calls={art.get('extern_calls')}, triton_kernel_families={len(art.get('triton_kernel_families') or {})}")

# 9 one forcing call per forward pass; --compare of the record against itself is INVALID on the run-id check
fl = [json.loads(l) for l in (rec / "hook" / "forcing_rank0.jsonl").read_text().splitlines() if l.strip()] if (rec / "hook" / "forcing_rank0.jsonl").exists() else []
p = subprocess.run([sys.executable, "run_e6.py", "--compare", str(rec), str(rec)], capture_output=True, text=True, cwd="/root/census/probes/shape")
self_out = (p.stdout + p.stderr).strip().splitlines()
ok = len(fl) == len(fwd["record"]) and any("INVALID" in l for l in self_out)
add(9, ok, f"forcing calls {len(fl)} vs forward passes {len(fwd['record'])}; self-compare says: {self_out[-1][:200] if self_out else '(no output)'}")

# 10 batch-sharded sampling off. The field is ParallelConfig.enable_batch_sharded_sampling; vLLM 0.29.0 leaves it
# unset (None) rather than False and does not print it in the resolved config, so the test is: the field exists,
# its resolved value is not true, the runner resolved to V1, and the hook's batch-sharded refusal path was not
# taken (a record with hashes was written).
import dataclasses
from vllm.config import ParallelConfig
fields = {f.name: f for f in dataclasses.fields(ParallelConfig)}
present = "enable_batch_sharded_sampling" in fields
value = ParallelConfig().enable_batch_sharded_sampling if present else "field absent"
printed = {k: ("batch_shard" in (v.get("resolved_config") or "")) for k, v in runs.items()}
hashed = all(any(row.get("h") for row in r.get("requests", [])) for r in fwd["record"][:1])
ok = present and not value and all(v.get("resolved_model_runner") == "v1" for v in runs.values()) and hashed
add(10, ok, f"ParallelConfig.enable_batch_sharded_sampling present={present} resolved={value!r} (not printed in resolved_config: {printed}); resolved_model_runner v1 in record and replay; hook wrote hashed rows, so the batch-sharded refusal path was not taken")

fails = [n for n, ok, _ in res if ok is False]
for n, ok, msg in res:
    print(f"SMOKE {n:>2} {'PASS' if ok else ('N/A ' if ok is None else 'FAIL')} :: {msg}")
print("SMOKE_RESULT", "PASS" if not fails else f"FAIL points {fails}")
