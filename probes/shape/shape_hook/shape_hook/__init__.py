"""Batch-shape instrumentation for vLLM, installed as a general plugin.

Loaded by vLLM in the engine core and in every worker process through the
``vllm.general_plugins`` entry point (vllm/plugins/__init__.py,
vllm/v1/worker/worker_base.py:245-247), so it works with tensor
parallelism. It stays outside the engine source: it monkeypatches four
call sites of vLLM 0.28.0 (pinned census sha 5769a7382cb1 for line
references):

* ``GPUModelRunner.execute_model`` (vllm/v1/worker/gpu_model_runner.py:4069):
  reads the scheduler output and the input batch for the shape vector.
* ``CudagraphDispatcher.dispatch`` (vllm/v1/cudagraph_dispatcher.py:235):
  records the runtime mode and the padded token count of the graph used.
* the model's ``compute_logits`` (called at gpu_model_runner.py:4387):
  hashes each sampled row of the last hidden state and takes the argmax
  of the logits, one row per scheduled request in input-batch order.
* ``FusedMoERouter.select_experts``
  (vllm/model_executor/layers/fused_moe/router/fused_moe_router.py:45):
  per-expert token counts of the first MoE layer of each pass.

Output: one JSON line per forward pass in ``$SHAPE_HOOK_OUT/rank<r>.jsonl``.
Nothing is written when ``SHAPE_HOOK_OUT`` is unset.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time

_state = {"registered": False, "step": 0, "moe_counts": None, "dispatch": None, "hashes": None, "seen": set(), "env": None}


def _env_record() -> dict:
    import torch
    try:
        import importlib.metadata as md
        pk = {p: md.version(p) for p in ("vllm", "flashinfer-python", "torch", "triton") if _has(md, p)}
    except Exception:  # noqa: BLE001
        pk = {}
    rec = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "packages": pk,
           "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "VLLM_ATTENTION_BACKEND", "VLLM_MARLIN_USE_ATOMIC_ADD", "CUBLAS_WORKSPACE_CONFIG", "VLLM_ENABLE_V1_MULTIPROCESSING")}}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec.update({"gpu": p.name, "sm": f"{p.major}{p.minor}", "gpu_count": torch.cuda.device_count()})
        try:
            rec["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split("\n")[0].strip()
        except Exception:  # noqa: BLE001
            rec["driver"] = None
    return rec


def _has(md, name: str) -> bool:
    try:
        md.version(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _out_path() -> str | None:
    d = os.environ.get("SHAPE_HOOK_OUT")
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"rank{_rank()}.jsonl")


def _row_hash(t) -> str:
    import torch
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        b = t.view(torch.int16).cpu().numpy().tobytes()
    elif t.dtype == torch.float32:
        b = t.view(torch.int32).cpu().numpy().tobytes()
    else:
        b = t.cpu().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _patch_dispatcher():
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    orig = CudagraphDispatcher.dispatch

    def dispatch(self, *a, **kw):
        out = orig(self, *a, **kw)
        try:
            mode, desc = out
            _state["dispatch"] = {"cudagraph_mode": str(getattr(mode, "name", mode)),
                                  "padded_num_tokens": int(getattr(desc, "num_tokens", -1)) if desc is not None else None,
                                  "uniform_decode": bool(getattr(desc, "uniform_decode", False)) if desc is not None else None}
        except Exception as e:  # noqa: BLE001
            _state["dispatch"] = {"error": repr(e)[:80]}
        return out

    CudagraphDispatcher.dispatch = dispatch


def _patch_router():
    try:
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    except Exception:  # noqa: BLE001
        return
    orig = FusedMoERouter.select_experts

    def select_experts(self, *a, **kw):
        out = orig(self, *a, **kw)
        try:
            if _state["moe_counts"] is None:
                import torch
                topk_ids = out[1]
                n = int(getattr(self, "global_num_experts", 0) or getattr(getattr(self, "moe", None), "num_experts", 0) or int(topk_ids.max().item()) + 1)
                _state["moe_counts"] = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=n).tolist()
        except Exception as e:  # noqa: BLE001
            _state["moe_counts"] = {"error": repr(e)[:80]}
        return out

    FusedMoERouter.select_experts = select_experts


def _wrap_compute_logits(model):
    if getattr(model, "_shape_hook_wrapped", False):
        return
    orig = model.compute_logits

    def compute_logits(hidden_states, *a, **kw):
        logits = orig(hidden_states, *a, **kw)
        try:
            import torch
            hs = hidden_states
            rows = [_row_hash(hs[i]) for i in range(hs.shape[0])]
            am = torch.argmax(logits.float(), dim=-1).tolist() if logits is not None else None
            _state["hashes"] = {"hidden_rows": rows, "argmax": am}
        except Exception as e:  # noqa: BLE001
            _state["hashes"] = {"error": repr(e)[:80]}
        return logits

    model.compute_logits = compute_logits
    model._shape_hook_wrapped = True


def _patch_runner():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig = GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *a, **kw):
        path = _out_path()
        if path is None:
            return orig(self, scheduler_output, *a, **kw)
        _state["moe_counts"] = None
        _state["dispatch"] = None
        _state["hashes"] = None
        try:
            _wrap_compute_logits(self.model)
        except Exception:  # noqa: BLE001
            pass
        t0 = time.time()
        out = orig(self, scheduler_output, *a, **kw)
        try:
            rec = _record(self, scheduler_output)
            rec["wall_ms"] = round((time.time() - t0) * 1000, 1)
            if _state["env"] is None:
                _state["env"] = _env_record()
                rec["env"] = _state["env"]
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            with open(path, "a") as f:
                f.write(json.dumps({"error": repr(e)[:200], "step": _state["step"]}) + "\n")
        _state["step"] += 1
        return out

    GPUModelRunner.execute_model = execute_model


def _record(runner, so) -> dict:
    ib = runner.input_batch
    num_reqs = getattr(ib, "num_reqs", None) or len(ib.req_ids)
    req_ids = list(ib.req_ids[:num_reqs])
    sched = dict(getattr(so, "num_scheduled_tokens", {}) or {})
    new_hits = {}
    for r in getattr(so, "scheduled_new_reqs", []) or []:
        new_hits[r.req_id] = int(getattr(r, "num_computed_tokens", 0))
    reqs = []
    hashes = _state["hashes"] or {}
    rows = hashes.get("hidden_rows") or []
    am = hashes.get("argmax") or []
    for i, rid in enumerate(req_ids):
        q = int(sched.get(rid, 0))
        computed = int(ib.num_computed_tokens_cpu[i])
        prompt = int(ib.num_prompt_tokens[i]) if hasattr(ib, "num_prompt_tokens") else None
        first = rid not in _state["seen"]
        if first:
            _state["seen"].add(rid)
        reqs.append({"req": rid, "q": q, "computed": computed, "kv": computed + q, "prompt": prompt,
                     "phase": "prefill" if (prompt is not None and computed < prompt) else "decode",
                     "cache_hit": new_hits.get(rid) if first else None,
                     "h": rows[i] if i < len(rows) else None, "argmax": am[i] if i < len(am) else None})
    pc = runner.vllm_config.parallel_config
    try:
        backend = runner.attn_groups[0][0].backend.get_name()
    except Exception:  # noqa: BLE001
        backend = None
    rec = {"step": _state["step"], "rank": _rank(), "total_scheduled": int(getattr(so, "total_num_scheduled_tokens", sum(sched.values()))),
           "num_reqs": len(req_ids), "dispatch": _state["dispatch"], "attention_backend": backend,
           "parallel": {"tp": pc.tensor_parallel_size, "pp": pc.pipeline_parallel_size, "dp": getattr(pc, "data_parallel_size", 1), "ep": bool(getattr(pc, "enable_expert_parallel", False))},
           "moe_expert_counts_first_layer": _state["moe_counts"], "requests": reqs}
    rec["shape_vector"] = shape_vector(rec)
    return rec


def shape_vector(rec: dict) -> str:
    """The integers the hypothesis is about, as a canonical string, content-free."""
    d = rec.get("dispatch") or {}
    parts = [f"pad={d.get('padded_num_tokens')}", f"mode={d.get('cudagraph_mode')}", f"tot={rec['total_scheduled']}", f"n={rec['num_reqs']}",
             f"tp={rec['parallel']['tp']}", f"pp={rec['parallel']['pp']}", f"ep={rec['parallel']['ep']}", f"attn={rec.get('attention_backend')}"]
    seqs = sorted((r["q"], r["kv"], r["computed"], r["phase"]) for r in rec["requests"])
    parts.append("seqs=" + ";".join(f"{q}/{kv}/{c}/{p[0]}" for q, kv, c, p in seqs))
    return "|".join(parts)


def _force_fp8_per_tensor():
    """Arm X2: make vLLM's online FP8 use per-tensor dynamic activation scales.

    fp8.py:310-322 picks per-token dynamic quantisation whenever the CUTLASS
    FP8 GEMM is available and per-tensor otherwise, so the arm disables the
    CUTLASS check. The GEMM then runs through torch._scaled_mm with a
    per-tensor scale, which is the batch statistic the hypothesis predicts
    to couple requests.
    """
    try:
        import vllm.model_executor.layers.quantization.fp8 as fp8
        fp8.cutlass_fp8_supported = lambda: False
        import vllm.model_executor.layers.quantization.utils.w8a8_utils as w
        if hasattr(w, "cutlass_fp8_supported"):
            w.cutlass_fp8_supported = lambda: False
    except Exception:  # noqa: BLE001
        pass


def register():
    if _state["registered"]:
        return
    _state["registered"] = True
    if os.environ.get("SHAPE_FORCE_FP8_PER_TENSOR") == "1":
        _force_fp8_per_tensor()
    if not os.environ.get("SHAPE_HOOK_OUT"):
        return
    _patch_dispatcher()
    _patch_router()
    _patch_runner()
