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

_state = {"registered": False, "step": 0, "moe_counts": None, "dispatch": None, "hashes": None, "batch": None, "pending": None, "seen": set(), "env": None}


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


def _dispatch_record(out) -> dict:
    """Normalise the two dispatcher return shapes: (mode, BatchDescriptor) at the
    pinned sha, BatchExecutionDescriptor(cg_mode, num_tokens, num_reqs) in 0.28.0."""
    try:
        if isinstance(out, tuple):
            mode, desc = out
        else:
            desc, mode = out, getattr(out, "cg_mode", None)
        def _i(v):
            try:
                return int(v)
            except Exception:  # noqa: BLE001
                return str(v)
        return {"cudagraph_mode": str(getattr(mode, "name", mode)),
                "padded_num_tokens": _i(getattr(desc, "num_tokens", None)) if desc is not None else None,
                "padded_num_reqs": _i(getattr(desc, "num_reqs", None)) if desc is not None and hasattr(desc, "num_reqs") else None,
                "uniform_decode": bool(getattr(desc, "uniform_decode", False)) if desc is not None else None}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:80]}


def _dispatch_impl(self, orig, *a, **kw):
    out = orig(self, *a, **kw)
    _state["dispatch"] = _dispatch_record(out)
    return out


def _patch_dispatcher():
    for modname, clsname in (("vllm.v1.cudagraph_dispatcher", "CudagraphDispatcher"), ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager")):
        try:
            mod = __import__(modname, fromlist=[clsname])
            cls = getattr(mod, clsname)
        except Exception:  # noqa: BLE001
            continue
        orig = cls.dispatch

        def dispatch(self, *a, _orig=orig, **kw):
            return _dispatch_impl(self, _orig, *a, **kw)

        cls.dispatch = dispatch


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


def _hook_first_moe(model):
    """Per-expert token counts of the first MoE layer, from the router logits the
    layer receives: top-k of softmax(logits) reproduces the routing of the
    softmax-then-topk models (Qwen1.5-MoE, Mixtral); other routers are
    recorded as 'unsupported'."""
    if getattr(model, "_shape_hook_moe", False):
        return
    model._shape_hook_moe = True
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except Exception:  # noqa: BLE001
        return
    for name, mod in model.named_modules():
        if isinstance(mod, FusedMoE):
            def pre_hook(m, args, kwargs=None, _name=name):
                if _state["moe_counts"] is not None:
                    return
                try:
                    import torch
                    logits = args[1] if len(args) > 1 else (kwargs or {}).get("router_logits")
                    k = int(getattr(m, "top_k", 0) or 0)
                    e = int(getattr(m, "global_num_experts", 0) or logits.shape[-1])
                    if logits is None or k <= 0:
                        _state["moe_counts"] = "unsupported"
                        return
                    scoring = str(getattr(m, "scoring_func", "softmax"))
                    probs = torch.softmax(logits.float(), dim=-1) if scoring == "softmax" else logits.float()
                    topk = torch.topk(probs, k, dim=-1).indices
                    _state["moe_counts"] = torch.bincount(topk.flatten(), minlength=e).tolist()
                except Exception as ex:  # noqa: BLE001
                    _state["moe_counts"] = {"error": repr(ex)[:80]}
            mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
            break


def _patch_fp8_quant():
    """Record the first dynamic per-tensor FP8 activation scale of each pass (arm X2)."""
    try:
        import vllm._custom_ops as ops
    except Exception:  # noqa: BLE001
        return
    orig = ops.scaled_fp8_quant

    def scaled_fp8_quant(*a, **kw):
        out = orig(*a, **kw)
        try:
            if _state.get("fp8_scale") is None and isinstance(out, tuple) and len(out) >= 2 and out[1] is not None and out[1].numel() == 1:
                _state["fp8_scale"] = float(out[1].float().item())
        except Exception:  # noqa: BLE001
            pass
        return out

    ops.scaled_fp8_quant = scaled_fp8_quant


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


def _patch_gather():
    """vLLM 0.28.0 gathers the scheduled batch's CPU state in batch order in
    GPUModelRunner.gather_batch_req_state; capture it, with the computed-token
    counts read at that moment (before the pass runs)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if not hasattr(GPUModelRunner, "gather_batch_req_state"):
        return
    orig = GPUModelRunner.gather_batch_req_state

    def gather(self, scheduler_output, dummy_run, *a, **kw):
        return _gather_impl(self, orig, scheduler_output, dummy_run, *a, **kw)

    global _gather_wrapper
    _gather_wrapper = gather
    GPUModelRunner.gather_batch_req_state = gather


_gather_wrapper = None


def _gather_impl(self, orig, scheduler_output, dummy_run, *a, **kw):
    if True:
        out = orig(self, scheduler_output, dummy_run, *a, **kw)
        try:
            b = out[0] if isinstance(out, tuple) else out
            if b is not None and not dummy_run:
                import numpy as np
                rs = self.req_states
                idx = np.asarray(b.idx_mapping_np)[: len(b.req_ids)]
                computed = np.asarray(rs.num_computed_tokens_np)[idx].tolist()
                _state["batch"] = {"req_ids": list(b.req_ids), "q": np.asarray(b.num_scheduled_tokens)[: len(b.req_ids)].tolist(),
                                   "computed": computed, "prompt": np.asarray(b.prefill_len_np)[: len(b.req_ids)].tolist() if hasattr(b, "prefill_len_np") else None,
                                   "prefilling": np.asarray(b.is_prefilling_np)[: len(b.req_ids)].tolist() if hasattr(b, "is_prefilling_np") else None}
        except Exception as e:  # noqa: BLE001
            _state["batch"] = {"error": repr(e)[:120]}
        return out


def _patch_runner():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig = GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *a, **kw):
        return _execute(self, orig, scheduler_output, *a, **kw)

    GPUModelRunner.execute_model = execute_model
    if hasattr(GPUModelRunner, "sample_tokens"):  # 0.28.0 two-phase step: logits are computed here
        sorig = GPUModelRunner.sample_tokens

        def sample_tokens(self, *a, **kw):
            return _sample_impl(self, sorig, *a, **kw)

        GPUModelRunner.sample_tokens = sample_tokens
    # Something in the worker's initialisation rebinds the class attribute after
    # plugins load (observed on 0.28.0), so the worker entry point also installs
    # the wrapper on the runner instance the first time it runs a batch.
    try:
        from vllm.v1.worker.gpu_worker import Worker
        worig = Worker.execute_model

        def worker_execute_model(self, scheduler_output, *a, **kw):
            mr = getattr(self, "model_runner", None)
            if mr is not None and not getattr(mr, "_shape_hook_instance", False):
                import functools
                bound = mr.execute_model
                if getattr(bound, "__func__", None) is not execute_model:
                    mr.execute_model = functools.partial(_execute, mr, lambda _self, so, *aa, **kk: bound(so, *aa, **kk))
                gb = getattr(mr, "gather_batch_req_state", None)
                if gb is not None and getattr(gb, "__func__", None) is not _gather_wrapper:
                    mr.gather_batch_req_state = functools.partial(_gather_impl, mr, lambda _self, so, dr, *aa, **kk: gb(so, dr, *aa, **kk))
                st = getattr(mr, "sample_tokens", None)
                if st is not None and not getattr(mr, "_shape_hook_sample", False):
                    mr.sample_tokens = functools.partial(_sample_impl, mr, lambda _self, *aa, **kk: st(*aa, **kk))
                    mr._shape_hook_sample = True
                cm = getattr(mr, "cudagraph_manager", None) or getattr(mr, "cudagraph_dispatcher", None)
                if cm is not None and not getattr(cm, "_shape_hook_instance", False):
                    dm = cm.dispatch
                    cm.dispatch = functools.partial(_dispatch_impl, cm, lambda _self, *aa, **kk: dm(*aa, **kk))
                    cm._shape_hook_instance = True
                mr._shape_hook_instance = True
            return worig(self, scheduler_output, *a, **kw)

        Worker.execute_model = worker_execute_model
    except Exception:  # noqa: BLE001
        pass


def _execute(self, orig, scheduler_output, *a, **kw):
    if True:
        path = _out_path()
        if path is None:
            return orig(self, scheduler_output, *a, **kw)
        _state["moe_counts"] = None
        _state["dispatch"] = None
        _state["hashes"] = None
        _state["fp8_scale"] = None
        try:
            _wrap_compute_logits(self.model)
            _hook_first_moe(self.model)
        except Exception:  # noqa: BLE001
            pass
        _flush(path)  # a record left from a pass that had no sampling phase
        t0 = time.time()
        out = orig(self, scheduler_output, *a, **kw)
        try:
            rec = _record(self, scheduler_output)
            rec["wall_ms"] = round((time.time() - t0) * 1000, 1)
            if _state["env"] is None:
                _state["env"] = _env_record()
                rec["env"] = _state["env"]
            _state["pending"] = rec
            if _state["hashes"]:  # logits were computed inside execute_model (pinned-sha layout)
                _flush(path)
        except Exception as e:  # noqa: BLE001
            with open(path, "a") as f:
                f.write(json.dumps({"error": repr(e)[:200], "step": _state["step"]}) + "\n")
        _state["step"] += 1
        return out


def _flush(path: str) -> None:
    rec = _state.get("pending")
    if not rec:
        return
    hashes = _state.get("hashes") or {}
    rows = hashes.get("hidden_rows") or []
    am = hashes.get("argmax") or []
    for i, r in enumerate(rec.get("requests", [])):
        if r.get("h") is None and i < len(rows):
            r["h"] = rows[i]
        if r.get("argmax") is None and i < len(am):
            r["argmax"] = am[i]
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    _state["pending"] = None
    _state["hashes"] = None


def _sample_impl(self, orig, *a, **kw):
    out = orig(self, *a, **kw)
    path = _out_path()
    if path is not None:
        _flush(path)
    return out


def _batch_rows(runner, so) -> list[dict]:
    """(req id, query len, computed, prompt len, prefilling) per batch position."""
    sched = dict(getattr(so, "num_scheduled_tokens", {}) or {})
    b = _state.get("batch")
    if b and "req_ids" in b:  # vLLM 0.28.0 layout
        rows = []
        for i, rid in enumerate(b["req_ids"]):
            q = int(b["q"][i]) if i < len(b["q"]) else int(sched.get(rid, 0))
            computed = int(b["computed"][i])
            prompt = int(b["prompt"][i]) if b.get("prompt") is not None else None
            prefilling = bool(b["prefilling"][i]) if b.get("prefilling") is not None else (prompt is not None and computed < prompt)
            rows.append({"req": rid, "q": q, "computed": computed, "prompt": prompt, "prefilling": prefilling})
        return rows
    ib = runner.input_batch  # pinned-sha layout (5769a7382cb1)
    num_reqs = getattr(ib, "num_reqs", None) or len(ib.req_ids)
    rows = []
    for i, rid in enumerate(list(ib.req_ids[:num_reqs])):
        computed = int(ib.num_computed_tokens_cpu[i])
        prompt = int(ib.num_prompt_tokens[i]) if hasattr(ib, "num_prompt_tokens") else None
        rows.append({"req": rid, "q": int(sched.get(rid, 0)), "computed": computed, "prompt": prompt, "prefilling": prompt is not None and computed < prompt})
    return rows


def _record(runner, so) -> dict:
    new_hits = {}
    for r in getattr(so, "scheduled_new_reqs", []) or []:
        new_hits[r.req_id] = int(getattr(r, "num_computed_tokens", 0))
    reqs = []
    hashes = _state["hashes"] or {}
    rows = hashes.get("hidden_rows") or []
    am = hashes.get("argmax") or []
    batch_rows = _batch_rows(runner, so)
    req_ids = [r["req"] for r in batch_rows]
    sched = dict(getattr(so, "num_scheduled_tokens", {}) or {})
    for i, br in enumerate(batch_rows):
        rid = br["req"]
        first = rid not in _state["seen"]
        if first:
            _state["seen"].add(rid)
        reqs.append({"req": rid, "q": br["q"], "computed": br["computed"], "kv": br["computed"] + br["q"], "prompt": br["prompt"],
                     "phase": "prefill" if br["prefilling"] else "decode",
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
           "moe_expert_counts_first_layer": _state["moe_counts"], "fp8_scale_first_call": _state.get("fp8_scale"), "requests": reqs}
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
    _patch_gather()
    _patch_fp8_quant()
    _patch_runner()
