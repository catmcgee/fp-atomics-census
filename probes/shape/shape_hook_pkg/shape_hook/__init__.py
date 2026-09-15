"""Batch-shape instrumentation for vLLM, installed as a general plugin.

Loaded by vLLM in the engine core and in every worker process through the
``vllm.general_plugins`` entry point (vllm/plugins/__init__.py,
vllm/v1/worker/worker_base.py:245-247), so it works with tensor
parallelism. It stays outside the engine source: it monkeypatches four
call sites of vLLM 0.28.0 (line references are to the former census pin,
5769a7382cb1):

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

Record schema 3 (line references are to the vLLM 0.28.0 release tag). Every
schema 2 field keeps its name and meaning; schema 3 adds, per row:

* ``new_token_ids``: the token ids the row consumes in this pass, positions
  ``computed .. computed+q-1`` of the request's token sequence. vLLM 0.28.0
  has two GPU model runners, selected by ``VllmConfig.use_v2_model_runner``
  (vllm/config/vllm.py:615-664; the worker picks the class at
  vllm/v1/worker/gpu_worker.py:424-438). The ids are read from the runner's
  own per-request host state after the pass: on the V1 runner from
  ``self.requests[req_id]`` (``dict[str, CachedRequestState]``,
  gpu_model_runner.py:725; ``CachedRequestState.get_token_id`` joins
  ``prompt_token_ids`` and ``output_token_ids``, gpu_input_batch.py:79-89,
  and returns -1 beyond the known tokens); on the V2 runner from
  ``self.req_states.all_token_ids`` (a host-visible UVA tensor of shape
  ``[max_num_reqs, max_model_len]``, vllm/v1/worker/gpu/states.py:34-39,
  indexed through ``req_id_to_index``, states.py:27; the sampled token of
  the previous pass was written into it by ``postprocess_sampled``,
  vllm/v1/worker/gpu/model_runner.py:1367-1391). A row whose ids cannot be
  read, whose count is not ``q``, that contains a negative id (the V1
  runner's async-scheduling placeholder, gpu_model_runner.py:3900 and
  4993-4994), or whose prompt positions disagree with the admitted
  ``prompt_token_ids`` carries ``invalid`` with a reason.
* ``prompt_sha256``: SHA-256 over the request's full prompt token ids as
  unsigned 32-bit little-endian integers, computed once per request from
  ``NewRequestData.prompt_token_ids`` (vllm/v1/core/sched/output.py:35-37)
  and cached until the request finishes.

and per record:

* ``admitted``: requests scheduled for the first time in this pass, taken
  from ``SchedulerOutput.scheduled_new_reqs`` (output.py:194-197), each with
  its full ``prompt_token_ids``, ``num_computed_tokens`` (the prefix-cache
  hit, output.py:42) and ``prompt_sha256``. The prompt ids are recorded once,
  in the forward record of the admitting pass.
* ``resumed``: requests rescheduled after preemption. The V1 path lists them
  in ``scheduled_cached_reqs.resumed_req_ids`` (output.py:121,
  scheduler.py:1508-1509); the V2 path folds them into
  ``scheduled_new_reqs`` (scheduler.py:1195-1204), so a request in
  ``scheduled_new_reqs`` that this hook has seen before is resumed, not
  admitted.
* ``finished``: ``SchedulerOutput.finished_req_ids``, the requests that
  finished between the previous and this scheduler call (output.py:221-224).
* ``preempted``: ``SchedulerOutput.preempted_req_ids``. The field comment
  says it is used only by the V2 runner (output.py:231-233), but the
  scheduler fills it for both runners from ``reset_preempted_req_ids``
  (scheduler.py:1279, added in ``_preempt_request`` at scheduler.py:1377 and
  cleared at scheduler.py:1427). Preemption is therefore observable in
  0.28.0 through this field.
* ``model_runner`` (``"v1"`` or ``"v2"``) and ``scheduler`` (``max_num_seqs``,
  ``max_num_batched_tokens``, ``enable_chunked_prefill``,
  ``async_scheduling``), so a replay can check it ran the same runner and
  scheduler limits.

Empty scheduler calls (``total_num_scheduled_tokens == 0``) keep their
``event: "empty_scheduler_call"`` record and additionally carry ``finished``
and ``preempted``, because a request that finishes in the last forward pass
is reported in the following scheduler call.

vLLM 0.29.0 (release tag v0.29.0 points to the commit of 8 September 2026;
line numbers are from the current census pin, 98dff2a81d74, while the
references above are to 5769a7382cb1). Every call site and field read above keeps its name,
signature and meaning; only line numbers move. V1 runner: ``execute_model``
at gpu_model_runner.py:4249, ``sample_tokens`` at :4628, the
``compute_logits`` call at :4560, ``self.requests`` at :725,
``get_token_id`` at gpu_input_batch.py:79. V2 runner:
``gather_batch_req_state`` at vllm/v1/worker/gpu/model_runner.py:1106, with
``BatchReqState`` and ``vllm/v1/worker/gpu/states.py`` byte-identical to
0.28.0. ``CudagraphDispatcher.dispatch``, the V2 ``CudaGraphManager.dispatch``
return type, ``FusedMoERouter.select_experts``, the scheduler output fields
(output.py:36-49, 137, 223-259), the logits-processor interface and loader,
the sampler and the ``vllm.general_plugins`` loader are unchanged, the last
six byte for byte. Two things differ:

* the default runner. 0.29.0 selects V2 unless a feature V2 lacks is
  configured (vllm/config/vllm.py:645-676 and :2539-2612); 0.28.0 first
  required a dense generate model or one of eight listed architectures
  (0.28.0 vllm.py:69-93 and :615-712). That list already held
  ``Qwen2MoeForCausalLM``, so the census's own models resolve the same
  runner on both releases. Custom logits processors remain a V2 gap (0.29.0
  vllm.py:2606), so E6 with its processor resolves V1 on both, and 0.29.0
  raises rather than falling back when a V1-only feature meets a V2-only one
  (vllm.py:2614-2644, :2735). ``model_runner`` still says which ran.
* batch-sharded sampling, new and opt-in on the V2 runner
  (``ParallelConfig.enable_batch_sharded_sampling``, default off), computes
  logits through ``compute_logits_local`` (gpu/model_runner.py:1405-1418),
  which this hook does not wrap; a record in that configuration is refused
  with an error line instead of being written without hashes.

Additive per-record fields, no schema change:

* ``linear_quant``: the linear method, activation quantisation key and GEMM
  kernel class of the first quantised linear layer
  (``quant_method.activation_quant_key`` and ``quant_method.fp8_linear``,
  the same attributes in 0.28.0 and 0.29.0), or the first linear layer's
  method when none is quantised. It records the resolved FP8 scheme that
  earlier campaigns could only infer, which matters because 0.29.0 reorders
  the CUDA FP8 kernel priority list, moving Marlin from first to sixth
  (model_executor/kernels/linear/__init__.py:402-412).
* ``fp8_forcing``: which modules the per-tensor arm rebound, or null when
  the arm was not requested.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
import subprocess
import time

SCHEMA = 3

_state = {"registered": False, "step": 0, "forward_passes": 0, "moe_counts": None, "dispatch": None, "hashes": None, "batch": None, "pending": None, "seen": set(), "prompts": {}, "env": None, "linear_quant": None, "fp8_forcing": None}


def _env_record() -> dict:
    import torch
    try:
        import importlib.metadata as md
        pk = {p: md.version(p) for p in ("vllm", "flashinfer-python", "torch", "triton") if _has(md, p)}
    except Exception:  # noqa: BLE001
        pk = {}
    rec = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "packages": pk,
           "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "VLLM_ATTENTION_BACKEND", "VLLM_MARLIN_USE_ATOMIC_ADD", "CUBLAS_WORKSPACE_CONFIG", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER",
                                                  "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_DISABLE_COMPILE_CACHE", "TORCHINDUCTOR_DETERMINISTIC")}}
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
    header = json.dumps({"dtype": str(t.dtype), "shape": list(t.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + b).hexdigest()


def prompt_sha256(token_ids) -> str:
    """SHA-256 over token ids packed as unsigned 32-bit little-endian integers.

    Raises on anything that is not a sequence of ids in [0, 2**32); a prompt
    digest is never computed from placeholders or embeddings.
    """
    ids = [int(t) for t in token_ids]
    if any(t < 0 or t >= 2**32 for t in ids):
        raise ValueError("token id outside the unsigned 32-bit range")
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def row_validity(new_token_ids, q: int, computed: int, prompt_token_ids) -> str | None:
    """Reason a schema 3 row is invalid, or None.

    ``prompt_token_ids`` is the admitted prompt (None when unknown); the
    positions of the row that fall inside the prompt must reproduce it.
    """
    if new_token_ids is None:
        return "token ids unavailable from the runner's per-request state"
    if not isinstance(new_token_ids, list) or len(new_token_ids) != q:
        return f"len(new_token_ids) {None if new_token_ids is None else len(new_token_ids)} != q {q}"
    if any((not isinstance(t, int)) or t < 0 for t in new_token_ids):
        return "negative or non-integer token id (placeholder for a token not materialised on the host)"
    if prompt_token_ids is None:
        return "prompt token ids unavailable for this request"
    inside = max(0, min(computed + q, len(prompt_token_ids)) - computed)
    if inside > 0 and new_token_ids[:inside] != [int(t) for t in prompt_token_ids[computed:computed + inside]]:
        return "new_token_ids disagree with the admitted prompt token ids"
    return None


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
            if _state.get("moe_source") != "actual_router":
                import torch
                topk_ids = out[1]
                n = int(getattr(self, "global_num_experts", 0) or getattr(getattr(self, "moe", None), "num_experts", 0) or int(topk_ids.max().item()) + 1)
                _state["moe_counts"] = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=n).tolist()
                _state["moe_source"] = "actual_router"
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
                    _state["moe_source"] = "estimated_from_router_logits"
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
            if not bool(torch.isfinite(hs).all()):
                raise ValueError("nonfinite hidden states")
            if logits is not None and (bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()) or not bool(torch.isfinite(logits).any(dim=-1).all())):
                raise ValueError("invalid logits (NaN, positive infinity or no finite token)")
            rows = [_row_hash(hs[i]) for i in range(hs.shape[0])]
            am = torch.argmax(logits.float(), dim=-1).tolist() if logits is not None else None
            _state["hashes"] = {"hidden_rows": rows, "argmax": am,
                                "logit_rows": [_row_hash(logits[i]) for i in range(logits.shape[0])] if logits is not None else []}
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


def _ids(values) -> list[str]:
    return sorted(str(v) for v in (values or []))


def _execute(self, orig, scheduler_output, *a, **kw):
    if True:
        path = _out_path()
        if path is None:
            return orig(self, scheduler_output, *a, **kw)
        _flush(path)  # preserve the previous pass before clearing step-local state
        for key in ("moe_counts", "moe_source", "dispatch", "hashes", "fp8_scale", "batch"):
            _state[key] = None
        if int(getattr(scheduler_output, "total_num_scheduled_tokens", sum(getattr(scheduler_output, "num_scheduled_tokens", {}).values()))) == 0:
            out = orig(self, scheduler_output, *a, **kw)
            with open(path, "a") as f:
                f.write(json.dumps({"schema": SCHEMA, "event": "empty_scheduler_call", "step": _state["step"],
                                    "total_scheduled": 0, "requests": [],
                                    "finished": _ids(getattr(scheduler_output, "finished_req_ids", None)),
                                    "preempted": _ids(getattr(scheduler_output, "preempted_req_ids", None))}) + "\n")
            _forget_finished(scheduler_output)
            _state["step"] += 1
            return out
        try:
            _wrap_compute_logits(self.model)
            _hook_first_moe(self.model)
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
            _state["pending"] = rec
            if not all(str(r["req"]).startswith("_warmup") for r in rec["requests"]):
                _state["forward_passes"] += 1
            if _state["hashes"]:  # logits were computed inside execute_model (pinned-sha layout)
                _flush(path)
        except Exception as e:  # noqa: BLE001
            with open(path, "a") as f:
                f.write(json.dumps({"error": repr(e)[:200], "step": _state["step"]}) + "\n")
        _forget_finished(scheduler_output)
        _state["step"] += 1
        return out


def _forget_finished(scheduler_output) -> None:
    for rid in getattr(scheduler_output, "finished_req_ids", None) or []:
        _state["prompts"].pop(str(rid), None)


def _flush(path: str) -> None:
    rec = _state.get("pending")
    if not rec:
        return
    hashes = _state.get("hashes") or {}
    rows = hashes.get("hidden_rows") or []
    am = hashes.get("argmax") or []
    logits = hashes.get("logit_rows") or []
    for i, r in enumerate(rec.get("requests", [])):
        if r.get("h") is None and i < len(rows):
            r["h"] = rows[i]
        if r.get("argmax") is None and i < len(am):
            r["argmax"] = am[i]
        if i < len(logits):
            r["logits_h"] = logits[i]
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
    if b and "req_ids" in b:  # vLLM 0.28.0 V2 runner layout
        rows = []
        for i, rid in enumerate(b["req_ids"]):
            q = int(b["q"][i]) if i < len(b["q"]) else int(sched.get(rid, 0))
            computed = int(b["computed"][i])
            prompt = int(b["prompt"][i]) if b.get("prompt") is not None else None
            prefilling = bool(b["prefilling"][i]) if b.get("prefilling") is not None else (prompt is not None and computed < prompt)
            rows.append({"req": rid, "q": q, "computed": computed, "prompt": prompt, "prefilling": prefilling})
        return rows
    ib = runner.input_batch  # pinned-sha and 0.28.0 V1 runner layout (gpu_input_batch.py:165-172)
    num_reqs = getattr(ib, "num_reqs", None) or len(ib.req_ids)
    rows = []
    for i, rid in enumerate(list(ib.req_ids[:num_reqs])):
        computed = int(ib.num_computed_tokens_cpu[i])
        prompt = int(ib.num_prompt_tokens[i]) if hasattr(ib, "num_prompt_tokens") else None
        rows.append({"req": rid, "q": int(sched.get(rid, 0)), "computed": computed, "prompt": prompt, "prefilling": prompt is not None and computed < prompt})
    return rows


def _model_runner_layout(runner) -> str:
    """'v2' for vllm/v1/worker/gpu/model_runner.py (has req_states), else 'v1'."""
    return "v2" if hasattr(runner, "req_states") else "v1"


def _row_token_ids(runner, rid: str, computed: int, q: int) -> list[int] | None:
    """Token ids at positions computed..computed+q-1 from the runner's host state.

    V1: gpu_model_runner.py:725 ``self.requests`` and gpu_input_batch.py:79-89
    ``get_token_id`` (returns -1 for unknown positions, which row_validity
    rejects). V2: vllm/v1/worker/gpu/states.py:34-39 ``all_token_ids`` (UVA,
    host-visible) indexed by states.py:27 ``req_id_to_index``; the device is
    synchronised first so the previous pass's sampled token write
    (vllm/v1/worker/gpu/model_runner.py:1367-1391) is complete.
    """
    if _model_runner_layout(runner) == "v2":
        rs = runner.req_states
        idx = rs.req_id_to_index.get(rid)
        if idx is None:
            return None
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return [int(t) for t in rs.all_token_ids.gpu[idx, computed:computed + q].to("cpu").tolist()]
    req = getattr(runner, "requests", {}).get(rid)
    if req is None:
        return None
    return [int(req.get_token_id(p)) for p in range(computed, computed + q)]


def linear_quant(model) -> dict:
    """Resolved linear method of the first quantised linear layer, else the first linear layer's method.

    Reads ``quant_method`` and, when present, ``quant_method.activation_quant_key`` and
    ``quant_method.fp8_linear`` (0.28.0 and 0.29.0 quantization/fp8.py and online/fp8.py).
    """
    first = None
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None or not hasattr(qm, "apply") or "Linear" not in type(qm).__name__:
            continue
        rec = {"layer": name, "method": type(qm).__name__,
               "activation_quant_key": str(qm.activation_quant_key) if getattr(qm, "activation_quant_key", None) is not None else None,
               "kernel": type(qm.fp8_linear).__name__ if getattr(qm, "fp8_linear", None) is not None else None}
        if type(qm).__name__ != "UnquantizedLinearMethod":
            return rec
        first = first or rec
    return first


def _linear_quant(runner):
    if _state.get("linear_quant") is None:
        try:
            model = runner.get_model() if hasattr(runner, "get_model") else runner.model
            _state["linear_quant"] = linear_quant(model)
        except Exception as e:  # noqa: BLE001
            _state["linear_quant"] = {"error": repr(e)[:80]}
    return _state["linear_quant"]


def _scheduler_limits(runner) -> dict:
    sc = getattr(runner.vllm_config, "scheduler_config", None)
    return {k: getattr(sc, k, None) for k in ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill", "async_scheduling")}


def _record(runner, so) -> dict:
    new_hits = {}
    admitted, resumed = [], []
    for r in getattr(so, "scheduled_new_reqs", []) or []:
        rid = str(r.req_id)
        new_hits[rid] = int(getattr(r, "num_computed_tokens", 0))
        if rid in _state["seen"]:
            resumed.append(rid)  # V2 folds resumed requests into scheduled_new_reqs (scheduler.py:1195-1204)
            continue
        ids = getattr(r, "prompt_token_ids", None)
        ids = [int(t) for t in ids] if ids is not None else None
        digest = prompt_sha256(ids) if ids is not None else None
        _state["prompts"][rid] = {"ids": ids, "sha256": digest}
        admitted.append({"req": rid, "prompt_token_ids": ids, "prompt_len": len(ids) if ids is not None else None,
                         "num_computed_tokens": new_hits[rid], "prompt_sha256": digest})
    cached = getattr(so, "scheduled_cached_reqs", None)
    for rid in getattr(cached, "resumed_req_ids", None) or []:
        if str(rid) not in resumed:
            resumed.append(str(rid))
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
        row = {"req": rid, "q": br["q"], "computed": br["computed"], "kv": br["computed"] + br["q"], "prompt": br["prompt"],
               "phase": "prefill" if br["prefilling"] else "decode",
               "cache_hit": new_hits.get(rid) if first else None,
               "h": rows[i] if i < len(rows) else None, "argmax": am[i] if i < len(am) else None}
        if not str(rid).startswith("_warmup"):
            prompt = _state["prompts"].get(rid) or {"ids": None, "sha256": None}
            try:
                ids = _row_token_ids(runner, rid, int(br["computed"]), int(br["q"]))
            except Exception as e:  # noqa: BLE001
                ids, reason = None, f"token id read failed: {repr(e)[:80]}"
            else:
                reason = None
            row["new_token_ids"] = ids
            row["prompt_sha256"] = prompt["sha256"]
            invalid = reason or row_validity(ids, int(br["q"]), int(br["computed"]), prompt["ids"])
            if invalid:
                row["invalid"] = invalid
        reqs.append(row)
    if set(req_ids) != set(sched) or sum(r["q"] for r in reqs) != sum(sched.values()):
        raise ValueError("captured batch does not match scheduled requests and token counts")
    cc = runner.vllm_config.compilation_config
    actual_compile = int(cc.mode) != 0
    actual_graphs = str(getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)) != "NONE"
    for env_key, actual in (("SHAPE_EXPECT_COMPILE", actual_compile), ("SHAPE_EXPECT_GRAPHS", actual_graphs)):
        if env_key in os.environ and actual != bool(int(os.environ[env_key])):
            raise ValueError(f"worker resolved configuration contradicts {env_key}")
    pc = runner.vllm_config.parallel_config
    if getattr(pc, "enable_batch_sharded_sampling", False):
        # 0.29.0 V2: logits come from compute_logits_local, which the hook does not wrap.
        raise ValueError("batch-sharded sampling bypasses the compute_logits wrapper; not recorded")
    try:
        backend = runner.attn_groups[0][0].backend.get_name()
    except Exception:  # noqa: BLE001
        backend = None
    rec = {"schema": SCHEMA, "event": "forward", "run_id": os.environ.get("SHAPE_RUN_ID"),
           "hash_schema": "sha256-dtype-shape-bytes", "resolved_compile": str(cc.mode), "resolved_cudagraph": str(cc.cudagraph_mode),
           "step": _state["step"], "rank": _rank(), "total_scheduled": int(getattr(so, "total_num_scheduled_tokens", sum(sched.values()))),
           "num_reqs": len(req_ids), "dispatch": _state["dispatch"], "attention_backend": backend,
           "parallel": {"tp": pc.tensor_parallel_size, "pp": pc.pipeline_parallel_size, "dp": getattr(pc, "data_parallel_size", 1), "ep": bool(getattr(pc, "enable_expert_parallel", False))},
           "model_runner": _model_runner_layout(runner), "scheduler": _scheduler_limits(runner),
           "admitted": admitted, "resumed": sorted(resumed),
           "finished": _ids(getattr(so, "finished_req_ids", None)), "preempted": _ids(getattr(so, "preempted_req_ids", None)),
           "moe_counts_source": _state.get("moe_source"), "moe_expert_counts_first_layer": _state["moe_counts"], "fp8_scale_first_call": _state.get("fp8_scale"),
           "linear_quant": _linear_quant(runner), "fp8_forcing": _state.get("fp8_forcing"), "requests": reqs}
    rec["shape_vector"] = shape_vector(rec)
    return rec


def shape_vector(rec: dict) -> str:
    """Ordered recorded shape. No assertion that this exhausts planner state."""
    return json.dumps({"schema": 2, "total": rec["total_scheduled"], "num_reqs": rec["num_reqs"],
                       "dispatch": rec.get("dispatch"), "parallel": rec["parallel"],
                       "attention_backend": rec.get("attention_backend"),
                       "ordered_rows": [[r[k] for k in ("q", "kv", "computed", "phase")] for r in rec["requests"]]},
                      sort_keys=True, separators=(",", ":"))


def _force_fp8_per_tensor():
    """Arm X2: make vLLM's online FP8 use per-tensor dynamic activation scales.

    In the census's pinned vLLM source, the online FP8 linear method
    (quantization/online/fp8.py, Fp8PerTensorOnlineLinearMethod) picks
    per-token dynamic activation scales when cutlass_fp8_supported() returns
    True and per-tensor scales otherwise; the per-tensor scale is the batch
    statistic the hypothesis predicts to couple requests, and the arm
    patches that check. Both files are byte-identical in 0.28.0 and 0.29.0,
    and both take the function by name (``from ... w8a8_utils import
    cutlass_fp8_supported``), so rebinding it in ``w8a8_utils`` alone does
    not reach a module that has already imported it. This rebinds the name
    in every vLLM module that holds it, ``quantization.online.fp8``
    included, and fails closed when it finds none: the 15 September
    per-tensor attempt ran vLLM's per-token default and nothing detected
    that the forcing had not taken effect. Runners must still set
    SHAPE_FORCE_FP8_PER_TENSOR before register(), which reads it once. The
    GEMM kernel is chosen separately; on H100 with vLLM 0.28.0 the engine
    selected CutlassFP8ScaledMMLinearKernel, and 0.29.0 reorders that
    priority list (model_executor/kernels/linear/__init__.py:402-416), so
    the resolved scheme is recorded per pass in ``linear_quant`` rather than
    inferred. Under compilation the hook observes no scale, so the scale
    field alone does not attest a compiled forced arm; ``linear_quant`` does.
    """
    import sys as _sys
    def forced() -> bool:
        return False
    patched, preloaded = [], []
    for name in ("vllm.model_executor.layers.quantization.utils.w8a8_utils",
                 "vllm.model_executor.layers.quantization.fp8",
                 "vllm.model_executor.layers.quantization.online.fp8"):
        if name in _sys.modules:
            preloaded.append(name)
        try:
            mod = __import__(name, fromlist=["cutlass_fp8_supported"])
        except Exception:  # noqa: BLE001  not every build carries every module
            continue
        if hasattr(mod, "cutlass_fp8_supported"):
            mod.cutlass_fp8_supported = forced
            patched.append(name)
    for name, mod in list(_sys.modules.items()):
        if name.startswith("vllm.") and name not in patched and getattr(mod, "cutlass_fp8_supported", None) is not None:
            mod.cutlass_fp8_supported = forced
            patched.append(name)
    _state["fp8_forcing"] = {"patched_modules": sorted(patched), "already_imported": sorted(preloaded)}
    if not patched:
        raise RuntimeError("SHAPE_FORCE_FP8_PER_TENSOR is set but no vLLM module exposes cutlass_fp8_supported; "
                           "the arm would otherwise run the per-token default unnoticed")


def flush() -> None:
    """Write any buffered record; runners call this after each generate call in-process."""
    path = _out_path()
    if path is not None:
        _flush(path)


def forward_passes() -> int:
    """Number of non-warmup forward passes recorded so far in this process."""
    return int(_state["forward_passes"])


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
