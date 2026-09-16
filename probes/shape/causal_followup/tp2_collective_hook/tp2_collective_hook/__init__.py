"""Trace the actual backend selected by every pinned vLLM TP all-reduce call.

This plugin changes no collective result.  It wraps the unmodified dispatcher
and its backend objects, then records which nested backend returned the result.
It is deliberately pinned to the two vLLM 0.29 source files reviewed for this
experiment and refuses to load against other bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = 1
PINNED = {
    "cuda_communicator.py": "03ece681b0bef349cb78dabc74585efea2cc160ae28bd711c81dfd807ca8dff5",
    "flashinfer_all_reduce.py": "607a9090f141afab59bbac73d2f6d06f557b8b13e150c2af9bc06d66d2ef3825",
}
_lock = threading.Lock()
_registered = False


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    return int(os.environ.get("RANK", "0"))


def _write(event: dict[str, Any]) -> None:
    root = os.environ.get("TP2_COLLECTIVE_TRACE_DIR")
    if not root:
        return
    path = Path(root) / f"rank{_rank()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": SCHEMA,
        "run_id": os.environ.get("TP2_COLLECTIVE_RUN_ID"),
        "pid": os.getpid(),
        "rank": _rank(),
        "time_ns": time.time_ns(),
        **event,
    }
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(input_: Any) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in input_.shape],
        "dtype": str(input_.dtype),
        "device": str(input_.device),
        "contiguous": bool(input_.is_contiguous()),
        "numel": int(input_.numel()),
        "nbytes": int(input_.nbytes),
    }


def _temporary_wrapper(obj: Any, name: str, backend: str, selected: list[str]):
    """Install an instance wrapper and return an exact restoration callback."""
    namespace = getattr(obj, "__dict__", {})
    existed = name in namespace
    previous = namespace.get(name)
    bound = getattr(obj, name)

    def wrapper(*args, **kwargs):
        output = bound(*args, **kwargs)
        if output is not None:
            selected.append(backend)
        return output

    setattr(obj, name, wrapper)

    def restore() -> None:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)

    return restore


def _patch_dispatch(cuda_module: Any) -> None:
    cls = cuda_module.CudaCommunicator
    if getattr(cls, "_tp2_collective_trace", False):
        return
    original = cls.all_reduce

    def all_reduce(self, input_):
        selected: list[str] = []
        restores: list[Callable[[], None]] = []
        candidates = (
            (getattr(self, "fi_ar_comm", None), "all_reduce", "FLASHINFER"),
            (getattr(self, "qr_comm", None), "quick_all_reduce", "QUICK_REDUCE"),
            (getattr(self, "aiter_ar_comm", None), "custom_all_reduce", "AITER_CUSTOM"),
            (getattr(self, "ca_comm", None), "custom_all_reduce", "CUSTOM"),
            (getattr(self, "symm_mem_comm", None), "all_reduce", "SYMM_MEM"),
            (getattr(self, "pynccl_comm", None), "all_reduce", "PYNCCL"),
        )
        nccl_symm = {"accepted": False}
        prior_gate = cuda_module.should_nccl_symm_mem_allreduce

        def gate(*args, **kwargs):
            accepted = prior_gate(*args, **kwargs)
            nccl_symm["accepted"] = nccl_symm["accepted"] or bool(accepted)
            return accepted

        try:
            for obj, name, backend in candidates:
                if obj is not None and hasattr(obj, name):
                    restores.append(_temporary_wrapper(obj, name, backend, selected))
            cuda_module.should_nccl_symm_mem_allreduce = gate
            output = original(self, input_)
            backend = selected[-1] if selected else (
                "NCCL_SYMM_MEM" if nccl_symm["accepted"] else "TORCH_DIST"
            )
            _write(
                {
                    "event": "dispatch",
                    "backend": backend,
                    "nested_backends": selected,
                    "group": self.unique_name or "<unnamed>",
                    "world_size": int(self.world_size),
                    "tensor": _tensor(input_),
                }
            )
            return output
        except BaseException as exc:
            _write(
                {
                    "event": "dispatch_error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "group": self.unique_name or "<unnamed>",
                    "world_size": int(self.world_size),
                    "tensor": _tensor(input_),
                }
            )
            raise
        finally:
            cuda_module.should_nccl_symm_mem_allreduce = prior_gate
            for restore in reversed(restores):
                restore()

    cls.all_reduce = all_reduce
    cls._tp2_collective_trace = True


def _patch_workspace(fi_module: Any) -> None:
    if getattr(fi_module, "_tp2_collective_trace", False):
        return
    original = fi_module._create_workspace

    def create(backend, world_size, rank, max_token_num, hidden_dim, dtype, group):
        workspace = original(
            backend, world_size, rank, max_token_num, hidden_dim, dtype, group
        )
        actual = getattr(workspace, "backend", None) if workspace is not None else None
        _write(
            {
                "event": "workspace",
                "requested_backend": str(backend),
                "actual_backend": None if actual is None else str(actual),
                "created": workspace is not None,
                "world_size": int(world_size),
                "group_rank": int(rank),
                "max_token_num": int(max_token_num),
                "hidden_dim": int(hidden_dim),
                "dtype": str(dtype),
            }
        )
        return workspace

    fi_module._create_workspace = create
    fi_module._tp2_collective_trace = True


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    trace_dir = os.environ.get("TP2_COLLECTIVE_TRACE_DIR")
    if not trace_dir:
        return
    expected = os.environ.get("TP2_COLLECTIVE_EXPECT_BACKEND")
    configured = os.environ.get("VLLM_FLASHINFER_ALLREDUCE_BACKEND")
    if expected not in {"trtllm", "mnnvl"} or configured != expected:
        raise RuntimeError(
            "TP2 collective trace requires equal explicit "
            "TP2_COLLECTIVE_EXPECT_BACKEND and VLLM_FLASHINFER_ALLREDUCE_BACKEND"
        )
    import vllm.distributed.device_communicators.cuda_communicator as cuda_module
    import vllm.distributed.device_communicators.flashinfer_all_reduce as fi_module

    files = {
        "cuda_communicator.py": Path(cuda_module.__file__).resolve(),
        "flashinfer_all_reduce.py": Path(fi_module.__file__).resolve(),
    }
    actual = {name: _sha256(path) for name, path in files.items()}
    if actual != PINNED:
        raise RuntimeError(f"unreviewed vLLM collective source bytes: {actual!r}")
    _patch_workspace(fi_module)
    _patch_dispatch(cuda_module)
    _write(
        {
            "event": "registered",
            "expected_backend": expected,
            "configured_backend": configured,
            "source_sha256": actual,
        }
    )
