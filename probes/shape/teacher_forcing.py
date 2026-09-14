"""Teacher forcing for vLLM 0.28.0 through its V1 logits-processor interface.

The processor is passed as a type to ``LLM(logits_processors=[...])``
(vllm/entrypoints/llm.py:220 and :334 forward it to ``ModelConfig.logits_processors``,
vllm/config/model.py:351). The V1 GPU model runner builds one instance per
worker with ``build_logitsprocs`` (vllm/v1/worker/gpu_model_runner.py:739-775;
vllm/v1/sample/logits_processor/__init__.py:185-218 accepts already-loaded
subclasses of ``LogitsProcessor`` at :119-125). The V2 runner
(vllm/v1/worker/gpu/model_runner.py) has no custom logits-processor support:
``custom logits processors`` is listed among its unsupported features
(vllm/config/vllm.py:2459-2467), which makes ``use_v2_model_runner`` fall back
to V1 (vllm.py:657-664) or, when ``VLLM_USE_V2_MODEL_RUNNER=1`` is forced,
raise (vllm.py:2478-2487). E6 therefore runs both the recorded and the replayed
arm with this processor configured and with the V1 runner, and the hook records
``model_runner`` so the comparison can check it.

Per-request state arrives through ``update_state(BatchUpdate)``: the persistent
batch appends ``(index, sampling_params, prompt_token_ids, output_token_ids)``
for every added request (vllm/v1/worker/gpu_input_batch.py:339-345) and the
runner delivers it before each pass (gpu_input_batch.py:840-856). The
``output_token_ids`` element is a reference to the request's live output list
(vllm/v1/sample/logits_processor/interface.py:44-48), so its length is the
number of tokens sampled so far. The forced continuation travels in
``SamplingParams.extra_args`` (vllm/sampling_params.py:345-348, reserved for
plugins), under the key ``EXTRA_ARGS_KEY``.

``apply`` runs inside ``Sampler.apply_logits_processors`` on a float32 copy of
the logits before greedy sampling (vllm/v1/sample/sampler.py:96-103 and
:400-402; ``greedy_sample`` at :261 reads the processed tensor), and after the
raw top-k logprobs were taken from the unprocessed logits (sampler.py:81-95),
so the reported logprobs and the hook's ``argmax`` (taken in
``compute_logits``) stay free-running observations while the sampled token is
forced. The row of a request keeps the logit of the recorded next token and
every other logit becomes the most negative finite value of the dtype, so
argmax selects the recorded token. Each call appends one line to
``$SHAPE_HOOK_OUT/forcing_rank<r>.jsonl`` with, per forced row, the forced
token, the argmax before forcing and whether they agree.

Limits: forcing works for one sampled token per request per pass (no
speculative decoding, which 0.28.0 also rejects with custom processors,
logits_processor/__init__.py:201-204); the processor fails closed when a
request has more sampled tokens than forced tokens, when an output token id is
the -1 placeholder the V1 runner uses under async scheduling
(gpu_model_runner.py:3900), or when the forced token's logit is not finite.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

try:  # the real interface when vLLM is installed
    from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor, MoveDirectionality
    VLLM_INTERFACE = True
except Exception:  # noqa: BLE001  vLLM absent: a stub with the same surface, for CPU tests
    VLLM_INTERFACE = False

    class MoveDirectionality(Enum):  # type: ignore[no-redef]
        UNIDIRECTIONAL = auto()
        SWAP = auto()

    @dataclass(frozen=True)
    class BatchUpdate:  # type: ignore[no-redef]
        batch_size: int
        removed: Sequence[int]
        added: Sequence[tuple]
        moved: Sequence[tuple]

    class LogitsProcessor:  # type: ignore[no-redef]
        """Stub of vllm/v1/sample/logits_processor/interface.py:60-108."""

        @classmethod
        def validate_params(cls, sampling_params):
            return None

        def __init__(self, vllm_config, device, is_pin_memory) -> None:
            raise NotImplementedError

        def apply(self, logits):
            raise NotImplementedError

        def is_argmax_invariant(self) -> bool:
            raise NotImplementedError

        def update_state(self, batch_update) -> None:
            raise NotImplementedError


EXTRA_ARGS_KEY = "shape_forced_token_ids"


def forced_tokens_of(sampling_params) -> list[int] | None:
    """The forced continuation carried by a request, or None when absent."""
    extra = getattr(sampling_params, "extra_args", None) or {}
    if EXTRA_ARGS_KEY not in extra:
        return None
    forced = extra[EXTRA_ARGS_KEY]
    if not isinstance(forced, list) or not forced or any((not isinstance(t, int)) or isinstance(t, bool) or t < 0 for t in forced):
        raise ValueError(f"{EXTRA_ARGS_KEY} must be a non-empty list of non-negative token ids")
    max_tokens = getattr(sampling_params, "max_tokens", None)
    if max_tokens is not None and len(forced) != int(max_tokens):
        raise ValueError(f"{EXTRA_ARGS_KEY} has {len(forced)} tokens but max_tokens is {max_tokens}")
    if float(getattr(sampling_params, "temperature", 0.0)) != 0.0:
        raise ValueError("teacher forcing requires greedy sampling (temperature 0)")
    return [int(t) for t in forced]


def process_dict_updates(req_entries: dict, batch_update, new_state: Callable) -> bool:
    """Same contract as vllm/v1/sample/logits_processor/builtin.py:289-327:
    apply added, then removed, then moved (unidirectional or swap)."""
    if not batch_update:
        return False
    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        state = new_state(params, prompt_tok_ids, output_tok_ids)
        if state is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True
    if req_entries:
        for index in batch_update.removed:
            if req_entries.pop(index, None) is not None:
                updated = True
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry
    return updated


class TeacherForcingLogitsProcessor(LogitsProcessor):
    """Force greedy sampling to a recorded continuation, one token per pass."""

    @classmethod
    def validate_params(cls, sampling_params):
        forced_tokens_of(sampling_params)  # raises ValueError with the reason

    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool) -> None:
        self.device = device
        self.req_info: dict[int, tuple[list[int], list[int]]] = {}  # batch index -> (forced, live output ids)
        self.calls = 0
        self.log_path = _log_path()

    def is_argmax_invariant(self) -> bool:
        return False  # the whole point is to change the argmax

    @staticmethod
    def _new_state(params, prompt_tok_ids, output_tok_ids):
        forced = forced_tokens_of(params)
        return None if forced is None else (forced, output_tok_ids)

    def update_state(self, batch_update) -> None:
        process_dict_updates(self.req_info, batch_update, self._new_state)

    def plan(self) -> list[tuple[int, int, int]]:
        """(batch index, tokens generated so far, forced token) per forced request; fails closed."""
        out = []
        for index in sorted(self.req_info):
            forced, output_ids = self.req_info[index]
            if any(int(t) < 0 for t in output_ids):
                raise RuntimeError(f"batch index {index}: placeholder output token id; teacher forcing needs async scheduling off")
            k = len(output_ids)
            if k >= len(forced):
                raise RuntimeError(f"batch index {index}: {k} tokens generated but only {len(forced)} forced")
            out.append((index, k, int(forced[k])))
        return out

    def apply(self, logits):
        import torch
        self.calls += 1
        plan = self.plan()
        rows_log = []
        if plan:
            rows = torch.tensor([p[0] for p in plan], dtype=torch.long, device=logits.device)
            cols = torch.tensor([p[2] for p in plan], dtype=torch.long, device=logits.device)
            if int(rows.max()) >= logits.shape[0] or int(cols.max()) >= logits.shape[-1]:
                raise RuntimeError("forced row or token index outside the logits tensor")
            before = torch.argmax(logits[rows], dim=-1)
            keep = logits[rows, cols].clone()
            if not bool(torch.isfinite(keep).all()):
                raise RuntimeError("forced token has a non-finite logit")
            logits[rows] = torch.finfo(logits.dtype).min
            logits[rows, cols] = keep
            for (index, k, target), free in zip(plan, before.tolist()):
                rows_log.append({"index": index, "generated": k, "forced": target, "argmax_before_forcing": int(free), "equal": int(free) == target})
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"call": self.calls, "rank": _rank(), "rows": rows_log}) + "\n")
        return logits


def _rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _log_path() -> str | None:
    d = os.environ.get("SHAPE_HOOK_OUT")
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"forcing_rank{_rank()}.jsonl")


def read_forcing_log(hook_dir, rank: int = 0) -> list[dict]:
    p = os.path.join(str(hook_dir), f"forcing_rank{rank}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]
