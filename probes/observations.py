"""Canonical, vocabulary-labelled sparse output observations (CPU only)."""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "labelled-topk-v2"


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sparse_output(tokens, positions) -> dict:
    """Preserve request/position boundaries, vocabulary IDs and binary64 values.

    These are API logprobs, not a full-logit or KV-cache commitment. Hex strings
    retain signed zero and all bits of the Python floats returned by the API.
    """
    tokens = list(tokens)
    positions = list(positions)
    if len(tokens) != len(positions):
        raise ValueError("one logprob position is required per generated token")
    labelled = []
    for pos in positions:
        pairs = sorted((int(t), float(v)) for t, v in pos)
        if not pairs or len({t for t, _ in pairs}) != len(pairs):
            raise ValueError("empty or duplicate vocabulary IDs in top logprobs")
        if any(math.isnan(v) or v == math.inf for _, v in pairs):
            raise ValueError("invalid API logprob")
        labelled.append([[t, v.hex()] for t, v in pairs])
    observation = {"schema": SCHEMA, "tokens": tokens, "logprobs": labelled,
                   "value_encoding": "python-float64-hex", "scope": "sampled-tokens-and-sparse-logprobs"}
    return {**observation, "hash": hashlib.sha256(canonical_bytes(observation)).hexdigest()}
