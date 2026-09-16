#!/usr/bin/env python3
"""Compare CPU rounding formulas with captured tensors from a verified bundle."""

import argparse
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path

import torch

from post_attention_operator_localise import tensor_compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def load(cell, name):
        path = args.bundle / "raw" / cell / "tensors" / f"{name}.pt.gz"
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return torch.load(io.BytesIO(gzip.decompress(data)), map_location="cpu", weights_only=True)

    projection = load("compiled-operator", "o_projection_output")
    residual = load("compiled-operator", "residual")
    weight = load("compiled-operator", "norm_weight")
    target = load("eager-boundary", "norm_output")
    rows = []
    for round_sum in (False, True):
        for round_normalised in (False, True):
            summed = projection.float() + residual.float()
            if round_sum:
                summed = summed.to(target.dtype).float()
            normalised = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
            if round_normalised:
                normalised = normalised.to(target.dtype).float()
            result = (normalised * weight.float()).to(target.dtype)
            rows.append({
                "sum_round_bf16": round_sum,
                "normalised_round_bf16": round_normalised,
                **tensor_compare(torch, target, result),
            })
    report = {
        "schema": 1,
        "scope": "CPU formula comparison after independent bundle verification; reduction order is stack-sensitive, not a GPU cause test.",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "source_sha256": sources,
        "rows": rows,
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
