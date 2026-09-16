#!/usr/bin/env python3
"""Two-rank CUDA/NCCL smoke used before any causal arm launches."""
import json
import os

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"expected exactly two visible GPUs, found {torch.cuda.device_count()}")
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
torch.cuda.synchronize()
if value.item() != 3.0:
    raise RuntimeError(f"NCCL sum is {value.item()} instead of 3")
print(json.dumps({"rank": rank, "gpu": torch.cuda.get_device_name(rank), "nccl_sum": value.item()}))
dist.destroy_process_group()
