"""All-reduce probes: NCCL, custom all-reduce, FlashInfer fusion, NVLS.

    torchrun --nproc_per_node 2 probes/probe_collectives.py --which nccl|flashinfer

nccl: torch.distributed.all_reduce on a bf16 tensor, twice, per rank
    bit-compare (class C rows vllm-0182, sglang-0269, flashinfer-0043, DeepEP-0012). Run once with the
    defaults and once with NCCL_ALGO=Tree NCCL_PROTO=Simple NCCL_MAX_NCHANNELS=1.
flashinfer: flashinfer.comm.trtllm_allreduce_fusion in kAllReduce mode
    (flashinfer-0023, expected identical) and, where available, the NVLS
    mixed_comm path (flashinfer-0026, class C).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from common import run_twice


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["nccl", "flashinfer"], required=True)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    x = torch.randn(1 << 22, device="cuda", dtype=torch.bfloat16)
    if args.which == "nccl":
        def run():
            y = x.clone()
            dist.all_reduce(y)
            return [y]
        ok = run_twice(f"nccl_allreduce_rank{rank}_{os.environ.get('NCCL_ALGO', 'default')}", run)
    else:
        import flashinfer.comm as comm
        world = dist.get_world_size()
        ws = comm.create_allreduce_fusion_workspace(rank, world, 1 << 22, torch.bfloat16)
        def run():
            return [comm.allreduce_fusion(x, ws, pattern=comm.AllReduceFusionPattern.kAllReduce)]
        ok = run_twice(f"flashinfer_allreduce_fusion_rank{rank}", run)
    dist.barrier()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
