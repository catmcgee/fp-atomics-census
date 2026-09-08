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

from common import run_twice, tensor_hash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["nccl", "flashinfer"], required=True)
    ap.add_argument("--pattern", choices=["random", "cancellation"], default="random")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    x = torch.randn(1 << 22, device="cuda", dtype=torch.bfloat16)
    world = dist.get_world_size()
    if args.pattern == "cancellation":
        if world < 3:
            raise ValueError("cancellation requires at least three ranks; two ranks only test a commutative sum")
        x.fill_((256.0, 1.0, -256.0)[rank] if rank < 3 else 0.0)
    extra = {"world_size": world, "rank": rank, "input_seed": rank, "pattern": args.pattern,
             "input_hash": tensor_hash([x]),
             "scope": "two-rank smoke test" if world == 2 else "multi-rank reduction",
             "topology": os.environ.get("NCCL_TOPO_FILE"), "topology_status": "unrecorded unless supplied"}
    if args.which == "nccl":
        def run():
            y = x.clone()
            dist.all_reduce(y)
            return [y]
        ok = run_twice(f"nccl_allreduce_rank{rank}_{os.environ.get('NCCL_ALGO', 'default')}_{args.pattern}", run, extra=extra)
    else:
        import flashinfer.comm as comm
        world = dist.get_world_size()
        hidden = 4096
        xx = x[: 512 * hidden].view(512, hidden).contiguous()
        ws = comm.create_allreduce_fusion_workspace(backend="trtllm", world_size=world, rank=rank, max_token_num=512,
                                                    hidden_dim=hidden, dtype=torch.bfloat16, gpus_per_node=world)
        def run():
            return [comm.allreduce_fusion(xx, ws, pattern=comm.AllReduceFusionPattern.kAllReduce)]
        ok = run_twice(f"flashinfer_allreduce_fusion_rank{rank}_{args.pattern}", run, extra={**extra, "backend": "trtllm"})
    dist.barrier()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
