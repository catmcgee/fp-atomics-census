"""Two-rank NCCL all-reduce on GPUs 0 and 1 (diagnostic only; not part of any arm)."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    x = torch.ones(1 << 20, device="cuda") * (rank + 1)
    dist.all_reduce(x); torch.cuda.synchronize()
    print(f"rank {rank} allreduce ok {x[0].item()}", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
