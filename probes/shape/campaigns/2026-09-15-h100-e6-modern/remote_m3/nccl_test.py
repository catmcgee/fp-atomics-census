"""Two-rank NCCL init and all-reduce on the pod's two GPUs, so a TP=2 arm does not fail at ncclCommInitRank
without the campaign knowing why. Prints NCCL_OK or the exception."""
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def w(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531", RANK=str(rank), WORLD_SIZE="2")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    t = torch.ones(1024, device=f"cuda:{rank}")
    dist.all_reduce(t)
    print(f"RANK {rank} allreduce sum {t.sum().item()}", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    try:
        mp.spawn(w, nprocs=2, join=True)
        print("NCCL_OK")
    except Exception as e:
        print("NCCL_FAILED", repr(e)[:400])
