import os, time, torch, torch.distributed as dist
t0 = time.time()
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
print(f"rank {r} init done in {time.time()-t0:.1f}s", flush=True)
x = torch.ones(1 << 22, device="cuda", dtype=torch.bfloat16)
t1 = time.time(); dist.all_reduce(x); torch.cuda.synchronize()
print(f"rank {r} all_reduce done in {time.time()-t1:.1f}s value={x[0].item()}", flush=True)
dist.barrier(); dist.destroy_process_group()
