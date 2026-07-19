// Class A2: split-K partials combined in a fixed order under a lock. The only
// atomics are the integer lock operations; the float accumulation is an
// ordinary load-add-store performed by one block at a time in split order.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int val = 1;
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

__global__ void serialized_reduce(float* C, const float* partial, int* locks, int split_idx) {
  barrier_acquire(locks, split_idx);
  C[threadIdx.x] += partial[split_idx * blockDim.x + threadIdx.x];
  barrier_release(locks);
}
