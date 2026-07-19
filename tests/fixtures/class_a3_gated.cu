// Class A3: the float atomic is only reached when a determinism flag is off.
template <bool Deterministic>
__global__ void accumulate_dq(float* dq_accum, const float* dq, int* semaphore, int n_block) {
  if constexpr (Deterministic) {
    // wait for the previous n_block, then add in order
    while (atomicAdd(semaphore, 0) != n_block) {}
    dq_accum[threadIdx.x] += dq[threadIdx.x];
    __threadfence();
    if (threadIdx.x == 0) atomicAdd(semaphore, 1);
  } else {
    atomicAdd(&dq_accum[threadIdx.x], dq[threadIdx.x]);
  }
}

__global__ void runtime_gated(float* out, const float* in, bool use_atomics) {
  if (!use_atomics) {
    out[threadIdx.x] = in[threadIdx.x];
    return;
  }
  atomicAdd(&out[threadIdx.x % 8], in[threadIdx.x]);
}
