// Class A on shared memory: warps within a block race on a shared float.
__global__ void block_reduce_shared(const float* in, float* out) {
  __shared__ float acc[32];
  if (threadIdx.x < 32) acc[threadIdx.x] = 0.f;
  __syncthreads();
  atomicAdd(&acc[threadIdx.x % 32], in[threadIdx.x]);
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = acc[0];
}
