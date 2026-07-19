// Class A1: atomicAdd on float where each address is written by exactly one
// thread. The scanner must still report it; triage decides A1 from the
// address computation (blockIdx and threadIdx only, no gather).
__global__ void no_contention(float* out, const float* in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&out[i], in[i]);
}
