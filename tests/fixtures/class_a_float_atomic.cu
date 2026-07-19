// Class A: plain float atomicAdd on global memory, contended across blocks.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void split_k_accumulate(const float* __restrict__ partial,
                                   float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // every split-K block adds into the same output element
    atomicAdd(&out[i % 128], partial[i]);
  }
}

__global__ void packed_half_accumulate(__half2* out, const __half2* v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out + (i % 64), v[i]);
}

__global__ void bf16_accumulate(__nv_bfloat16* out, __nv_bfloat16 v) {
  atomicAdd(out, v);
}

__global__ void double_sub(double* out, double v) {
  atomicSub(reinterpret_cast<float*>(out), static_cast<float>(v));
}
