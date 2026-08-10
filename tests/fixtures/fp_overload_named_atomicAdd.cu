// A device overload that reuses the primitive name. The definition is a
// site (its body calls atomicCAS on a reinterpreted half), but callers of
// atomicAdd elsewhere must not be linked to it as a wrapper.
#include <cuda_fp16.h>
__device__ __forceinline__ __half atomicAdd(__half* address, __half val) {
  unsigned int* base = (unsigned int*)((size_t)address & ~2);
  unsigned int old = *base, assumed;
  do {
    assumed = old;
    __half h = __ushort_as_half(((size_t)address & 2) ? (old >> 16) : (old & 0xffff));
    unsigned short s = __half_as_ushort(__hadd(h, val));
    unsigned int n = ((size_t)address & 2) ? (old & 0xffff) | (s << 16) : (old & 0xffff0000) | s;
    old = atomicCAS(base, assumed, n);
  } while (assumed != old);
  return val;
}

__global__ void int_counter(int* c) { atomicAdd(c, 1); }
__global__ void half_acc(__half* out, __half v) { atomicAdd(out, v); }
