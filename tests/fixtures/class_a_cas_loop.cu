// Class A: atomicCAS loop implementing a float add (pre-sm_20 style, still
// used for types without native atomicAdd).
__device__ float atomic_add_float_cas(float* addr, float val) {
  int* addr_as_int = (int*)addr;
  int old = *addr_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed,
                    __float_as_int(val + __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void use_cas(float* out, const float* in) {
  atomic_add_float_cas(out, in[threadIdx.x]);
}
