// Class B: atomicMax/atomicMin on float via integer reinterpretation. Exact
// and order-invariant, even though the operand is a float.
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

__global__ void absmax(const float* x, float* scale, int n) {
  float local = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local = fmaxf(local, fabsf(x[i]));
  atomicMaxFloat(scale, local);
}
