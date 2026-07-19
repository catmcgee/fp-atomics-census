// Macro wrapper: the definition has no dtype; each use site has one.
#define ATOMIC_ADD(ptr, val) atomicAdd((ptr), (val))

__global__ void via_macro(float* fout, int* iout, float v) {
  ATOMIC_ADD(&fout[threadIdx.x % 4], v);
  ATOMIC_ADD(iout, 1);
}
