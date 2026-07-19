// Class A: inline PTX float reductions to global memory.
__device__ __forceinline__ void red_add_f32(float* addr, float v) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" : : "l"(addr), "f"(v) : "memory");
}

__device__ __forceinline__ void atom_add_f32(float* addr, float v) {
  float old;
  asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(old) : "l"(addr), "f"(v) : "memory");
}

__device__ __forceinline__ void red_add_bf16x2(unsigned* addr, unsigned v) {
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" : : "l"(addr), "r"(v) : "memory");
}

__device__ __forceinline__ void multimem_red_add(float* mc_ptr, float v) {
  asm volatile("multimem.red.release.sys.global.add.f32 [%0], %1;" : : "l"(mc_ptr), "f"(v) : "memory");
}
