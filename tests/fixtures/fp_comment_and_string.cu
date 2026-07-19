// False positive: atomicAdd only appears in comments and string literals.
// We do not use atomicAdd(out, v) here because the reduction is serialised.
/* atomicAdd(&acc, x) would be non-deterministic */
__global__ void no_atomics(float* out, const float* in) {
  const char* msg = "atomicAdd(out, v) not used; red.global.add.f32 neither";
  out[threadIdx.x] = in[threadIdx.x] * 2.0f;
}
#if 0
__global__ void dead(float* out) { atomicAdd(out, 1.0f); }
#endif
