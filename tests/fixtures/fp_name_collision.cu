// Identifiers that merely contain the pattern are not sites.
__device__ void non_atomicAdd_helper(float* p, float v) { *p += v; }
struct atomicAddConfig { int unused; };
__global__ void k(float* p) { non_atomicAdd_helper(p, 1.f); atomicAddConfig c{0}; (void)c; }
