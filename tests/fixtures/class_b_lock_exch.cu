// Class B: atomicExch and atomicCAS used purely as a lock.
__global__ void locked_update(int* lock, float* shared_value, float v) {
  while (atomicCAS(lock, 0, 1) != 0) {}
  *shared_value = *shared_value + v;  // ordinary store under the lock
  __threadfence();
  atomicExch(lock, 0);
}
