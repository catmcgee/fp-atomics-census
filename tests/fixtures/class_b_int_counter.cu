// Class B: integer counters and flags. Exact in value.
__global__ void count_tokens(const int* expert_ids, int* counts, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[expert_ids[i]], 1);
}

__global__ void slot_assign(const int* expert_ids, int* cursor, int* slots, int n) {
  // B-indirect candidate: the returned ticket chooses the slot for later data.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) slots[i] = atomicAdd(&cursor[expert_ids[i]], 1);
}

__global__ void flags(unsigned* f, unsigned long long* big) {
  atomicOr(f, 1u << threadIdx.x);
  atomicInc(f + 1, 1024u);
  atomicAdd(big, 1ull);
}
