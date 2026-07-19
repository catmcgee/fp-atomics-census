// Class A: TMA bulk reduce-add from shared to global. The add happens in the
// memory system; when several CTAs reduce into the same tile the order is
// arrival order.
#pragma once
#include <cstdint>

struct SM90_BULK_REDUCE_ADD {
  static void copy(void const* smem_ptr, void* gmem_ptr, int32_t bytes) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_ptr));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                 : : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(bytes) : "memory");
  }
};

template <class T>
void reduce_tile(T const* smem, T* gmem, int bytes) {
  SM90_BULK_REDUCE_ADD::copy(smem, gmem, bytes);
  cute::SM90_TMA_REDUCE_ADD_2D::copy(nullptr, smem, 0, 0);
}
