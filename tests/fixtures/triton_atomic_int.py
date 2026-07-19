# Class B: integer atomics in Triton, including a ticket counter that decides
# slot placement (B-indirect candidate).
import triton
import triton.language as tl


@triton.jit
def count_kernel(ids_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    ids = tl.load(ids_ptr + offs, mask=offs < N, other=0)
    tl.atomic_add(counts_ptr + ids, 1, mask=offs < N)


@triton.jit
def slot_kernel(ids_ptr, cursor_ptr, slot_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    ids = tl.load(ids_ptr + offs, mask=offs < N, other=0)
    slot = tl.atomic_add(cursor_ptr + ids, 1, mask=offs < N)
    tl.store(slot_ptr + offs, slot, mask=offs < N)


@triton.jit
def lock_kernel(lock_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) != 0:
        pass
    tl.atomic_xchg(lock_ptr, 0)
