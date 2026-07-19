# Class A: tl.atomic_add on a float pointer (split-K style accumulation).
import triton
import triton.language as tl


@triton.jit
def splitk_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        a = tl.load(a_ptr + offs + k)
        acc += a
    c_ptrs = c_ptr + offs % N
    tl.atomic_add(c_ptrs, acc, mask=offs < M, sem="relaxed")


@triton.jit
def scale_kernel(x_ptr, s_ptr, N, BLOCK: tl.constexpr):
    # Class B: atomic max on float is exact.
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.atomic_max(s_ptr, tl.max(tl.abs(x), axis=0))
