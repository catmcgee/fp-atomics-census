# FlashAttention-4 style CuTe DSL kernels: atomics are emitted through helper
# functions wrapping nvvm.atomicrmw or inline PTX strings, not tl.atomic_*.
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, nvvm


@cute.jit
def atomic_add_fp32(a, gmem_ptr, *, loc=None, ip=None):
    nvvm.atomicrmw(res=None, op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=a)


@cute.jit
def bulk_reduce_add(smem_ptr, gmem_ptr, nbytes):
    llvm.inline_asm(
        None,
        [gmem_ptr, smem_ptr, nbytes],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
    )


@cute.kernel
def bwd_kernel(acc_dQ, tdQgdQaccum):
    for i in cutlass.range(cute.size(acc_dQ), unroll_full=True):
        atomic_add_fp32(acc_dQ[i], cute.elem_pointer(tdQgdQaccum, i))
    cute.arch.atomic_add(tdQgdQaccum, 1)
