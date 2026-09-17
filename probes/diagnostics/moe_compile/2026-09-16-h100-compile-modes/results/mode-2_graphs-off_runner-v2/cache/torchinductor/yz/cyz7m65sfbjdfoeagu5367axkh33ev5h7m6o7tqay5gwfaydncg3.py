# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/6r/c6r3oih7x4m5umdxvnjapgm3pvnlozom3w2a2g7taaroafsjijvs.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'RoundRobinComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': 0, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': 131072, 'no_x_dim_1': False, 'xnumel_1': 131072, 'no_x_dim_2': False, 'xnumel_2': 131072}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    if pid % 3 == 0:
        pid_offset = pid // 3
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex // 2048
        x0 = (xindex % 2048)
        x2 = xindex
        tmp0 = (x1).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x1).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 60, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (x0 + 2048*(x1)), tmp6 & xmask, other=0.0).to(tl.float32)
        tmp8 = tmp0 >= tmp5
        tmp9 = tl.full([1], 64, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.full([1], 0.0, tl.float32)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp8, tmp11, tmp12)
        tmp14 = tl.where(tmp6, tmp7, tmp13)
        tl.store(out_ptr0 + (x2), tmp14, xmask)
    elif pid % 3 == 1:
        pid_offset = pid // 3
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex // 2048
        x3 = (xindex % 2048)
        x5 = xindex
        tmp15 = (x4).to(tl.int32)
        tmp16 = tl.full([1], 0, tl.int64)
        tmp17 = tmp15 >= tmp16
        tmp18 = (x4).to(tl.int64)
        tmp19 = (tmp18).to(tl.int64)
        tmp20 = tl.full([1], 60, tl.int64)
        tmp21 = tmp19 < tmp20
        tmp22 = tl.load(in_ptr1 + (x3 + 2048*(x4)), tmp21 & xmask, other=0.0).to(tl.float32)
        tmp23 = tmp15 >= tmp20
        tmp24 = tl.full([1], 64, tl.int64)
        tmp25 = tmp15 < tmp24
        tmp26 = tl.full([1], 0.0, tl.float32)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp23, tmp26, tmp27)
        tmp29 = tl.where(tmp21, tmp22, tmp28)
        tl.store(out_ptr1 + (x5), tmp29, xmask)
    elif pid % 3 == 2:
        pid_offset = pid // 3
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex // 2048
        x6 = (xindex % 2048)
        x8 = xindex
        tmp30 = (x7).to(tl.int32)
        tmp31 = tl.full([1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = (x7).to(tl.int64)
        tmp34 = (tmp33).to(tl.int64)
        tmp35 = tl.full([1], 60, tl.int64)
        tmp36 = tmp34 < tmp35
        tmp37 = tl.load(in_ptr2 + (x6 + 2048*(x7)), tmp36 & xmask, other=0.0).to(tl.float32)
        tmp38 = tmp30 >= tmp35
        tmp39 = tl.full([1], 64, tl.int64)
        tmp40 = tmp30 < tmp39
        tmp41 = tl.full([1], 0.0, tl.float32)
        tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
        tmp43 = tl.where(tmp38, tmp41, tmp42)
        tmp44 = tl.where(tmp36, tmp37, tmp43)
        tl.store(out_ptr2 + (x8), tmp44, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.001523712
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/ji/cji75sisojg73tz65evvmostfi7zz2unomktng6vxnpoe7uhkjtc.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'kernel_num_gb': 0.13428736, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp1) & (tmp1 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp1 < 151936")
        tmp3 = tl.load(in_ptr1 + (r0_1 + 2048*tmp1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp9) & (tmp9 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 151936")
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2048*tmp9), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 2048.0, tl.float32)
        tmp14 = (tmp7 / tmp13)
        tmp15 = tl.full([1, 1], 1e-06, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp21, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/b5/cb57q4wcc5rzt3alixe5elprz2ajohaecsx5kh5m35w6cetjwsix.py
# Topologically Sorted Source Nodes: [split, cos_sin, chunk, query, chunk_1, key, chunk_2, unsqueeze, mul_2, unsqueeze_1, mul_3, o1, mul_4, mul_5, o2, output, unsqueeze_2, mul_6, unsqueeze_3, mul_7, o1_1, mul_8, mul_9, o2_1, output_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   chunk => split
#   chunk_1 => split_1
#   chunk_2 => split_2
#   cos_sin => index
#   key => view_2
#   mul_2 => mul_45
#   mul_3 => mul_48
#   mul_4 => mul_53
#   mul_5 => mul_56
#   mul_6 => mul_84
#   mul_7 => mul_87
#   mul_8 => mul_92
#   mul_9 => mul_95
#   o1 => sub_26
#   o1_1 => sub_43
#   o2 => add_97
#   o2_1 => add_159
#   output => cat
#   output_1 => cat_1
#   query => view
#   split => split_with_sizes
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
# Graph fragment:
#   %addmm : Tensor "bf16[s72, 6144][6144, 1]cuda:0" = PlaceHolder[target=addmm]
#   %arg8_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg6_1 : Tensor "bf16[32768, 128][128, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm, [2048, 2048, 2048], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg6_1, [%arg8_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view : Tensor "bf16[s72, 16, 128][6144, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [%arg0_1, -1, 128]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view, 64, -1), kwargs = {})
#   %view_2 : Tensor "bf16[s72, 16, 128][6144, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg0_1, -1, 128]), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 64, -1), kwargs = {})
#   %unsqueeze : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_45 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_48 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze_1), kwargs = {})
#   %sub_26 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_45, %mul_48), kwargs = {})
#   %mul_53 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze), kwargs = {})
#   %mul_56 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze_1), kwargs = {})
#   %add_97 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %mul_56), kwargs = {})
#   %cat : Tensor "bf16[s72, 16, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_26, %add_97], -1), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_84 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_87 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_43 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_84, %mul_87), kwargs = {})
#   %mul_92 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_95 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_159 : Tensor "bf16[s72, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %mul_95), kwargs = {})
#   %cat_1 : Tensor "bf16[s72, 16, 128][2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_43, %add_159], -1), kwargs = {})
#   return %cat,%cat_1
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 671219712}, 'kernel_num_gb': 0.67960832, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 16)
    x2 = xindex // 2048
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (128*x1 + 6144*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 32768)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 32768")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (64 + 128*x1 + 6144*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (64 + 128*x1 + 6144*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 32768)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 32768")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (128*x1 + 6144*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tmp41 = tl.load(in_ptr0 + (2048 + 128*x1 + 6144*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41 * tmp14
    tmp43 = tl.load(in_ptr0 + (2112 + 128*x1 + 6144*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp44 = tmp43 * tmp17
    tmp45 = tmp42 - tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp6, tmp45, tmp46)
    tmp48 = tl.load(in_ptr0 + (2112 + 128*x1 + 6144*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp49 = tmp48 * tmp32
    tmp50 = tl.load(in_ptr0 + (2048 + 128*x1 + 6144*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tmp50 * tmp35
    tmp52 = tmp49 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp22, tmp52, tmp53)
    tmp55 = tl.where(tmp6, tmp47, tmp54)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
    tl.store(out_ptr1 + (x4), tmp55, xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/7b/c7bc5vcmukgvimmi62rnacvsxkhbeutt22zox3afntaswueom62o.py
# Topologically Sorted Source Nodes: [long, output_parallel, x_3, to_8, x_4, pow_2, variance_1, add_4, rsqrt_1, x_5, to_10, x_6], Original ATen: [aten._to_copy, aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_219
#   long => convert_element_type
#   output_parallel => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_10 => convert_element_type_11
#   to_8 => convert_element_type_9
#   variance_1 => mean_1
#   x_3 => convert_element_type_8
#   x_4 => add_206
#   x_5 => mul_137
#   x_6 => mul_142
# Graph fragment:
#   %mm : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg1_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %buf11 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf11]
#   %arg11_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %add_206 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_8, %convert_element_type_9), kwargs = {})
#   %pow_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_206, 2), kwargs = {})
#   %mean_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_219 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_219,), kwargs = {})
#   %mul_137 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_206, %rsqrt_1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_137, torch.bfloat16), kwargs = {})
#   %mul_142 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, %arg11_1), kwargs = {})
#   return %buf11,%mul_142
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 65536, 'r0_': 201330688}, 'kernel_num_gb': 0.201396224, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.int64)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp3 < 151936")
        tmp5 = tl.load(in_ptr2 + (r0_1 + 2048*tmp3), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp1 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp2.to(tl.int64)
        tl.device_assert(((0 <= tmp14) & (tmp14 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp14 < 151936")
        tmp16 = tl.load(in_ptr2 + (r0_1 + 2048*tmp14), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp13 + tmp17
        tmp19 = tl.full([1, 1], 2048.0, tl.float32)
        tmp20 = (tmp10 / tmp19)
        tmp21 = tl.full([1, 1], 1e-06, tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp25 * tmp26
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp27, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/2z/c2zd3jakmjsg4ybrif2fkutst4ojcysjtuxgtmale2vyuqqyfjd7.py
# Topologically Sorted Source Nodes: [long, output_parallel, x_3, to_8, x_4, result, x_7, x_residual, to_13, x_8, pow_3, variance_2, add_7, rsqrt_2, x_9, to_15, x_10], Original ATen: [aten._to_copy, aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_7 => add_273
#   long => convert_element_type
#   output_parallel => embedding
#   pow_3 => pow_3
#   result => add_247
#   rsqrt_2 => rsqrt_2
#   to_13 => convert_element_type_15
#   to_15 => convert_element_type_17
#   to_8 => convert_element_type_9
#   variance_2 => mean_2
#   x_10 => mul_175
#   x_3 => convert_element_type_8
#   x_4 => add_206
#   x_7 => convert_element_type_14
#   x_8 => add_260
#   x_9 => mul_170
#   x_residual => convert_element_type_10
# Graph fragment:
#   %getitem_12 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %getitem_13 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_13]
#   %mm : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg1_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_260 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_260]
#   %buf19 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf19]
#   %arg14_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %add_206 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_8, %convert_element_type_9), kwargs = {})
#   %add_247 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, %getitem_13), kwargs = {})
#   %convert_element_type_14 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_247, torch.float32), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_206, torch.bfloat16), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_10, torch.float32), kwargs = {})
#   %add_260 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_14, %convert_element_type_15), kwargs = {})
#   %pow_3 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_260, 2), kwargs = {})
#   %mean_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_273 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_273,), kwargs = {})
#   %mul_170 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_260, %rsqrt_2), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_170, torch.bfloat16), kwargs = {})
#   %mul_175 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %arg14_1), kwargs = {})
#   return %add_260,%buf19,%mul_175
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_4 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*i32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 65536, 'r0_': 603983872}, 'kernel_num_gb': 0.46983168, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.int64)
        tl.device_assert(((0 <= tmp7) & (tmp7 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 151936")
        tmp9 = tl.load(in_ptr4 + (r0_1 + 2048*tmp7), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp5 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp3 + tmp13
        tmp15 = tmp14 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
        tl.store(out_ptr0 + (r0_1 + 2048*x0), tmp14, r0_mask & xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp19 = tl.load(out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], 2048.0, tl.float32)
        tmp21 = (tmp17 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp26 * tmp27
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp28, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/zo/czozi3yvya5uihp52mhnc5icorkaci5eujfvpnz54rbkvkqhnzc4.py
# Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, pow_4, variance_3, add_11, rsqrt_3, x_13, to_24, x_14], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_11 => add_478
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   to_22 => convert_element_type_24
#   to_24 => convert_element_type_26
#   variance_3 => mean_3
#   x_11 => convert_element_type_23
#   x_12 => add_465
#   x_13 => mul_299
#   x_14 => mul_304
#   x_residual_1 => convert_element_type_16
# Graph fragment:
#   %mm_2 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %add_260 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_260]
#   %buf30 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf30]
#   %arg19_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %convert_element_type_23 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_260, torch.bfloat16), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_16, torch.float32), kwargs = {})
#   %add_465 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_23, %convert_element_type_24), kwargs = {})
#   %pow_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_465, 2), kwargs = {})
#   %mean_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_478 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_478,), kwargs = {})
#   %mul_299 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_465, %rsqrt_3), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_299, torch.bfloat16), kwargs = {})
#   %mul_304 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %arg19_1), kwargs = {})
#   return %buf30,%mul_304
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 335548416}, 'kernel_num_gb': 0.268439552, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp1 + tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp11 + tmp14
        tmp16 = tl.full([1, 1], 2048.0, tl.float32)
        tmp17 = (tmp8 / tmp16)
        tmp18 = tl.full([1, 1], 1e-06, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/er/cermcabu4zwxowefgh7vlsxpw4ymmaxwwdfuxbu2w5crodlwraph.py
# Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, result_1, x_15, x_residual_2, to_27, x_16, pow_5, variance_4, add_14, rsqrt_4, x_17, to_29, x_18], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_14 => add_532
#   pow_5 => pow_5
#   result_1 => add_506
#   rsqrt_4 => rsqrt_4
#   to_22 => convert_element_type_24
#   to_27 => convert_element_type_30
#   to_29 => convert_element_type_32
#   variance_4 => mean_4
#   x_11 => convert_element_type_23
#   x_12 => add_465
#   x_15 => convert_element_type_29
#   x_16 => add_519
#   x_17 => mul_332
#   x_18 => mul_337
#   x_residual_1 => convert_element_type_16
#   x_residual_2 => convert_element_type_25
# Graph fragment:
#   %getitem_26 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_26]
#   %getitem_27 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_27]
#   %mm_2 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %add_260 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_260]
#   %buf37 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf37]
#   %arg22_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_23 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_260, torch.bfloat16), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_16, torch.float32), kwargs = {})
#   %add_465 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_23, %convert_element_type_24), kwargs = {})
#   %add_506 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, %getitem_27), kwargs = {})
#   %convert_element_type_29 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_506, torch.float32), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_465, torch.bfloat16), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_25, torch.float32), kwargs = {})
#   %add_519 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_29, %convert_element_type_30), kwargs = {})
#   %pow_5 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_519, 2), kwargs = {})
#   %mean_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_532 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_532,), kwargs = {})
#   %mul_332 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_519, %rsqrt_4), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_332, torch.bfloat16), kwargs = {})
#   %mul_337 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_32, %arg22_1), kwargs = {})
#   return %buf37,%mul_337
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 469766144}, 'kernel_num_gb': 0.40265728, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp5 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp3 + tmp11
        tmp13 = tmp12 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp22 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp20 + tmp28
        tmp30 = tl.full([1, 1], 2048.0, tl.float32)
        tmp31 = (tmp15 / tmp30)
        tmp32 = tl.full([1, 1], 1e-06, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = libdevice.rsqrt(tmp33)
        tmp35 = tmp29 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp38 = tmp36 * tmp37
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp38, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/ys/cyslh7mpbsi6gqx4nunbteswlwzbv5qywoibinxzhlu26yrmiv2e.py
# Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, result_1, x_15, x_residual_2, to_27, x_16, x_19, x_residual_3, to_36, x_20, pow_6, variance_5, add_18, rsqrt_5, x_21, to_38, x_22], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_18 => add_737
#   pow_6 => pow_6
#   result_1 => add_506
#   rsqrt_5 => rsqrt_5
#   to_22 => convert_element_type_24
#   to_27 => convert_element_type_30
#   to_36 => convert_element_type_39
#   to_38 => convert_element_type_41
#   variance_5 => mean_5
#   x_11 => convert_element_type_23
#   x_12 => add_465
#   x_15 => convert_element_type_29
#   x_16 => add_519
#   x_19 => convert_element_type_38
#   x_20 => add_724
#   x_21 => mul_461
#   x_22 => mul_466
#   x_residual_1 => convert_element_type_16
#   x_residual_2 => convert_element_type_25
#   x_residual_3 => convert_element_type_31
# Graph fragment:
#   %mm_4 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %getitem_26 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_26]
#   %getitem_27 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_27]
#   %mm_2 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %add_260 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_260]
#   %add_724 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_724]
#   %buf49 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf49]
#   %arg27_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convert_element_type_23 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_260, torch.bfloat16), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_16, torch.float32), kwargs = {})
#   %add_465 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_23, %convert_element_type_24), kwargs = {})
#   %add_506 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, %getitem_27), kwargs = {})
#   %convert_element_type_29 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_506, torch.float32), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_465, torch.bfloat16), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_25, torch.float32), kwargs = {})
#   %add_519 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_29, %convert_element_type_30), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_4, torch.float32), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_519, torch.bfloat16), kwargs = {})
#   %convert_element_type_39 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_31, torch.float32), kwargs = {})
#   %add_724 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, %convert_element_type_39), kwargs = {})
#   %pow_6 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_724, 2), kwargs = {})
#   %mean_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %add_737 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_737,), kwargs = {})
#   %mul_461 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_724, %rsqrt_5), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_461, torch.bfloat16), kwargs = {})
#   %mul_466 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_41, %arg27_1), kwargs = {})
#   return %add_724,%buf49,%mul_466
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 805310464}, 'kernel_num_gb': 0.603983872, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp7 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp5 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp1 + tmp16
        tmp18 = tmp17 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp17, r0_mask & xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp22 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.full([1, 1], 2048.0, tl.float32)
        tmp24 = (tmp20 / tmp23)
        tmp25 = tl.full([1, 1], 1e-06, tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = tmp29 * tmp30
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/fq/cfqiqnu4pjkyqlpfearexn36qdfpnbdbrxxzggk4xetjx4ckadum.py
# Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, pow_7, variance_6, add_21, rsqrt_6, x_25, to_43, x_26], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_21 => add_791
#   pow_7 => pow_7
#   result_2 => add_765
#   rsqrt_6 => rsqrt_6
#   to_41 => convert_element_type_45
#   to_43 => convert_element_type_47
#   variance_6 => mean_6
#   x_23 => convert_element_type_44
#   x_24 => add_778
#   x_25 => mul_494
#   x_26 => mul_499
#   x_residual_4 => convert_element_type_40
# Graph fragment:
#   %getitem_40 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_40]
#   %getitem_41 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_41]
#   %add_724 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_724]
#   %buf56 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf56]
#   %arg30_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %add_765 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, %getitem_41), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_765, torch.float32), kwargs = {})
#   %convert_element_type_40 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_724, torch.bfloat16), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_40, torch.float32), kwargs = {})
#   %add_778 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, %convert_element_type_45), kwargs = {})
#   %pow_7 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_778, 2), kwargs = {})
#   %mean_6 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_791 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-06), kwargs = {})
#   %rsqrt_6 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_791,), kwargs = {})
#   %mul_494 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_778, %rsqrt_6), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_494, torch.bfloat16), kwargs = {})
#   %mul_499 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_47, %arg30_1), kwargs = {})
#   return %buf56,%mul_499
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 402657280}, 'kernel_num_gb': 0.335548416, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp3 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp15 + tmp18
        tmp20 = tl.full([1, 1], 2048.0, tl.float32)
        tmp21 = (tmp10 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp26 * tmp27
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp28, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/go/cgofkopkos3sozjvvmzjo5zde3fmzcjho3rhx2chmefzxjtqfxhq.py
# Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, x_27, x_residual_5, to_50, x_28, pow_8, variance_7, add_25, rsqrt_7, x_29, to_52, x_30], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_25 => add_996
#   pow_8 => pow_8
#   result_2 => add_765
#   rsqrt_7 => rsqrt_7
#   to_41 => convert_element_type_45
#   to_50 => convert_element_type_54
#   to_52 => convert_element_type_56
#   variance_7 => mean_7
#   x_23 => convert_element_type_44
#   x_24 => add_778
#   x_27 => convert_element_type_53
#   x_28 => add_983
#   x_29 => mul_623
#   x_30 => mul_628
#   x_residual_4 => convert_element_type_40
#   x_residual_5 => convert_element_type_46
# Graph fragment:
#   %mm_6 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %getitem_40 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_40]
#   %getitem_41 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_41]
#   %add_724 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_724]
#   %buf67 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf67]
#   %arg35_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %add_765 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, %getitem_41), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_765, torch.float32), kwargs = {})
#   %convert_element_type_40 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_724, torch.bfloat16), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_40, torch.float32), kwargs = {})
#   %add_778 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, %convert_element_type_45), kwargs = {})
#   %convert_element_type_53 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %convert_element_type_46 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_778, torch.bfloat16), kwargs = {})
#   %convert_element_type_54 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_46, torch.float32), kwargs = {})
#   %add_983 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_53, %convert_element_type_54), kwargs = {})
#   %pow_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_983, 2), kwargs = {})
#   %mean_7 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_996 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_996,), kwargs = {})
#   %mul_623 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_983, %rsqrt_7), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_623, torch.bfloat16), kwargs = {})
#   %mul_628 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_56, %arg35_1), kwargs = {})
#   return %buf67,%mul_628
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 469766144}, 'kernel_num_gb': 0.40265728, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp5 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp1 + tmp11
        tmp13 = tmp12 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp22 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp18 + tmp28
        tmp30 = tl.full([1, 1], 2048.0, tl.float32)
        tmp31 = (tmp15 / tmp30)
        tmp32 = tl.full([1, 1], 1e-06, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = libdevice.rsqrt(tmp33)
        tmp35 = tmp29 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp38 = tmp36 * tmp37
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp38, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/a6/ca6zqk3opzp2ubmp7afcvxpxzvlrfncwxeu6ewvdtycwm4i23gwj.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_10 = async_compile.triton('triton_poi_fused_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'RoundRobinComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': 0, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': 131072, 'no_x_dim_1': False, 'xnumel_1': 131072, 'no_x_dim_2': False, 'xnumel_2': 131072, 'no_x_dim_3': False, 'xnumel_3': 131072}, 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    if pid % 4 == 0:
        pid_offset = pid // 4
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex // 2048
        x0 = (xindex % 2048)
        x2 = xindex
        tmp0 = (x1).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x1).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 60, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (x0 + 2048*(x1)), tmp6 & xmask, other=0.0).to(tl.float32)
        tmp8 = tmp0 >= tmp5
        tmp9 = tl.full([1], 64, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.full([1], 0.0, tl.float32)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp8, tmp11, tmp12)
        tmp14 = tl.where(tmp6, tmp7, tmp13)
        tl.store(out_ptr0 + (x2), tmp14, xmask)
    elif pid % 4 == 1:
        pid_offset = pid // 4
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex // 2048
        x3 = (xindex % 2048)
        x5 = xindex
        tmp15 = (x4).to(tl.int32)
        tmp16 = tl.full([1], 0, tl.int64)
        tmp17 = tmp15 >= tmp16
        tmp18 = (x4).to(tl.int64)
        tmp19 = (tmp18).to(tl.int64)
        tmp20 = tl.full([1], 60, tl.int64)
        tmp21 = tmp19 < tmp20
        tmp22 = tl.load(in_ptr1 + (x3 + 2048*(x4)), tmp21 & xmask, other=0.0).to(tl.float32)
        tmp23 = tmp15 >= tmp20
        tmp24 = tl.full([1], 64, tl.int64)
        tmp25 = tmp15 < tmp24
        tmp26 = tl.full([1], 0.0, tl.float32)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp23, tmp26, tmp27)
        tmp29 = tl.where(tmp21, tmp22, tmp28)
        tl.store(out_ptr1 + (x5), tmp29, xmask)
    elif pid % 4 == 2:
        pid_offset = pid // 4
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex // 2048
        x6 = (xindex % 2048)
        x8 = xindex
        tmp30 = (x7).to(tl.int32)
        tmp31 = tl.full([1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = (x7).to(tl.int64)
        tmp34 = (tmp33).to(tl.int64)
        tmp35 = tl.full([1], 60, tl.int64)
        tmp36 = tmp34 < tmp35
        tmp37 = tl.load(in_ptr2 + (x6 + 2048*(x7)), tmp36 & xmask, other=0.0).to(tl.float32)
        tmp38 = tmp30 >= tmp35
        tmp39 = tl.full([1], 64, tl.int64)
        tmp40 = tmp30 < tmp39
        tmp41 = tl.full([1], 0.0, tl.float32)
        tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
        tmp43 = tl.where(tmp38, tmp41, tmp42)
        tmp44 = tl.where(tmp36, tmp37, tmp43)
        tl.store(out_ptr2 + (x8), tmp44, xmask)
    elif pid % 4 == 3:
        pid_offset = pid // 4
        xnumel = 131072
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex // 2048
        x9 = (xindex % 2048)
        x11 = xindex
        tmp45 = (x10).to(tl.int32)
        tmp46 = tl.full([1], 0, tl.int64)
        tmp47 = tmp45 >= tmp46
        tmp48 = (x10).to(tl.int64)
        tmp49 = (tmp48).to(tl.int64)
        tmp50 = tl.full([1], 60, tl.int64)
        tmp51 = tmp49 < tmp50
        tmp52 = tl.load(in_ptr3 + (x9 + 2048*(x10)), tmp51 & xmask, other=0.0).to(tl.float32)
        tmp53 = tmp45 >= tmp50
        tmp54 = tl.full([1], 64, tl.int64)
        tmp55 = tmp45 < tmp54
        tmp56 = tl.full([1], 0.0, tl.float32)
        tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
        tmp58 = tl.where(tmp53, tmp56, tmp57)
        tmp59 = tl.where(tmp51, tmp52, tmp58)
        tl.store(out_ptr3 + (x11), tmp59, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((2048, 64), (1, 2048), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.002031616
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/22/c22f24kjgnmxzvzyc2uzsr6ksxi6nc37adqiowhh4zby3d5bw5qb.py
# Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, x_27, x_residual_5, to_50, x_28, result_3, x_31, x_residual_6, to_55, x_32, pow_9, variance_8, add_28, rsqrt_8, x_33, to_57, x_34], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_28 => add_1050
#   pow_9 => pow_9
#   result_2 => add_765
#   result_3 => add_1024
#   rsqrt_8 => rsqrt_8
#   to_41 => convert_element_type_45
#   to_50 => convert_element_type_54
#   to_55 => convert_element_type_60
#   to_57 => convert_element_type_62
#   variance_8 => mean_8
#   x_23 => convert_element_type_44
#   x_24 => add_778
#   x_27 => convert_element_type_53
#   x_28 => add_983
#   x_31 => convert_element_type_59
#   x_32 => add_1037
#   x_33 => mul_656
#   x_34 => mul_661
#   x_residual_4 => convert_element_type_40
#   x_residual_5 => convert_element_type_46
#   x_residual_6 => convert_element_type_55
# Graph fragment:
#   %getitem_54 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_54]
#   %getitem_55 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_55]
#   %mm_6 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %getitem_40 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_40]
#   %getitem_41 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_41]
#   %add_724 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_724]
#   %add_1037 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_1037]
#   %buf75 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf75]
#   %arg38_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %add_765 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, %getitem_41), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_765, torch.float32), kwargs = {})
#   %convert_element_type_40 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_724, torch.bfloat16), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_40, torch.float32), kwargs = {})
#   %add_778 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, %convert_element_type_45), kwargs = {})
#   %convert_element_type_53 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %convert_element_type_46 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_778, torch.bfloat16), kwargs = {})
#   %convert_element_type_54 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_46, torch.float32), kwargs = {})
#   %add_983 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_53, %convert_element_type_54), kwargs = {})
#   %add_1024 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, %getitem_55), kwargs = {})
#   %convert_element_type_59 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1024, torch.float32), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_983, torch.bfloat16), kwargs = {})
#   %convert_element_type_60 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_55, torch.float32), kwargs = {})
#   %add_1037 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_59, %convert_element_type_60), kwargs = {})
#   %pow_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1037, 2), kwargs = {})
#   %mean_8 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [-1], True), kwargs = {})
#   %add_1050 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_8, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1050,), kwargs = {})
#   %mul_656 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1037, %rsqrt_8), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_656, torch.bfloat16), kwargs = {})
#   %mul_661 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_62, %arg38_1), kwargs = {})
#   return %add_1037,%buf75,%mul_661
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 872419328}, 'kernel_num_gb': 0.671092736, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp5 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp3 + tmp18
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp19, r0_mask & xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.full([1, 1], 2048.0, tl.float32)
        tmp26 = (tmp22 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/ov/cov5o4np36dizlpvy6hkuqz43w3ewax5mhzqqsjpzqhzmrueecdb.py
# Topologically Sorted Source Nodes: [output_119], Original ATen: [aten.t, aten.mm]
# Source node to ATen node mapping:
#   output_119 => constant_pad_nd_default, permute_71
# Graph fragment:
#   %arg196_1 : Tensor "bf16[60, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg196_1]
#   %permute_71 : Tensor "bf16[2048, 60][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg196_1, [1, 0]), kwargs = {})
#   %constant_pad_nd_default : Tensor "bf16[2048, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_71, [0, 4, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused_mm_t_12 = async_compile.triton('triton_poi_fused_mm_t_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mm_t_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 770048}, 'kernel_num_gb': 0.000507904, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_t_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 2048
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = (x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x1).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 60, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (x0 + 2048*(x1)), tmp6, other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 64, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.full([1], 0.0, tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /root/mode-matrix-2/mode-2_graphs-off_runner-v2/cache/torchinductor/l5/cl5xv2flynsd32ch3gjn2wtxp3357opkj5jfrrvyd3piuelaszam.py
# Topologically Sorted Source Nodes: [result_23, x_191, x_residual_46, to_335, x_192, pow_49, variance_48, add_168, rsqrt_48, x_193, to_337, x_194], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_168 => add_6230
#   pow_49 => pow_49
#   result_23 => add_6204
#   rsqrt_48 => rsqrt_48
#   to_335 => convert_element_type_360
#   to_337 => convert_element_type_362
#   variance_48 => mean_48
#   x_191 => convert_element_type_359
#   x_192 => add_6217
#   x_193 => mul_3896
#   x_194 => mul_3901
#   x_residual_46 => convert_element_type_355
# Graph fragment:
#   %getitem_334 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_334]
#   %getitem_335 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=getitem_335]
#   %add_6163 : Tensor "f32[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=add_6163]
#   %buf448 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf448]
#   %arg198_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg198_1]
#   %add_6204 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_334, %getitem_335), kwargs = {})
#   %convert_element_type_359 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6204, torch.float32), kwargs = {})
#   %convert_element_type_355 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6163, torch.bfloat16), kwargs = {})
#   %convert_element_type_360 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_355, torch.float32), kwargs = {})
#   %add_6217 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_359, %convert_element_type_360), kwargs = {})
#   %pow_49 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6217, 2), kwargs = {})
#   %mean_48 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_49, [-1], True), kwargs = {})
#   %add_6230 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_48, 1e-06), kwargs = {})
#   %rsqrt_48 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6230,), kwargs = {})
#   %mul_3896 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6217, %rsqrt_48), kwargs = {})
#   %convert_element_type_362 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3896, torch.bfloat16), kwargs = {})
#   %mul_3901 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_362, %arg198_1), kwargs = {})
#   return %buf448,%mul_3901
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 402657280}, 'kernel_num_gb': 0.335548416, 'kernel_flop': 0, 'backend_hash': '12CB7AF2462FDF7F0782918BFFA38096B8D56EE547964A5304EFBEAD2F02B50A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp3 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp15 + tmp18
        tmp20 = tl.full([1, 1], 2048.0, tl.float32)
        tmp21 = (tmp10 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp26 * tmp27
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp28, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1 = args
        args.clear()
        s72 = arg0_1
        s80 = s72
        assert_size_stride(arg1_1, (s72, ), (1, ), 'input')
        assert_size_stride(arg2_1, (151936, 2048), (2048, 1), 'input')
        assert_size_stride(arg3_1, (2048, ), (1, ), 'input')
        assert_size_stride(arg12_1, (60, 2048), (2048, 1), 'input')
        assert_size_stride(arg20_1, (60, 2048), (2048, 1), 'input')
        assert_size_stride(arg28_1, (60, 2048), (2048, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf13 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            buf32 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            buf51 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            buf1 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_4, output_9, output_14], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg12_1, arg20_1, arg28_1, buf13, buf32, buf51, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [output_4, output_9, output_14], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_1.run(arg1_1, arg2_1, arg3_1, buf1, s72, 2048, stream=raw_stream0)
            del arg12_1
            del arg20_1
            del arg28_1
            del arg3_1
            assert_size_stride(arg4_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg5_1, (6144, 2048), (2048, 1), 'input')
            buf2 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [long, output_parallel, x, pow_1, variance, add, rsqrt, x_1, to_1, x_2, output_parallel_1], Original ATen: [aten._to_copy, aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg4_1, buf1, reinterpret_tensor(arg5_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf2)
            del arg4_1
            del arg5_1
            assert_size_stride(arg8_1, (s72, ), (1, ), 'input')
            assert_size_stride(arg6_1, (32768, 128), (128, 1), 'input')
            arg8_1 = copy_if_misaligned(arg8_1)
            buf3 = reinterpret_tensor(buf1, (s72, 16, 128), (2048, 128, 1), 0); del buf1  # reuse
            buf5 = empty_strided_cuda((s72, 16, 128), (2048, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, cos_sin, chunk, query, chunk_1, key, chunk_2, unsqueeze, mul_2, unsqueeze_1, mul_3, o1, mul_4, mul_5, o2, output, unsqueeze_2, mul_6, unsqueeze_3, mul_7, o1_1, mul_8, mul_9, o2_1, output_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf2, arg8_1, arg6_1, buf3, buf5, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf4 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, value, kv_cache_dummy_dep], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf6 = torch.ops.vllm.unified_kv_cache_update.default(buf5, reinterpret_tensor(buf2, (s72, 16, 128), (6144, 128, 1), 4096), arg9_1)
            buf7 = buf6
            assert_alignment(buf7, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf6
            # Topologically Sorted Source Nodes: [split, unified_attention_with_output, value], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf3, buf5, reinterpret_tensor(buf2, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf4, (s72, 16, 128), (2048, 128, 1), 0), arg9_1, None, None, buf7)
            del arg9_1
            del buf3
            del buf7
            assert_size_stride(arg10_1, (2048, 2048), (2048, 1), 'input')
            buf10 = reinterpret_tensor(buf5, (s72, 2048), (2048, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output, output_parallel_2], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf4, reinterpret_tensor(arg10_1, (2048, 2048), (1, 2048), 0), out=buf10)
            del arg10_1
            assert_size_stride(arg11_1, (2048, ), (1, ), 'input')
            buf12 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [long, output_parallel, x_3, to_8, x_4, pow_2, variance_1, add_4, rsqrt_1, x_5, to_10, x_6], Original ATen: [aten._to_copy, aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3.run(buf10, arg1_1, arg2_1, arg11_1, buf12, s72, 2048, stream=raw_stream0)
            del arg11_1
            buf14 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_4], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf12, buf13, out=buf14)
            # Topologically Sorted Source Nodes: [output_4, moe_forward_shared], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf15 = torch.ops.vllm.moe_forward_shared.default(buf12, reinterpret_tensor(buf14, (s72, 60), (64, 1), 0), buf12, None, arg13_1, 0)
            del arg13_1
            buf16 = buf15[0]
            assert_size_stride(buf16, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf16, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf17 = buf15[1]
            assert_size_stride(buf17, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf17, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf15
            assert_size_stride(arg14_1, (2048, ), (1, ), 'input')
            buf18 = empty_strided_cuda((s72, 2048), (2048, 1), torch.float32)
            buf20 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [long, output_parallel, x_3, to_8, x_4, result, x_7, x_residual, to_13, x_8, pow_3, variance_2, add_7, rsqrt_2, x_9, to_15, x_10], Original ATen: [aten._to_copy, aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_4.run(buf16, buf17, buf10, arg1_1, arg2_1, arg14_1, buf18, buf20, s72, 2048, stream=raw_stream0)
            del arg14_1
            del arg1_1
            del arg2_1
            del buf10
            assert_size_stride(arg15_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg16_1, (6144, 2048), (2048, 1), 'input')
            buf21 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [pow_3, variance_2, add_7, rsqrt_2, x_9, to_15, x_10, output_parallel_3], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg15_1, buf20, reinterpret_tensor(arg16_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf21)
            del arg15_1
            del arg16_1
            buf22 = reinterpret_tensor(buf20, (s72, 16, 128), (2048, 128, 1), 0); del buf20  # reuse
            buf24 = reinterpret_tensor(buf17, (s72, 16, 128), (2048, 128, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [split_1, cos_sin_1, chunk_3, query_3, chunk_4, key_3, chunk_5, unsqueeze_4, mul_14, unsqueeze_5, mul_15, o1_2, mul_16, mul_17, o2_2, output_5, unsqueeze_6, mul_18, unsqueeze_7, mul_19, o1_3, mul_20, mul_21, o2_3, output_6], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf21, arg8_1, arg6_1, buf22, buf24, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf23 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [split_1, value_1, kv_cache_dummy_dep_1], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf25 = torch.ops.vllm.unified_kv_cache_update.default(buf24, reinterpret_tensor(buf21, (s72, 16, 128), (6144, 128, 1), 4096), arg17_1)
            buf26 = buf25
            assert_alignment(buf26, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf25
            # Topologically Sorted Source Nodes: [split_1, unified_attention_with_output_1, value_1], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf22, buf24, reinterpret_tensor(buf21, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf23, (s72, 16, 128), (2048, 128, 1), 0), arg17_1, None, None, buf26)
            del arg17_1
            del buf26
            assert_size_stride(arg18_1, (2048, 2048), (2048, 1), 'input')
            buf29 = reinterpret_tensor(buf24, (s72, 2048), (2048, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_1, output_parallel_4], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf23, reinterpret_tensor(arg18_1, (2048, 2048), (1, 2048), 0), out=buf29)
            del arg18_1
            assert_size_stride(arg19_1, (2048, ), (1, ), 'input')
            buf31 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, pow_4, variance_3, add_11, rsqrt_3, x_13, to_24, x_14], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf29, buf18, arg19_1, buf31, s72, 2048, stream=raw_stream0)
            del arg19_1
            buf33 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [output_9], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf31, buf32, out=buf33)
            # Topologically Sorted Source Nodes: [output_9, moe_forward_shared_1], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf34 = torch.ops.vllm.moe_forward_shared.default(buf31, reinterpret_tensor(buf33, (s72, 60), (64, 1), 0), buf31, None, arg21_1, 0)
            del arg21_1
            del buf33
            buf35 = buf34[0]
            assert_size_stride(buf35, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf35, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf36 = buf34[1]
            assert_size_stride(buf36, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf36, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf34
            assert_size_stride(arg22_1, (2048, ), (1, ), 'input')
            buf38 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, result_1, x_15, x_residual_2, to_27, x_16, pow_5, variance_4, add_14, rsqrt_4, x_17, to_29, x_18], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf35, buf36, buf29, buf18, arg22_1, buf38, s72, 2048, stream=raw_stream0)
            del arg22_1
            assert_size_stride(arg23_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg24_1, (6144, 2048), (2048, 1), 'input')
            buf39 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, result_1, x_15, x_residual_2, to_27, x_16, pow_5, variance_4, add_14, rsqrt_4, x_17, to_29, x_18, output_parallel_5], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg23_1, buf38, reinterpret_tensor(arg24_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf39)
            del arg23_1
            del arg24_1
            buf40 = reinterpret_tensor(buf38, (s72, 16, 128), (2048, 128, 1), 0); del buf38  # reuse
            buf42 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [split_2, cos_sin_2, chunk_6, query_6, chunk_7, key_6, chunk_8, unsqueeze_8, mul_26, unsqueeze_9, mul_27, o1_4, mul_28, mul_29, o2_4, output_10, unsqueeze_10, mul_30, unsqueeze_11, mul_31, o1_5, mul_32, mul_33, o2_5, output_11], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf39, arg8_1, arg6_1, buf40, buf42, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf41 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, value_2, kv_cache_dummy_dep_2], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf43 = torch.ops.vllm.unified_kv_cache_update.default(buf42, reinterpret_tensor(buf39, (s72, 16, 128), (6144, 128, 1), 4096), arg25_1)
            buf44 = buf43
            assert_alignment(buf44, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf43
            # Topologically Sorted Source Nodes: [split_2, unified_attention_with_output_2, value_2], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf40, buf42, reinterpret_tensor(buf39, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf41, (s72, 16, 128), (2048, 128, 1), 0), arg25_1, None, None, buf44)
            del arg25_1
            del buf39
            del buf40
            del buf44
            assert_size_stride(arg26_1, (2048, 2048), (2048, 1), 'input')
            buf47 = reinterpret_tensor(buf42, (s72, 2048), (2048, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_2, output_parallel_6], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf41, reinterpret_tensor(arg26_1, (2048, 2048), (1, 2048), 0), out=buf47)
            del arg26_1
            assert_size_stride(arg27_1, (2048, ), (1, ), 'input')
            buf48 = buf18; del buf18  # reuse
            buf50 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [x_11, x_residual_1, to_22, x_12, result_1, x_15, x_residual_2, to_27, x_16, x_19, x_residual_3, to_36, x_20, pow_6, variance_5, add_18, rsqrt_5, x_21, to_38, x_22], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf48, buf47, buf35, buf36, buf29, arg27_1, buf50, s72, 2048, stream=raw_stream0)
            del arg27_1
            del buf29
            del buf35
            buf52 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_14], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf50, buf51, out=buf52)
            # Topologically Sorted Source Nodes: [output_14, moe_forward_shared_2], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf53 = torch.ops.vllm.moe_forward_shared.default(buf50, reinterpret_tensor(buf52, (s72, 60), (64, 1), 0), buf50, None, arg29_1, 0)
            del arg29_1
            buf54 = buf53[0]
            assert_size_stride(buf54, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf54, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf55 = buf53[1]
            assert_size_stride(buf55, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf55, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf53
            assert_size_stride(arg30_1, (2048, ), (1, ), 'input')
            buf57 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, pow_7, variance_6, add_21, rsqrt_6, x_25, to_43, x_26], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf54, buf55, buf48, arg30_1, buf57, s72, 2048, stream=raw_stream0)
            del arg30_1
            assert_size_stride(arg31_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg32_1, (6144, 2048), (2048, 1), 'input')
            buf58 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, pow_7, variance_6, add_21, rsqrt_6, x_25, to_43, x_26, output_parallel_7], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg31_1, buf57, reinterpret_tensor(arg32_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf58)
            del arg31_1
            del arg32_1
            buf59 = reinterpret_tensor(buf57, (s72, 16, 128), (2048, 128, 1), 0); del buf57  # reuse
            buf61 = reinterpret_tensor(buf47, (s72, 16, 128), (2048, 128, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [split_3, cos_sin_3, chunk_9, query_9, chunk_10, key_9, chunk_11, unsqueeze_12, mul_38, unsqueeze_13, mul_39, o1_6, mul_40, mul_41, o2_6, output_15, unsqueeze_14, mul_42, unsqueeze_15, mul_43, o1_7, mul_44, mul_45, o2_7, output_16], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf58, arg8_1, arg6_1, buf59, buf61, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf60 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [split_3, value_3, kv_cache_dummy_dep_3], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf62 = torch.ops.vllm.unified_kv_cache_update.default(buf61, reinterpret_tensor(buf58, (s72, 16, 128), (6144, 128, 1), 4096), arg33_1)
            buf63 = buf62
            assert_alignment(buf63, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf62
            # Topologically Sorted Source Nodes: [split_3, unified_attention_with_output_3, value_3], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf59, buf61, reinterpret_tensor(buf58, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf60, (s72, 16, 128), (2048, 128, 1), 0), arg33_1, None, None, buf63)
            del arg33_1
            del buf58
            del buf59
            del buf63
            assert_size_stride(arg34_1, (2048, 2048), (2048, 1), 'input')
            buf66 = reinterpret_tensor(buf61, (s72, 2048), (2048, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_3, output_parallel_8], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf60, reinterpret_tensor(arg34_1, (2048, 2048), (1, 2048), 0), out=buf66)
            del arg34_1
            assert_size_stride(arg35_1, (2048, ), (1, ), 'input')
            buf68 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, x_27, x_residual_5, to_50, x_28, pow_8, variance_7, add_25, rsqrt_7, x_29, to_52, x_30], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf66, buf54, buf55, buf48, arg35_1, buf68, s72, 2048, stream=raw_stream0)
            del arg35_1
            assert_size_stride(arg36_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg44_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg52_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg60_1, (60, 2048), (2048, 1), 'input')
            buf69 = buf51; del buf51  # reuse
            buf88 = buf32; del buf32  # reuse
            buf107 = buf13; del buf13  # reuse
            buf125 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_19, output_24, output_29, output_34], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_10.run(arg36_1, arg44_1, arg52_1, arg60_1, buf69, buf88, buf107, buf125, stream=raw_stream0)
            del arg36_1
            del arg44_1
            del arg52_1
            del arg60_1
            buf70 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [output_19], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf68, buf69, out=buf70)
            del buf69
            # Topologically Sorted Source Nodes: [output_19, moe_forward_shared_3], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf71 = torch.ops.vllm.moe_forward_shared.default(buf68, reinterpret_tensor(buf70, (s72, 60), (64, 1), 0), buf68, None, arg37_1, 0)
            del arg37_1
            buf72 = buf71[0]
            assert_size_stride(buf72, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf72, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf73 = buf71[1]
            assert_size_stride(buf73, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf73, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf71
            assert_size_stride(arg38_1, (2048, ), (1, ), 'input')
            buf74 = buf48; del buf48  # reuse
            buf76 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [result_2, x_23, x_residual_4, to_41, x_24, x_27, x_residual_5, to_50, x_28, result_3, x_31, x_residual_6, to_55, x_32, pow_9, variance_8, add_28, rsqrt_8, x_33, to_57, x_34], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf74, buf72, buf73, buf66, buf54, buf55, arg38_1, buf76, s72, 2048, stream=raw_stream0)
            del arg38_1
            del buf54
            del buf55
            del buf66
            assert_size_stride(arg39_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg40_1, (6144, 2048), (2048, 1), 'input')
            buf77 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_9, variance_8, add_28, rsqrt_8, x_33, to_57, x_34, output_parallel_9], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg39_1, buf76, reinterpret_tensor(arg40_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf77)
            del arg39_1
            del arg40_1
            buf78 = reinterpret_tensor(buf76, (s72, 16, 128), (2048, 128, 1), 0); del buf76  # reuse
            buf80 = reinterpret_tensor(buf73, (s72, 16, 128), (2048, 128, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [split_4, cos_sin_4, chunk_12, query_12, chunk_13, key_12, chunk_14, unsqueeze_16, mul_50, unsqueeze_17, mul_51, o1_8, mul_52, mul_53, o2_8, output_20, unsqueeze_18, mul_54, unsqueeze_19, mul_55, o1_9, mul_56, mul_57, o2_9, output_21], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf77, arg8_1, arg6_1, buf78, buf80, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf79 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [split_4, value_4, kv_cache_dummy_dep_4], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf81 = torch.ops.vllm.unified_kv_cache_update.default(buf80, reinterpret_tensor(buf77, (s72, 16, 128), (6144, 128, 1), 4096), arg41_1)
            buf82 = buf81
            assert_alignment(buf82, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf81
            # Topologically Sorted Source Nodes: [split_4, unified_attention_with_output_4, value_4], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf78, buf80, reinterpret_tensor(buf77, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf79, (s72, 16, 128), (2048, 128, 1), 0), arg41_1, None, None, buf82)
            del arg41_1
            del buf82
            assert_size_stride(arg42_1, (2048, 2048), (2048, 1), 'input')
            buf85 = reinterpret_tensor(buf80, (s72, 2048), (2048, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_4, output_parallel_10], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf79, reinterpret_tensor(arg42_1, (2048, 2048), (1, 2048), 0), out=buf85)
            del arg42_1
            assert_size_stride(arg43_1, (2048, ), (1, ), 'input')
            buf87 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [x_35, x_residual_7, to_64, x_36, pow_10, variance_9, add_32, rsqrt_9, x_37, to_66, x_38], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf85, buf74, arg43_1, buf87, s72, 2048, stream=raw_stream0)
            del arg43_1
            buf89 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [output_24], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf87, buf88, out=buf89)
            # Topologically Sorted Source Nodes: [output_24, moe_forward_shared_4], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf90 = torch.ops.vllm.moe_forward_shared.default(buf87, reinterpret_tensor(buf89, (s72, 60), (64, 1), 0), buf87, None, arg45_1, 0)
            del arg45_1
            del buf89
            buf91 = buf90[0]
            assert_size_stride(buf91, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf91, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf92 = buf90[1]
            assert_size_stride(buf92, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf92, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf90
            assert_size_stride(arg46_1, (2048, ), (1, ), 'input')
            buf94 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [x_35, x_residual_7, to_64, x_36, result_4, x_39, x_residual_8, to_69, x_40, pow_11, variance_10, add_35, rsqrt_10, x_41, to_71, x_42], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf91, buf92, buf85, buf74, arg46_1, buf94, s72, 2048, stream=raw_stream0)
            del arg46_1
            assert_size_stride(arg47_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg48_1, (6144, 2048), (2048, 1), 'input')
            buf95 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [x_35, x_residual_7, to_64, x_36, result_4, x_39, x_residual_8, to_69, x_40, pow_11, variance_10, add_35, rsqrt_10, x_41, to_71, x_42, output_parallel_11], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg47_1, buf94, reinterpret_tensor(arg48_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf95)
            del arg47_1
            del arg48_1
            buf96 = reinterpret_tensor(buf94, (s72, 16, 128), (2048, 128, 1), 0); del buf94  # reuse
            buf98 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [split_5, cos_sin_5, chunk_15, query_15, chunk_16, key_15, chunk_17, unsqueeze_20, mul_62, unsqueeze_21, mul_63, o1_10, mul_64, mul_65, o2_10, output_25, unsqueeze_22, mul_66, unsqueeze_23, mul_67, o1_11, mul_68, mul_69, o2_11, output_26], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf95, arg8_1, arg6_1, buf96, buf98, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf97 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_5, value_5, kv_cache_dummy_dep_5], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf99 = torch.ops.vllm.unified_kv_cache_update.default(buf98, reinterpret_tensor(buf95, (s72, 16, 128), (6144, 128, 1), 4096), arg49_1)
            buf100 = buf99
            assert_alignment(buf100, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf99
            # Topologically Sorted Source Nodes: [split_5, unified_attention_with_output_5, value_5], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf96, buf98, reinterpret_tensor(buf95, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf97, (s72, 16, 128), (2048, 128, 1), 0), arg49_1, None, None, buf100)
            del arg49_1
            del buf100
            del buf95
            del buf96
            assert_size_stride(arg50_1, (2048, 2048), (2048, 1), 'input')
            buf103 = reinterpret_tensor(buf98, (s72, 2048), (2048, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_5, output_parallel_12], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf97, reinterpret_tensor(arg50_1, (2048, 2048), (1, 2048), 0), out=buf103)
            del arg50_1
            assert_size_stride(arg51_1, (2048, ), (1, ), 'input')
            buf104 = buf74; del buf74  # reuse
            buf106 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [x_35, x_residual_7, to_64, x_36, result_4, x_39, x_residual_8, to_69, x_40, x_43, x_residual_9, to_78, x_44, pow_12, variance_11, add_39, rsqrt_11, x_45, to_80, x_46], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf104, buf103, buf91, buf92, buf85, arg51_1, buf106, s72, 2048, stream=raw_stream0)
            del arg51_1
            del buf103
            del buf85
            buf108 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_29], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf106, buf107, out=buf108)
            # Topologically Sorted Source Nodes: [output_29, moe_forward_shared_5], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf109 = torch.ops.vllm.moe_forward_shared.default(buf106, reinterpret_tensor(buf108, (s72, 60), (64, 1), 0), buf106, None, arg53_1, 0)
            del arg53_1
            buf110 = buf109[0]
            assert_size_stride(buf110, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf110, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf111 = buf109[1]
            assert_size_stride(buf111, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf111, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf109
            assert_size_stride(arg54_1, (2048, ), (1, ), 'input')
            buf113 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [result_5, x_47, x_residual_10, to_83, x_48, pow_13, variance_12, add_42, rsqrt_12, x_49, to_85, x_50], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf110, buf111, buf104, arg54_1, buf113, s72, 2048, stream=raw_stream0)
            del arg54_1
            assert_size_stride(arg55_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg56_1, (6144, 2048), (2048, 1), 'input')
            buf114 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_5, x_47, x_residual_10, to_83, x_48, pow_13, variance_12, add_42, rsqrt_12, x_49, to_85, x_50, output_parallel_13], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg55_1, buf113, reinterpret_tensor(arg56_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf114)
            del arg55_1
            del arg56_1
            buf115 = reinterpret_tensor(buf113, (s72, 16, 128), (2048, 128, 1), 0); del buf113  # reuse
            buf117 = reinterpret_tensor(buf92, (s72, 16, 128), (2048, 128, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [split_6, cos_sin_6, chunk_18, query_18, chunk_19, key_18, chunk_20, unsqueeze_24, mul_74, unsqueeze_25, mul_75, o1_12, mul_76, mul_77, o2_12, output_30, unsqueeze_26, mul_78, unsqueeze_27, mul_79, o1_13, mul_80, mul_81, o2_13, output_31], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf114, arg8_1, arg6_1, buf115, buf117, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf116 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [split_6, value_6, kv_cache_dummy_dep_6], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf118 = torch.ops.vllm.unified_kv_cache_update.default(buf117, reinterpret_tensor(buf114, (s72, 16, 128), (6144, 128, 1), 4096), arg57_1)
            buf119 = buf118
            assert_alignment(buf119, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf118
            # Topologically Sorted Source Nodes: [split_6, unified_attention_with_output_6, value_6], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf115, buf117, reinterpret_tensor(buf114, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf116, (s72, 16, 128), (2048, 128, 1), 0), arg57_1, None, None, buf119)
            del arg57_1
            del buf114
            del buf115
            del buf119
            assert_size_stride(arg58_1, (2048, 2048), (2048, 1), 'input')
            buf122 = reinterpret_tensor(buf117, (s72, 2048), (2048, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_6, output_parallel_14], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf116, reinterpret_tensor(arg58_1, (2048, 2048), (1, 2048), 0), out=buf122)
            del arg58_1
            assert_size_stride(arg59_1, (2048, ), (1, ), 'input')
            buf124 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [result_5, x_47, x_residual_10, to_83, x_48, x_51, x_residual_11, to_92, x_52, pow_14, variance_13, add_46, rsqrt_13, x_53, to_94, x_54], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf122, buf110, buf111, buf104, arg59_1, buf124, s72, 2048, stream=raw_stream0)
            del arg59_1
            buf126 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [output_34], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf124, buf125, out=buf126)
            # Topologically Sorted Source Nodes: [output_34, moe_forward_shared_6], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf127 = torch.ops.vllm.moe_forward_shared.default(buf124, reinterpret_tensor(buf126, (s72, 60), (64, 1), 0), buf124, None, arg61_1, 0)
            del arg61_1
            buf128 = buf127[0]
            assert_size_stride(buf128, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf128, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf129 = buf127[1]
            assert_size_stride(buf129, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf129, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf127
            assert_size_stride(arg62_1, (2048, ), (1, ), 'input')
            buf130 = buf104; del buf104  # reuse
            buf132 = buf124; del buf124  # reuse
            # Topologically Sorted Source Nodes: [result_5, x_47, x_residual_10, to_83, x_48, x_51, x_residual_11, to_92, x_52, result_6, x_55, x_residual_12, to_97, x_56, pow_15, variance_14, add_49, rsqrt_14, x_57, to_99, x_58], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf130, buf128, buf129, buf122, buf110, buf111, arg62_1, buf132, s72, 2048, stream=raw_stream0)
            del arg62_1
            del buf110
            del buf111
            del buf122
            assert_size_stride(arg63_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg64_1, (6144, 2048), (2048, 1), 'input')
            buf133 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_15, variance_14, add_49, rsqrt_14, x_57, to_99, x_58, output_parallel_15], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg63_1, buf132, reinterpret_tensor(arg64_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf133)
            del arg63_1
            del arg64_1
            buf134 = reinterpret_tensor(buf132, (s72, 16, 128), (2048, 128, 1), 0); del buf132  # reuse
            buf136 = reinterpret_tensor(buf129, (s72, 16, 128), (2048, 128, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [split_7, cos_sin_7, chunk_21, query_21, chunk_22, key_21, chunk_23, unsqueeze_28, mul_86, unsqueeze_29, mul_87, o1_14, mul_88, mul_89, o2_14, output_35, unsqueeze_30, mul_90, unsqueeze_31, mul_91, o1_15, mul_92, mul_93, o2_15, output_36], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf133, arg8_1, arg6_1, buf134, buf136, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf135 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [split_7, value_7, kv_cache_dummy_dep_7], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf137 = torch.ops.vllm.unified_kv_cache_update.default(buf136, reinterpret_tensor(buf133, (s72, 16, 128), (6144, 128, 1), 4096), arg65_1)
            buf138 = buf137
            assert_alignment(buf138, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf137
            # Topologically Sorted Source Nodes: [split_7, unified_attention_with_output_7, value_7], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf134, buf136, reinterpret_tensor(buf133, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf135, (s72, 16, 128), (2048, 128, 1), 0), arg65_1, None, None, buf138)
            del arg65_1
            del buf138
            assert_size_stride(arg66_1, (2048, 2048), (2048, 1), 'input')
            buf141 = reinterpret_tensor(buf136, (s72, 2048), (2048, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_7, output_parallel_16], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf135, reinterpret_tensor(arg66_1, (2048, 2048), (1, 2048), 0), out=buf141)
            del arg66_1
            assert_size_stride(arg67_1, (2048, ), (1, ), 'input')
            buf143 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [x_59, x_residual_13, to_106, x_60, pow_16, variance_15, add_53, rsqrt_15, x_61, to_108, x_62], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf141, buf130, arg67_1, buf143, s72, 2048, stream=raw_stream0)
            del arg67_1
            assert_size_stride(arg68_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg76_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg84_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg92_1, (60, 2048), (2048, 1), 'input')
            buf144 = buf125; del buf125  # reuse
            buf163 = buf107; del buf107  # reuse
            buf181 = buf88; del buf88  # reuse
            buf200 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_39, output_44, output_49, output_54], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_10.run(arg68_1, arg76_1, arg84_1, arg92_1, buf144, buf163, buf181, buf200, stream=raw_stream0)
            del arg68_1
            del arg76_1
            del arg84_1
            del arg92_1
            buf145 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [output_39], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf143, buf144, out=buf145)
            del buf144
            # Topologically Sorted Source Nodes: [output_39, moe_forward_shared_7], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf146 = torch.ops.vllm.moe_forward_shared.default(buf143, reinterpret_tensor(buf145, (s72, 60), (64, 1), 0), buf143, None, arg69_1, 0)
            del arg69_1
            del buf145
            buf147 = buf146[0]
            assert_size_stride(buf147, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf147, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf148 = buf146[1]
            assert_size_stride(buf148, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf148, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf146
            assert_size_stride(arg70_1, (2048, ), (1, ), 'input')
            buf150 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [x_59, x_residual_13, to_106, x_60, result_7, x_63, x_residual_14, to_111, x_64, pow_17, variance_16, add_56, rsqrt_16, x_65, to_113, x_66], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf147, buf148, buf141, buf130, arg70_1, buf150, s72, 2048, stream=raw_stream0)
            del arg70_1
            assert_size_stride(arg71_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg72_1, (6144, 2048), (2048, 1), 'input')
            buf151 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [x_59, x_residual_13, to_106, x_60, result_7, x_63, x_residual_14, to_111, x_64, pow_17, variance_16, add_56, rsqrt_16, x_65, to_113, x_66, output_parallel_17], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg71_1, buf150, reinterpret_tensor(arg72_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf151)
            del arg71_1
            del arg72_1
            buf152 = reinterpret_tensor(buf150, (s72, 16, 128), (2048, 128, 1), 0); del buf150  # reuse
            buf154 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [split_8, cos_sin_8, chunk_24, query_24, chunk_25, key_24, chunk_26, unsqueeze_32, mul_98, unsqueeze_33, mul_99, o1_16, mul_100, mul_101, o2_16, output_40, unsqueeze_34, mul_102, unsqueeze_35, mul_103, o1_17, mul_104, mul_105, o2_17, output_41], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf151, arg8_1, arg6_1, buf152, buf154, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf153 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_8, value_8, kv_cache_dummy_dep_8], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf155 = torch.ops.vllm.unified_kv_cache_update.default(buf154, reinterpret_tensor(buf151, (s72, 16, 128), (6144, 128, 1), 4096), arg73_1)
            buf156 = buf155
            assert_alignment(buf156, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf155
            # Topologically Sorted Source Nodes: [split_8, unified_attention_with_output_8, value_8], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf152, buf154, reinterpret_tensor(buf151, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf153, (s72, 16, 128), (2048, 128, 1), 0), arg73_1, None, None, buf156)
            del arg73_1
            del buf151
            del buf152
            del buf156
            assert_size_stride(arg74_1, (2048, 2048), (2048, 1), 'input')
            buf159 = reinterpret_tensor(buf154, (s72, 2048), (2048, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_8, output_parallel_18], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf153, reinterpret_tensor(arg74_1, (2048, 2048), (1, 2048), 0), out=buf159)
            del arg74_1
            assert_size_stride(arg75_1, (2048, ), (1, ), 'input')
            buf160 = buf130; del buf130  # reuse
            buf162 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [x_59, x_residual_13, to_106, x_60, result_7, x_63, x_residual_14, to_111, x_64, x_67, x_residual_15, to_120, x_68, pow_18, variance_17, add_60, rsqrt_17, x_69, to_122, x_70], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf160, buf159, buf147, buf148, buf141, arg75_1, buf162, s72, 2048, stream=raw_stream0)
            del arg75_1
            del buf141
            del buf147
            buf164 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_44], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf162, buf163, out=buf164)
            # Topologically Sorted Source Nodes: [output_44, moe_forward_shared_8], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf165 = torch.ops.vllm.moe_forward_shared.default(buf162, reinterpret_tensor(buf164, (s72, 60), (64, 1), 0), buf162, None, arg77_1, 0)
            del arg77_1
            buf166 = buf165[0]
            assert_size_stride(buf166, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf166, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf167 = buf165[1]
            assert_size_stride(buf167, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf167, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf165
            assert_size_stride(arg78_1, (2048, ), (1, ), 'input')
            buf169 = buf162; del buf162  # reuse
            # Topologically Sorted Source Nodes: [result_8, x_71, x_residual_16, to_125, x_72, pow_19, variance_18, add_63, rsqrt_18, x_73, to_127, x_74], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf166, buf167, buf160, arg78_1, buf169, s72, 2048, stream=raw_stream0)
            del arg78_1
            assert_size_stride(arg79_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg80_1, (6144, 2048), (2048, 1), 'input')
            buf170 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_8, x_71, x_residual_16, to_125, x_72, pow_19, variance_18, add_63, rsqrt_18, x_73, to_127, x_74, output_parallel_19], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg79_1, buf169, reinterpret_tensor(arg80_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf170)
            del arg79_1
            del arg80_1
            buf171 = reinterpret_tensor(buf169, (s72, 16, 128), (2048, 128, 1), 0); del buf169  # reuse
            buf173 = reinterpret_tensor(buf159, (s72, 16, 128), (2048, 128, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [split_9, cos_sin_9, chunk_27, query_27, chunk_28, key_27, chunk_29, unsqueeze_36, mul_110, unsqueeze_37, mul_111, o1_18, mul_112, mul_113, o2_18, output_45, unsqueeze_38, mul_114, unsqueeze_39, mul_115, o1_19, mul_116, mul_117, o2_19, output_46], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf170, arg8_1, arg6_1, buf171, buf173, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf172 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [split_9, value_9, kv_cache_dummy_dep_9], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf174 = torch.ops.vllm.unified_kv_cache_update.default(buf173, reinterpret_tensor(buf170, (s72, 16, 128), (6144, 128, 1), 4096), arg81_1)
            buf175 = buf174
            assert_alignment(buf175, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf174
            # Topologically Sorted Source Nodes: [split_9, unified_attention_with_output_9, value_9], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf171, buf173, reinterpret_tensor(buf170, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf172, (s72, 16, 128), (2048, 128, 1), 0), arg81_1, None, None, buf175)
            del arg81_1
            del buf170
            del buf171
            del buf175
            assert_size_stride(arg82_1, (2048, 2048), (2048, 1), 'input')
            buf178 = reinterpret_tensor(buf173, (s72, 2048), (2048, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_9, output_parallel_20], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf172, reinterpret_tensor(arg82_1, (2048, 2048), (1, 2048), 0), out=buf178)
            del arg82_1
            assert_size_stride(arg83_1, (2048, ), (1, ), 'input')
            buf180 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [result_8, x_71, x_residual_16, to_125, x_72, x_75, x_residual_17, to_134, x_76, pow_20, variance_19, add_67, rsqrt_19, x_77, to_136, x_78], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf178, buf166, buf167, buf160, arg83_1, buf180, s72, 2048, stream=raw_stream0)
            del arg83_1
            buf182 = buf164; del buf164  # reuse
            # Topologically Sorted Source Nodes: [output_49], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf180, buf181, out=buf182)
            # Topologically Sorted Source Nodes: [output_49, moe_forward_shared_9], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf183 = torch.ops.vllm.moe_forward_shared.default(buf180, reinterpret_tensor(buf182, (s72, 60), (64, 1), 0), buf180, None, arg85_1, 0)
            del arg85_1
            buf184 = buf183[0]
            assert_size_stride(buf184, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf184, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf185 = buf183[1]
            assert_size_stride(buf185, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf185, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf183
            assert_size_stride(arg86_1, (2048, ), (1, ), 'input')
            buf186 = buf160; del buf160  # reuse
            buf188 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [result_8, x_71, x_residual_16, to_125, x_72, x_75, x_residual_17, to_134, x_76, result_9, x_79, x_residual_18, to_139, x_80, pow_21, variance_20, add_70, rsqrt_20, x_81, to_141, x_82], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf186, buf184, buf185, buf178, buf166, buf167, arg86_1, buf188, s72, 2048, stream=raw_stream0)
            del arg86_1
            del buf166
            del buf167
            del buf178
            assert_size_stride(arg87_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg88_1, (6144, 2048), (2048, 1), 'input')
            buf189 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_21, variance_20, add_70, rsqrt_20, x_81, to_141, x_82, output_parallel_21], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg87_1, buf188, reinterpret_tensor(arg88_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf189)
            del arg87_1
            del arg88_1
            buf190 = reinterpret_tensor(buf188, (s72, 16, 128), (2048, 128, 1), 0); del buf188  # reuse
            buf192 = reinterpret_tensor(buf185, (s72, 16, 128), (2048, 128, 1), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [split_10, cos_sin_10, chunk_30, query_30, chunk_31, key_30, chunk_32, unsqueeze_40, mul_122, unsqueeze_41, mul_123, o1_20, mul_124, mul_125, o2_20, output_50, unsqueeze_42, mul_126, unsqueeze_43, mul_127, o1_21, mul_128, mul_129, o2_21, output_51], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf189, arg8_1, arg6_1, buf190, buf192, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf191 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [split_10, value_10, kv_cache_dummy_dep_10], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf193 = torch.ops.vllm.unified_kv_cache_update.default(buf192, reinterpret_tensor(buf189, (s72, 16, 128), (6144, 128, 1), 4096), arg89_1)
            buf194 = buf193
            assert_alignment(buf194, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf193
            # Topologically Sorted Source Nodes: [split_10, unified_attention_with_output_10, value_10], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf190, buf192, reinterpret_tensor(buf189, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf191, (s72, 16, 128), (2048, 128, 1), 0), arg89_1, None, None, buf194)
            del arg89_1
            del buf194
            assert_size_stride(arg90_1, (2048, 2048), (2048, 1), 'input')
            buf197 = reinterpret_tensor(buf192, (s72, 2048), (2048, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_10, output_parallel_22], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf191, reinterpret_tensor(arg90_1, (2048, 2048), (1, 2048), 0), out=buf197)
            del arg90_1
            assert_size_stride(arg91_1, (2048, ), (1, ), 'input')
            buf199 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [x_83, x_residual_19, to_148, x_84, pow_22, variance_21, add_74, rsqrt_21, x_85, to_150, x_86], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf197, buf186, arg91_1, buf199, s72, 2048, stream=raw_stream0)
            del arg91_1
            buf201 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [output_54], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf199, buf200, out=buf201)
            # Topologically Sorted Source Nodes: [output_54, moe_forward_shared_10], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf202 = torch.ops.vllm.moe_forward_shared.default(buf199, reinterpret_tensor(buf201, (s72, 60), (64, 1), 0), buf199, None, arg93_1, 0)
            del arg93_1
            del buf201
            buf203 = buf202[0]
            assert_size_stride(buf203, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf203, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf204 = buf202[1]
            assert_size_stride(buf204, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf204, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf202
            assert_size_stride(arg94_1, (2048, ), (1, ), 'input')
            buf206 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [x_83, x_residual_19, to_148, x_84, result_10, x_87, x_residual_20, to_153, x_88, pow_23, variance_22, add_77, rsqrt_22, x_89, to_155, x_90], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf203, buf204, buf197, buf186, arg94_1, buf206, s72, 2048, stream=raw_stream0)
            del arg94_1
            assert_size_stride(arg95_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg96_1, (6144, 2048), (2048, 1), 'input')
            buf207 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [x_83, x_residual_19, to_148, x_84, result_10, x_87, x_residual_20, to_153, x_88, pow_23, variance_22, add_77, rsqrt_22, x_89, to_155, x_90, output_parallel_23], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg95_1, buf206, reinterpret_tensor(arg96_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf207)
            del arg95_1
            del arg96_1
            buf208 = reinterpret_tensor(buf206, (s72, 16, 128), (2048, 128, 1), 0); del buf206  # reuse
            buf210 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [split_11, cos_sin_11, chunk_33, query_33, chunk_34, key_33, chunk_35, unsqueeze_44, mul_134, unsqueeze_45, mul_135, o1_22, mul_136, mul_137, o2_22, output_55, unsqueeze_46, mul_138, unsqueeze_47, mul_139, o1_23, mul_140, mul_141, o2_23, output_56], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf207, arg8_1, arg6_1, buf208, buf210, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf209 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_11, value_11, kv_cache_dummy_dep_11], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf211 = torch.ops.vllm.unified_kv_cache_update.default(buf210, reinterpret_tensor(buf207, (s72, 16, 128), (6144, 128, 1), 4096), arg97_1)
            buf212 = buf211
            assert_alignment(buf212, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf211
            # Topologically Sorted Source Nodes: [split_11, unified_attention_with_output_11, value_11], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf208, buf210, reinterpret_tensor(buf207, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf209, (s72, 16, 128), (2048, 128, 1), 0), arg97_1, None, None, buf212)
            del arg97_1
            del buf207
            del buf208
            del buf212
            assert_size_stride(arg98_1, (2048, 2048), (2048, 1), 'input')
            buf215 = reinterpret_tensor(buf210, (s72, 2048), (2048, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_11, output_parallel_24], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf209, reinterpret_tensor(arg98_1, (2048, 2048), (1, 2048), 0), out=buf215)
            del arg98_1
            assert_size_stride(arg99_1, (2048, ), (1, ), 'input')
            buf216 = buf186; del buf186  # reuse
            buf218 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [x_83, x_residual_19, to_148, x_84, result_10, x_87, x_residual_20, to_153, x_88, x_91, x_residual_21, to_162, x_92, pow_24, variance_23, add_81, rsqrt_23, x_93, to_164, x_94], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf216, buf215, buf203, buf204, buf197, arg99_1, buf218, s72, 2048, stream=raw_stream0)
            del arg99_1
            del buf197
            del buf203
            assert_size_stride(arg100_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg108_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg116_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg124_1, (60, 2048), (2048, 1), 'input')
            buf219 = buf200; del buf200  # reuse
            buf237 = buf181; del buf181  # reuse
            buf256 = buf163; del buf163  # reuse
            buf275 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_59, output_64, output_69, output_74], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_10.run(arg100_1, arg108_1, arg116_1, arg124_1, buf219, buf237, buf256, buf275, stream=raw_stream0)
            del arg100_1
            del arg108_1
            del arg116_1
            del arg124_1
            buf220 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_59], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf218, buf219, out=buf220)
            del buf219
            # Topologically Sorted Source Nodes: [output_59, moe_forward_shared_11], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf221 = torch.ops.vllm.moe_forward_shared.default(buf218, reinterpret_tensor(buf220, (s72, 60), (64, 1), 0), buf218, None, arg101_1, 0)
            del arg101_1
            buf222 = buf221[0]
            assert_size_stride(buf222, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf222, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf223 = buf221[1]
            assert_size_stride(buf223, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf223, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf221
            assert_size_stride(arg102_1, (2048, ), (1, ), 'input')
            buf225 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [result_11, x_95, x_residual_22, to_167, x_96, pow_25, variance_24, add_84, rsqrt_24, x_97, to_169, x_98], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf222, buf223, buf216, arg102_1, buf225, s72, 2048, stream=raw_stream0)
            del arg102_1
            assert_size_stride(arg103_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg104_1, (6144, 2048), (2048, 1), 'input')
            buf226 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_11, x_95, x_residual_22, to_167, x_96, pow_25, variance_24, add_84, rsqrt_24, x_97, to_169, x_98, output_parallel_25], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg103_1, buf225, reinterpret_tensor(arg104_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf226)
            del arg103_1
            del arg104_1
            buf227 = reinterpret_tensor(buf225, (s72, 16, 128), (2048, 128, 1), 0); del buf225  # reuse
            buf229 = reinterpret_tensor(buf215, (s72, 16, 128), (2048, 128, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [split_12, cos_sin_12, chunk_36, query_36, chunk_37, key_36, chunk_38, unsqueeze_48, mul_146, unsqueeze_49, mul_147, o1_24, mul_148, mul_149, o2_24, output_60, unsqueeze_50, mul_150, unsqueeze_51, mul_151, o1_25, mul_152, mul_153, o2_25, output_61], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf226, arg8_1, arg6_1, buf227, buf229, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf228 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [split_12, value_12, kv_cache_dummy_dep_12], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf230 = torch.ops.vllm.unified_kv_cache_update.default(buf229, reinterpret_tensor(buf226, (s72, 16, 128), (6144, 128, 1), 4096), arg105_1)
            buf231 = buf230
            assert_alignment(buf231, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf230
            # Topologically Sorted Source Nodes: [split_12, unified_attention_with_output_12, value_12], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf227, buf229, reinterpret_tensor(buf226, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf228, (s72, 16, 128), (2048, 128, 1), 0), arg105_1, None, None, buf231)
            del arg105_1
            del buf226
            del buf227
            del buf231
            assert_size_stride(arg106_1, (2048, 2048), (2048, 1), 'input')
            buf234 = reinterpret_tensor(buf229, (s72, 2048), (2048, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_12, output_parallel_26], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf228, reinterpret_tensor(arg106_1, (2048, 2048), (1, 2048), 0), out=buf234)
            del arg106_1
            assert_size_stride(arg107_1, (2048, ), (1, ), 'input')
            buf236 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [result_11, x_95, x_residual_22, to_167, x_96, x_99, x_residual_23, to_176, x_100, pow_26, variance_25, add_88, rsqrt_25, x_101, to_178, x_102], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf234, buf222, buf223, buf216, arg107_1, buf236, s72, 2048, stream=raw_stream0)
            del arg107_1
            buf238 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [output_64], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf236, buf237, out=buf238)
            # Topologically Sorted Source Nodes: [output_64, moe_forward_shared_12], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf239 = torch.ops.vllm.moe_forward_shared.default(buf236, reinterpret_tensor(buf238, (s72, 60), (64, 1), 0), buf236, None, arg109_1, 0)
            del arg109_1
            buf240 = buf239[0]
            assert_size_stride(buf240, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf240, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf241 = buf239[1]
            assert_size_stride(buf241, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf241, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf239
            assert_size_stride(arg110_1, (2048, ), (1, ), 'input')
            buf242 = buf216; del buf216  # reuse
            buf244 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [result_11, x_95, x_residual_22, to_167, x_96, x_99, x_residual_23, to_176, x_100, result_12, x_103, x_residual_24, to_181, x_104, pow_27, variance_26, add_91, rsqrt_26, x_105, to_183, x_106], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf242, buf240, buf241, buf234, buf222, buf223, arg110_1, buf244, s72, 2048, stream=raw_stream0)
            del arg110_1
            del buf222
            del buf223
            del buf234
            assert_size_stride(arg111_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg112_1, (6144, 2048), (2048, 1), 'input')
            buf245 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_27, variance_26, add_91, rsqrt_26, x_105, to_183, x_106, output_parallel_27], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg111_1, buf244, reinterpret_tensor(arg112_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf245)
            del arg111_1
            del arg112_1
            buf246 = reinterpret_tensor(buf244, (s72, 16, 128), (2048, 128, 1), 0); del buf244  # reuse
            buf248 = reinterpret_tensor(buf241, (s72, 16, 128), (2048, 128, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [split_13, cos_sin_13, chunk_39, query_39, chunk_40, key_39, chunk_41, unsqueeze_52, mul_158, unsqueeze_53, mul_159, o1_26, mul_160, mul_161, o2_26, output_65, unsqueeze_54, mul_162, unsqueeze_55, mul_163, o1_27, mul_164, mul_165, o2_27, output_66], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf245, arg8_1, arg6_1, buf246, buf248, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf247 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [split_13, value_13, kv_cache_dummy_dep_13], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf249 = torch.ops.vllm.unified_kv_cache_update.default(buf248, reinterpret_tensor(buf245, (s72, 16, 128), (6144, 128, 1), 4096), arg113_1)
            buf250 = buf249
            assert_alignment(buf250, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf249
            # Topologically Sorted Source Nodes: [split_13, unified_attention_with_output_13, value_13], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf246, buf248, reinterpret_tensor(buf245, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf247, (s72, 16, 128), (2048, 128, 1), 0), arg113_1, None, None, buf250)
            del arg113_1
            del buf250
            assert_size_stride(arg114_1, (2048, 2048), (2048, 1), 'input')
            buf253 = reinterpret_tensor(buf248, (s72, 2048), (2048, 1), 0); del buf248  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_13, output_parallel_28], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf247, reinterpret_tensor(arg114_1, (2048, 2048), (1, 2048), 0), out=buf253)
            del arg114_1
            assert_size_stride(arg115_1, (2048, ), (1, ), 'input')
            buf255 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_residual_25, to_190, x_108, pow_28, variance_27, add_95, rsqrt_27, x_109, to_192, x_110], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf253, buf242, arg115_1, buf255, s72, 2048, stream=raw_stream0)
            del arg115_1
            buf257 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [output_69], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf255, buf256, out=buf257)
            # Topologically Sorted Source Nodes: [output_69, moe_forward_shared_13], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf258 = torch.ops.vllm.moe_forward_shared.default(buf255, reinterpret_tensor(buf257, (s72, 60), (64, 1), 0), buf255, None, arg117_1, 0)
            del arg117_1
            del buf257
            buf259 = buf258[0]
            assert_size_stride(buf259, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf259, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf260 = buf258[1]
            assert_size_stride(buf260, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf260, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf258
            assert_size_stride(arg118_1, (2048, ), (1, ), 'input')
            buf262 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_residual_25, to_190, x_108, result_13, x_111, x_residual_26, to_195, x_112, pow_29, variance_28, add_98, rsqrt_28, x_113, to_197, x_114], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf259, buf260, buf253, buf242, arg118_1, buf262, s72, 2048, stream=raw_stream0)
            del arg118_1
            assert_size_stride(arg119_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg120_1, (6144, 2048), (2048, 1), 'input')
            buf263 = buf245; del buf245  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_residual_25, to_190, x_108, result_13, x_111, x_residual_26, to_195, x_112, pow_29, variance_28, add_98, rsqrt_28, x_113, to_197, x_114, output_parallel_29], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg119_1, buf262, reinterpret_tensor(arg120_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf263)
            del arg119_1
            del arg120_1
            buf264 = reinterpret_tensor(buf262, (s72, 16, 128), (2048, 128, 1), 0); del buf262  # reuse
            buf266 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [split_14, cos_sin_14, chunk_42, query_42, chunk_43, key_42, chunk_44, unsqueeze_56, mul_170, unsqueeze_57, mul_171, o1_28, mul_172, mul_173, o2_28, output_70, unsqueeze_58, mul_174, unsqueeze_59, mul_175, o1_29, mul_176, mul_177, o2_29, output_71], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf263, arg8_1, arg6_1, buf264, buf266, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf265 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_14, value_14, kv_cache_dummy_dep_14], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf267 = torch.ops.vllm.unified_kv_cache_update.default(buf266, reinterpret_tensor(buf263, (s72, 16, 128), (6144, 128, 1), 4096), arg121_1)
            buf268 = buf267
            assert_alignment(buf268, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf267
            # Topologically Sorted Source Nodes: [split_14, unified_attention_with_output_14, value_14], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf264, buf266, reinterpret_tensor(buf263, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf265, (s72, 16, 128), (2048, 128, 1), 0), arg121_1, None, None, buf268)
            del arg121_1
            del buf263
            del buf264
            del buf268
            assert_size_stride(arg122_1, (2048, 2048), (2048, 1), 'input')
            buf271 = reinterpret_tensor(buf266, (s72, 2048), (2048, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_14, output_parallel_30], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf265, reinterpret_tensor(arg122_1, (2048, 2048), (1, 2048), 0), out=buf271)
            del arg122_1
            assert_size_stride(arg123_1, (2048, ), (1, ), 'input')
            buf272 = buf242; del buf242  # reuse
            buf274 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_residual_25, to_190, x_108, result_13, x_111, x_residual_26, to_195, x_112, x_115, x_residual_27, to_204, x_116, pow_30, variance_29, add_102, rsqrt_29, x_117, to_206, x_118], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf272, buf271, buf259, buf260, buf253, arg123_1, buf274, s72, 2048, stream=raw_stream0)
            del arg123_1
            del buf253
            del buf259
            buf276 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_74], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf274, buf275, out=buf276)
            # Topologically Sorted Source Nodes: [output_74, moe_forward_shared_14], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf277 = torch.ops.vllm.moe_forward_shared.default(buf274, reinterpret_tensor(buf276, (s72, 60), (64, 1), 0), buf274, None, arg125_1, 0)
            del arg125_1
            buf278 = buf277[0]
            assert_size_stride(buf278, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf278, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf279 = buf277[1]
            assert_size_stride(buf279, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf279, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf277
            assert_size_stride(arg126_1, (2048, ), (1, ), 'input')
            buf281 = buf274; del buf274  # reuse
            # Topologically Sorted Source Nodes: [result_14, x_119, x_residual_28, to_209, x_120, pow_31, variance_30, add_105, rsqrt_30, x_121, to_211, x_122], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf278, buf279, buf272, arg126_1, buf281, s72, 2048, stream=raw_stream0)
            del arg126_1
            assert_size_stride(arg127_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg128_1, (6144, 2048), (2048, 1), 'input')
            buf282 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_14, x_119, x_residual_28, to_209, x_120, pow_31, variance_30, add_105, rsqrt_30, x_121, to_211, x_122, output_parallel_31], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg127_1, buf281, reinterpret_tensor(arg128_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf282)
            del arg127_1
            del arg128_1
            buf283 = reinterpret_tensor(buf281, (s72, 16, 128), (2048, 128, 1), 0); del buf281  # reuse
            buf285 = reinterpret_tensor(buf271, (s72, 16, 128), (2048, 128, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [split_15, cos_sin_15, chunk_45, query_45, chunk_46, key_45, chunk_47, unsqueeze_60, mul_182, unsqueeze_61, mul_183, o1_30, mul_184, mul_185, o2_30, output_75, unsqueeze_62, mul_186, unsqueeze_63, mul_187, o1_31, mul_188, mul_189, o2_31, output_76], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf282, arg8_1, arg6_1, buf283, buf285, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf284 = buf260; del buf260  # reuse
            # Topologically Sorted Source Nodes: [split_15, value_15, kv_cache_dummy_dep_15], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf286 = torch.ops.vllm.unified_kv_cache_update.default(buf285, reinterpret_tensor(buf282, (s72, 16, 128), (6144, 128, 1), 4096), arg129_1)
            buf287 = buf286
            assert_alignment(buf287, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf286
            # Topologically Sorted Source Nodes: [split_15, unified_attention_with_output_15, value_15], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf283, buf285, reinterpret_tensor(buf282, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf284, (s72, 16, 128), (2048, 128, 1), 0), arg129_1, None, None, buf287)
            del arg129_1
            del buf282
            del buf283
            del buf287
            assert_size_stride(arg130_1, (2048, 2048), (2048, 1), 'input')
            buf290 = reinterpret_tensor(buf285, (s72, 2048), (2048, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_15, output_parallel_32], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf284, reinterpret_tensor(arg130_1, (2048, 2048), (1, 2048), 0), out=buf290)
            del arg130_1
            assert_size_stride(arg131_1, (2048, ), (1, ), 'input')
            buf292 = buf284; del buf284  # reuse
            # Topologically Sorted Source Nodes: [result_14, x_119, x_residual_28, to_209, x_120, x_123, x_residual_29, to_218, x_124, pow_32, variance_31, add_109, rsqrt_31, x_125, to_220, x_126], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf290, buf278, buf279, buf272, arg131_1, buf292, s72, 2048, stream=raw_stream0)
            del arg131_1
            assert_size_stride(arg132_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg140_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg148_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg156_1, (60, 2048), (2048, 1), 'input')
            buf293 = buf275; del buf275  # reuse
            buf312 = buf256; del buf256  # reuse
            buf331 = buf237; del buf237  # reuse
            buf349 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_79, output_84, output_89, output_94], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_10.run(arg132_1, arg140_1, arg148_1, arg156_1, buf293, buf312, buf331, buf349, stream=raw_stream0)
            del arg132_1
            del arg140_1
            del arg148_1
            del arg156_1
            buf294 = buf276; del buf276  # reuse
            # Topologically Sorted Source Nodes: [output_79], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf292, buf293, out=buf294)
            del buf293
            # Topologically Sorted Source Nodes: [output_79, moe_forward_shared_15], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf295 = torch.ops.vllm.moe_forward_shared.default(buf292, reinterpret_tensor(buf294, (s72, 60), (64, 1), 0), buf292, None, arg133_1, 0)
            del arg133_1
            buf296 = buf295[0]
            assert_size_stride(buf296, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf296, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf297 = buf295[1]
            assert_size_stride(buf297, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf297, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf295
            assert_size_stride(arg134_1, (2048, ), (1, ), 'input')
            buf298 = buf272; del buf272  # reuse
            buf300 = buf292; del buf292  # reuse
            # Topologically Sorted Source Nodes: [result_14, x_119, x_residual_28, to_209, x_120, x_123, x_residual_29, to_218, x_124, result_15, x_127, x_residual_30, to_223, x_128, pow_33, variance_32, add_112, rsqrt_32, x_129, to_225, x_130], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf298, buf296, buf297, buf290, buf278, buf279, arg134_1, buf300, s72, 2048, stream=raw_stream0)
            del arg134_1
            del buf278
            del buf279
            del buf290
            assert_size_stride(arg135_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg136_1, (6144, 2048), (2048, 1), 'input')
            buf301 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_33, variance_32, add_112, rsqrt_32, x_129, to_225, x_130, output_parallel_33], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg135_1, buf300, reinterpret_tensor(arg136_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf301)
            del arg135_1
            del arg136_1
            buf302 = reinterpret_tensor(buf300, (s72, 16, 128), (2048, 128, 1), 0); del buf300  # reuse
            buf304 = reinterpret_tensor(buf297, (s72, 16, 128), (2048, 128, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [split_16, cos_sin_16, chunk_48, query_48, chunk_49, key_48, chunk_50, unsqueeze_64, mul_194, unsqueeze_65, mul_195, o1_32, mul_196, mul_197, o2_32, output_80, unsqueeze_66, mul_198, unsqueeze_67, mul_199, o1_33, mul_200, mul_201, o2_33, output_81], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf301, arg8_1, arg6_1, buf302, buf304, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf303 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [split_16, value_16, kv_cache_dummy_dep_16], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf305 = torch.ops.vllm.unified_kv_cache_update.default(buf304, reinterpret_tensor(buf301, (s72, 16, 128), (6144, 128, 1), 4096), arg137_1)
            buf306 = buf305
            assert_alignment(buf306, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf305
            # Topologically Sorted Source Nodes: [split_16, unified_attention_with_output_16, value_16], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf302, buf304, reinterpret_tensor(buf301, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf303, (s72, 16, 128), (2048, 128, 1), 0), arg137_1, None, None, buf306)
            del arg137_1
            del buf306
            assert_size_stride(arg138_1, (2048, 2048), (2048, 1), 'input')
            buf309 = reinterpret_tensor(buf304, (s72, 2048), (2048, 1), 0); del buf304  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_16, output_parallel_34], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf303, reinterpret_tensor(arg138_1, (2048, 2048), (1, 2048), 0), out=buf309)
            del arg138_1
            assert_size_stride(arg139_1, (2048, ), (1, ), 'input')
            buf311 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [x_131, x_residual_31, to_232, x_132, pow_34, variance_33, add_116, rsqrt_33, x_133, to_234, x_134], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf309, buf298, arg139_1, buf311, s72, 2048, stream=raw_stream0)
            del arg139_1
            buf313 = buf294; del buf294  # reuse
            # Topologically Sorted Source Nodes: [output_84], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf311, buf312, out=buf313)
            # Topologically Sorted Source Nodes: [output_84, moe_forward_shared_16], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf314 = torch.ops.vllm.moe_forward_shared.default(buf311, reinterpret_tensor(buf313, (s72, 60), (64, 1), 0), buf311, None, arg141_1, 0)
            del arg141_1
            del buf313
            buf315 = buf314[0]
            assert_size_stride(buf315, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf315, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf316 = buf314[1]
            assert_size_stride(buf316, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf316, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf314
            assert_size_stride(arg142_1, (2048, ), (1, ), 'input')
            buf318 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [x_131, x_residual_31, to_232, x_132, result_16, x_135, x_residual_32, to_237, x_136, pow_35, variance_34, add_119, rsqrt_34, x_137, to_239, x_138], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf315, buf316, buf309, buf298, arg142_1, buf318, s72, 2048, stream=raw_stream0)
            del arg142_1
            assert_size_stride(arg143_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg144_1, (6144, 2048), (2048, 1), 'input')
            buf319 = buf301; del buf301  # reuse
            # Topologically Sorted Source Nodes: [x_131, x_residual_31, to_232, x_132, result_16, x_135, x_residual_32, to_237, x_136, pow_35, variance_34, add_119, rsqrt_34, x_137, to_239, x_138, output_parallel_35], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg143_1, buf318, reinterpret_tensor(arg144_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf319)
            del arg143_1
            del arg144_1
            buf320 = reinterpret_tensor(buf318, (s72, 16, 128), (2048, 128, 1), 0); del buf318  # reuse
            buf322 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [split_17, cos_sin_17, chunk_51, query_51, chunk_52, key_51, chunk_53, unsqueeze_68, mul_206, unsqueeze_69, mul_207, o1_34, mul_208, mul_209, o2_34, output_85, unsqueeze_70, mul_210, unsqueeze_71, mul_211, o1_35, mul_212, mul_213, o2_35, output_86], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf319, arg8_1, arg6_1, buf320, buf322, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf321 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_17, value_17, kv_cache_dummy_dep_17], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf323 = torch.ops.vllm.unified_kv_cache_update.default(buf322, reinterpret_tensor(buf319, (s72, 16, 128), (6144, 128, 1), 4096), arg145_1)
            buf324 = buf323
            assert_alignment(buf324, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf323
            # Topologically Sorted Source Nodes: [split_17, unified_attention_with_output_17, value_17], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf320, buf322, reinterpret_tensor(buf319, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf321, (s72, 16, 128), (2048, 128, 1), 0), arg145_1, None, None, buf324)
            del arg145_1
            del buf319
            del buf320
            del buf324
            assert_size_stride(arg146_1, (2048, 2048), (2048, 1), 'input')
            buf327 = reinterpret_tensor(buf322, (s72, 2048), (2048, 1), 0); del buf322  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_17, output_parallel_36], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf321, reinterpret_tensor(arg146_1, (2048, 2048), (1, 2048), 0), out=buf327)
            del arg146_1
            assert_size_stride(arg147_1, (2048, ), (1, ), 'input')
            buf328 = buf298; del buf298  # reuse
            buf330 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [x_131, x_residual_31, to_232, x_132, result_16, x_135, x_residual_32, to_237, x_136, x_139, x_residual_33, to_246, x_140, pow_36, variance_35, add_123, rsqrt_35, x_141, to_248, x_142], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf328, buf327, buf315, buf316, buf309, arg147_1, buf330, s72, 2048, stream=raw_stream0)
            del arg147_1
            del buf309
            del buf315
            buf332 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_89], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf330, buf331, out=buf332)
            # Topologically Sorted Source Nodes: [output_89, moe_forward_shared_17], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf333 = torch.ops.vllm.moe_forward_shared.default(buf330, reinterpret_tensor(buf332, (s72, 60), (64, 1), 0), buf330, None, arg149_1, 0)
            del arg149_1
            buf334 = buf333[0]
            assert_size_stride(buf334, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf334, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf335 = buf333[1]
            assert_size_stride(buf335, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf335, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf333
            assert_size_stride(arg150_1, (2048, ), (1, ), 'input')
            buf337 = buf330; del buf330  # reuse
            # Topologically Sorted Source Nodes: [result_17, x_143, x_residual_34, to_251, x_144, pow_37, variance_36, add_126, rsqrt_36, x_145, to_253, x_146], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf334, buf335, buf328, arg150_1, buf337, s72, 2048, stream=raw_stream0)
            del arg150_1
            assert_size_stride(arg151_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg152_1, (6144, 2048), (2048, 1), 'input')
            buf338 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_17, x_143, x_residual_34, to_251, x_144, pow_37, variance_36, add_126, rsqrt_36, x_145, to_253, x_146, output_parallel_37], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg151_1, buf337, reinterpret_tensor(arg152_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf338)
            del arg151_1
            del arg152_1
            buf339 = reinterpret_tensor(buf337, (s72, 16, 128), (2048, 128, 1), 0); del buf337  # reuse
            buf341 = reinterpret_tensor(buf327, (s72, 16, 128), (2048, 128, 1), 0); del buf327  # reuse
            # Topologically Sorted Source Nodes: [split_18, cos_sin_18, chunk_54, query_54, chunk_55, key_54, chunk_56, unsqueeze_72, mul_218, unsqueeze_73, mul_219, o1_36, mul_220, mul_221, o2_36, output_90, unsqueeze_74, mul_222, unsqueeze_75, mul_223, o1_37, mul_224, mul_225, o2_37, output_91], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf338, arg8_1, arg6_1, buf339, buf341, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf340 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [split_18, value_18, kv_cache_dummy_dep_18], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf342 = torch.ops.vllm.unified_kv_cache_update.default(buf341, reinterpret_tensor(buf338, (s72, 16, 128), (6144, 128, 1), 4096), arg153_1)
            buf343 = buf342
            assert_alignment(buf343, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf342
            # Topologically Sorted Source Nodes: [split_18, unified_attention_with_output_18, value_18], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf339, buf341, reinterpret_tensor(buf338, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf340, (s72, 16, 128), (2048, 128, 1), 0), arg153_1, None, None, buf343)
            del arg153_1
            del buf338
            del buf339
            del buf343
            assert_size_stride(arg154_1, (2048, 2048), (2048, 1), 'input')
            buf346 = reinterpret_tensor(buf341, (s72, 2048), (2048, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_18, output_parallel_38], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf340, reinterpret_tensor(arg154_1, (2048, 2048), (1, 2048), 0), out=buf346)
            del arg154_1
            assert_size_stride(arg155_1, (2048, ), (1, ), 'input')
            buf348 = buf340; del buf340  # reuse
            # Topologically Sorted Source Nodes: [result_17, x_143, x_residual_34, to_251, x_144, x_147, x_residual_35, to_260, x_148, pow_38, variance_37, add_130, rsqrt_37, x_149, to_262, x_150], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf346, buf334, buf335, buf328, arg155_1, buf348, s72, 2048, stream=raw_stream0)
            del arg155_1
            buf350 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [output_94], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf348, buf349, out=buf350)
            # Topologically Sorted Source Nodes: [output_94, moe_forward_shared_18], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf351 = torch.ops.vllm.moe_forward_shared.default(buf348, reinterpret_tensor(buf350, (s72, 60), (64, 1), 0), buf348, None, arg157_1, 0)
            del arg157_1
            buf352 = buf351[0]
            assert_size_stride(buf352, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf352, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf353 = buf351[1]
            assert_size_stride(buf353, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf353, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf351
            assert_size_stride(arg158_1, (2048, ), (1, ), 'input')
            buf354 = buf328; del buf328  # reuse
            buf356 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [result_17, x_143, x_residual_34, to_251, x_144, x_147, x_residual_35, to_260, x_148, result_18, x_151, x_residual_36, to_265, x_152, pow_39, variance_38, add_133, rsqrt_38, x_153, to_267, x_154], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf354, buf352, buf353, buf346, buf334, buf335, arg158_1, buf356, s72, 2048, stream=raw_stream0)
            del arg158_1
            del buf334
            del buf335
            del buf346
            assert_size_stride(arg159_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg160_1, (6144, 2048), (2048, 1), 'input')
            buf357 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_39, variance_38, add_133, rsqrt_38, x_153, to_267, x_154, output_parallel_39], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg159_1, buf356, reinterpret_tensor(arg160_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf357)
            del arg159_1
            del arg160_1
            buf358 = reinterpret_tensor(buf356, (s72, 16, 128), (2048, 128, 1), 0); del buf356  # reuse
            buf360 = reinterpret_tensor(buf353, (s72, 16, 128), (2048, 128, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [split_19, cos_sin_19, chunk_57, query_57, chunk_58, key_57, chunk_59, unsqueeze_76, mul_230, unsqueeze_77, mul_231, o1_38, mul_232, mul_233, o2_38, output_95, unsqueeze_78, mul_234, unsqueeze_79, mul_235, o1_39, mul_236, mul_237, o2_39, output_96], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf357, arg8_1, arg6_1, buf358, buf360, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf359 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [split_19, value_19, kv_cache_dummy_dep_19], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf361 = torch.ops.vllm.unified_kv_cache_update.default(buf360, reinterpret_tensor(buf357, (s72, 16, 128), (6144, 128, 1), 4096), arg161_1)
            buf362 = buf361
            assert_alignment(buf362, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf361
            # Topologically Sorted Source Nodes: [split_19, unified_attention_with_output_19, value_19], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf358, buf360, reinterpret_tensor(buf357, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf359, (s72, 16, 128), (2048, 128, 1), 0), arg161_1, None, None, buf362)
            del arg161_1
            del buf362
            assert_size_stride(arg162_1, (2048, 2048), (2048, 1), 'input')
            buf365 = reinterpret_tensor(buf360, (s72, 2048), (2048, 1), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_19, output_parallel_40], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf359, reinterpret_tensor(arg162_1, (2048, 2048), (1, 2048), 0), out=buf365)
            del arg162_1
            assert_size_stride(arg163_1, (2048, ), (1, ), 'input')
            buf367 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [x_155, x_residual_37, to_274, x_156, pow_40, variance_39, add_137, rsqrt_39, x_157, to_276, x_158], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf365, buf354, arg163_1, buf367, s72, 2048, stream=raw_stream0)
            del arg163_1
            assert_size_stride(arg164_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg172_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg180_1, (60, 2048), (2048, 1), 'input')
            assert_size_stride(arg188_1, (60, 2048), (2048, 1), 'input')
            buf368 = buf349; del buf349  # reuse
            buf387 = buf331; del buf331  # reuse
            buf405 = buf312; del buf312  # reuse
            buf424 = empty_strided_cuda((2048, 64), (1, 2048), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_99, output_104, output_109, output_114], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_10.run(arg164_1, arg172_1, arg180_1, arg188_1, buf368, buf387, buf405, buf424, stream=raw_stream0)
            del arg164_1
            del arg172_1
            del arg180_1
            del arg188_1
            buf369 = buf350; del buf350  # reuse
            # Topologically Sorted Source Nodes: [output_99], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf367, buf368, out=buf369)
            del buf368
            # Topologically Sorted Source Nodes: [output_99, moe_forward_shared_19], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf370 = torch.ops.vllm.moe_forward_shared.default(buf367, reinterpret_tensor(buf369, (s72, 60), (64, 1), 0), buf367, None, arg165_1, 0)
            del arg165_1
            del buf369
            buf371 = buf370[0]
            assert_size_stride(buf371, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf371, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf372 = buf370[1]
            assert_size_stride(buf372, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf372, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf370
            assert_size_stride(arg166_1, (2048, ), (1, ), 'input')
            buf374 = buf367; del buf367  # reuse
            # Topologically Sorted Source Nodes: [x_155, x_residual_37, to_274, x_156, result_19, x_159, x_residual_38, to_279, x_160, pow_41, variance_40, add_140, rsqrt_40, x_161, to_281, x_162], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf371, buf372, buf365, buf354, arg166_1, buf374, s72, 2048, stream=raw_stream0)
            del arg166_1
            assert_size_stride(arg167_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg168_1, (6144, 2048), (2048, 1), 'input')
            buf375 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [x_155, x_residual_37, to_274, x_156, result_19, x_159, x_residual_38, to_279, x_160, pow_41, variance_40, add_140, rsqrt_40, x_161, to_281, x_162, output_parallel_41], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg167_1, buf374, reinterpret_tensor(arg168_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf375)
            del arg167_1
            del arg168_1
            buf376 = reinterpret_tensor(buf374, (s72, 16, 128), (2048, 128, 1), 0); del buf374  # reuse
            buf378 = buf358; del buf358  # reuse
            # Topologically Sorted Source Nodes: [split_20, cos_sin_20, chunk_60, query_60, chunk_61, key_60, chunk_62, unsqueeze_80, mul_242, unsqueeze_81, mul_243, o1_40, mul_244, mul_245, o2_40, output_100, unsqueeze_82, mul_246, unsqueeze_83, mul_247, o1_41, mul_248, mul_249, o2_41, output_101], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf375, arg8_1, arg6_1, buf376, buf378, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf377 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_20, value_20, kv_cache_dummy_dep_20], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf379 = torch.ops.vllm.unified_kv_cache_update.default(buf378, reinterpret_tensor(buf375, (s72, 16, 128), (6144, 128, 1), 4096), arg169_1)
            buf380 = buf379
            assert_alignment(buf380, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf379
            # Topologically Sorted Source Nodes: [split_20, unified_attention_with_output_20, value_20], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf376, buf378, reinterpret_tensor(buf375, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf377, (s72, 16, 128), (2048, 128, 1), 0), arg169_1, None, None, buf380)
            del arg169_1
            del buf375
            del buf376
            del buf380
            assert_size_stride(arg170_1, (2048, 2048), (2048, 1), 'input')
            buf383 = reinterpret_tensor(buf378, (s72, 2048), (2048, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_20, output_parallel_42], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf377, reinterpret_tensor(arg170_1, (2048, 2048), (1, 2048), 0), out=buf383)
            del arg170_1
            assert_size_stride(arg171_1, (2048, ), (1, ), 'input')
            buf384 = buf354; del buf354  # reuse
            buf386 = buf377; del buf377  # reuse
            # Topologically Sorted Source Nodes: [x_155, x_residual_37, to_274, x_156, result_19, x_159, x_residual_38, to_279, x_160, x_163, x_residual_39, to_288, x_164, pow_42, variance_41, add_144, rsqrt_41, x_165, to_290, x_166], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf384, buf383, buf371, buf372, buf365, arg171_1, buf386, s72, 2048, stream=raw_stream0)
            del arg171_1
            del buf365
            del buf371
            buf388 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_104], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf386, buf387, out=buf388)
            del buf387
            # Topologically Sorted Source Nodes: [output_104, moe_forward_shared_20], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf389 = torch.ops.vllm.moe_forward_shared.default(buf386, reinterpret_tensor(buf388, (s72, 60), (64, 1), 0), buf386, None, arg173_1, 0)
            del arg173_1
            buf390 = buf389[0]
            assert_size_stride(buf390, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf390, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf391 = buf389[1]
            assert_size_stride(buf391, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf391, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf389
            assert_size_stride(arg174_1, (2048, ), (1, ), 'input')
            buf393 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [result_20, x_167, x_residual_40, to_293, x_168, pow_43, variance_42, add_147, rsqrt_42, x_169, to_295, x_170], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf390, buf391, buf384, arg174_1, buf393, s72, 2048, stream=raw_stream0)
            del arg174_1
            assert_size_stride(arg175_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg176_1, (6144, 2048), (2048, 1), 'input')
            buf394 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_20, x_167, x_residual_40, to_293, x_168, pow_43, variance_42, add_147, rsqrt_42, x_169, to_295, x_170, output_parallel_43], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg175_1, buf393, reinterpret_tensor(arg176_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf394)
            del arg175_1
            del arg176_1
            buf395 = reinterpret_tensor(buf393, (s72, 16, 128), (2048, 128, 1), 0); del buf393  # reuse
            buf397 = reinterpret_tensor(buf383, (s72, 16, 128), (2048, 128, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [split_21, cos_sin_21, chunk_63, query_63, chunk_64, key_63, chunk_65, unsqueeze_84, mul_254, unsqueeze_85, mul_255, o1_42, mul_256, mul_257, o2_42, output_105, unsqueeze_86, mul_258, unsqueeze_87, mul_259, o1_43, mul_260, mul_261, o2_43, output_106], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf394, arg8_1, arg6_1, buf395, buf397, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf396 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [split_21, value_21, kv_cache_dummy_dep_21], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf398 = torch.ops.vllm.unified_kv_cache_update.default(buf397, reinterpret_tensor(buf394, (s72, 16, 128), (6144, 128, 1), 4096), arg177_1)
            buf399 = buf398
            assert_alignment(buf399, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf398
            # Topologically Sorted Source Nodes: [split_21, unified_attention_with_output_21, value_21], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf395, buf397, reinterpret_tensor(buf394, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf396, (s72, 16, 128), (2048, 128, 1), 0), arg177_1, None, None, buf399)
            del arg177_1
            del buf394
            del buf395
            del buf399
            assert_size_stride(arg178_1, (2048, 2048), (2048, 1), 'input')
            buf402 = reinterpret_tensor(buf397, (s72, 2048), (2048, 1), 0); del buf397  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_21, output_parallel_44], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf396, reinterpret_tensor(arg178_1, (2048, 2048), (1, 2048), 0), out=buf402)
            del arg178_1
            assert_size_stride(arg179_1, (2048, ), (1, ), 'input')
            buf404 = buf396; del buf396  # reuse
            # Topologically Sorted Source Nodes: [result_20, x_167, x_residual_40, to_293, x_168, x_171, x_residual_41, to_302, x_172, pow_44, variance_43, add_151, rsqrt_43, x_173, to_304, x_174], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf402, buf390, buf391, buf384, arg179_1, buf404, s72, 2048, stream=raw_stream0)
            del arg179_1
            buf406 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [output_109], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf404, buf405, out=buf406)
            del buf405
            # Topologically Sorted Source Nodes: [output_109, moe_forward_shared_21], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf407 = torch.ops.vllm.moe_forward_shared.default(buf404, reinterpret_tensor(buf406, (s72, 60), (64, 1), 0), buf404, None, arg181_1, 0)
            del arg181_1
            buf408 = buf407[0]
            assert_size_stride(buf408, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf408, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf409 = buf407[1]
            assert_size_stride(buf409, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf409, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf407
            assert_size_stride(arg182_1, (2048, ), (1, ), 'input')
            buf410 = buf384; del buf384  # reuse
            buf412 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [result_20, x_167, x_residual_40, to_293, x_168, x_171, x_residual_41, to_302, x_172, result_21, x_175, x_residual_42, to_307, x_176, pow_45, variance_44, add_154, rsqrt_44, x_177, to_309, x_178], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf410, buf408, buf409, buf402, buf390, buf391, arg182_1, buf412, s72, 2048, stream=raw_stream0)
            del arg182_1
            del buf390
            del buf391
            del buf402
            assert_size_stride(arg183_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg184_1, (6144, 2048), (2048, 1), 'input')
            buf413 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pow_45, variance_44, add_154, rsqrt_44, x_177, to_309, x_178, output_parallel_45], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(arg183_1, buf412, reinterpret_tensor(arg184_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf413)
            del arg183_1
            del arg184_1
            buf414 = reinterpret_tensor(buf412, (s72, 16, 128), (2048, 128, 1), 0); del buf412  # reuse
            buf416 = reinterpret_tensor(buf409, (s72, 16, 128), (2048, 128, 1), 0); del buf409  # reuse
            # Topologically Sorted Source Nodes: [split_22, cos_sin_22, chunk_66, query_66, chunk_67, key_66, chunk_68, unsqueeze_88, mul_266, unsqueeze_89, mul_267, o1_44, mul_268, mul_269, o2_44, output_110, unsqueeze_90, mul_270, unsqueeze_91, mul_271, o1_45, mul_272, mul_273, o2_45, output_111], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf413, arg8_1, arg6_1, buf414, buf416, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            buf415 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [split_22, value_22, kv_cache_dummy_dep_22], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf417 = torch.ops.vllm.unified_kv_cache_update.default(buf416, reinterpret_tensor(buf413, (s72, 16, 128), (6144, 128, 1), 4096), arg185_1)
            buf418 = buf417
            assert_alignment(buf418, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf417
            # Topologically Sorted Source Nodes: [split_22, unified_attention_with_output_22, value_22], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf414, buf416, reinterpret_tensor(buf413, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf415, (s72, 16, 128), (2048, 128, 1), 0), arg185_1, None, None, buf418)
            del arg185_1
            del buf418
            assert_size_stride(arg186_1, (2048, 2048), (2048, 1), 'input')
            buf421 = reinterpret_tensor(buf416, (s72, 2048), (2048, 1), 0); del buf416  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_22, output_parallel_46], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf415, reinterpret_tensor(arg186_1, (2048, 2048), (1, 2048), 0), out=buf421)
            del arg186_1
            assert_size_stride(arg187_1, (2048, ), (1, ), 'input')
            buf423 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [x_179, x_residual_43, to_316, x_180, pow_46, variance_45, add_158, rsqrt_45, x_181, to_318, x_182], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf421, buf410, arg187_1, buf423, s72, 2048, stream=raw_stream0)
            del arg187_1
            buf425 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [output_114], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf423, buf424, out=buf425)
            # Topologically Sorted Source Nodes: [output_114, moe_forward_shared_22], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf426 = torch.ops.vllm.moe_forward_shared.default(buf423, reinterpret_tensor(buf425, (s72, 60), (64, 1), 0), buf423, None, arg189_1, 0)
            del arg189_1
            del buf425
            buf427 = buf426[0]
            assert_size_stride(buf427, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf427, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf428 = buf426[1]
            assert_size_stride(buf428, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf428, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf426
            assert_size_stride(arg190_1, (2048, ), (1, ), 'input')
            buf430 = buf423; del buf423  # reuse
            # Topologically Sorted Source Nodes: [x_179, x_residual_43, to_316, x_180, result_22, x_183, x_residual_44, to_321, x_184, pow_47, variance_46, add_161, rsqrt_46, x_185, to_323, x_186], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf427, buf428, buf421, buf410, arg190_1, buf430, s72, 2048, stream=raw_stream0)
            del arg190_1
            assert_size_stride(arg191_1, (6144, ), (1, ), 'input')
            assert_size_stride(arg192_1, (6144, 2048), (2048, 1), 'input')
            buf431 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [x_179, x_residual_43, to_316, x_180, result_22, x_183, x_residual_44, to_321, x_184, pow_47, variance_46, add_161, rsqrt_46, x_185, to_323, x_186, output_parallel_47], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg191_1, buf430, reinterpret_tensor(arg192_1, (2048, 6144), (1, 2048), 0), alpha=1, beta=1, out=buf431)
            del arg191_1
            del arg192_1
            buf432 = reinterpret_tensor(buf430, (s72, 16, 128), (2048, 128, 1), 0); del buf430  # reuse
            buf434 = buf414; del buf414  # reuse
            # Topologically Sorted Source Nodes: [split_23, cos_sin_23, chunk_69, query_69, chunk_70, key_69, chunk_71, unsqueeze_92, mul_278, unsqueeze_93, mul_279, o1_46, mul_280, mul_281, o2_46, output_115, unsqueeze_94, mul_282, unsqueeze_95, mul_283, o1_47, mul_284, mul_285, o2_47, output_116], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 2048*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf431, arg8_1, arg6_1, buf432, buf434, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            del arg6_1
            del arg8_1
            buf433 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_23, value_23, kv_cache_dummy_dep_23], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
            buf435 = torch.ops.vllm.unified_kv_cache_update.default(buf434, reinterpret_tensor(buf431, (s72, 16, 128), (6144, 128, 1), 4096), arg193_1)
            buf436 = buf435
            assert_alignment(buf436, 16, 'torch.ops.vllm.unified_kv_cache_update.default')
            del buf435
            # Topologically Sorted Source Nodes: [split_23, unified_attention_with_output_23, value_23], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_attention_with_output]
            torch.ops.vllm.unified_attention_with_output.default(buf432, buf434, reinterpret_tensor(buf431, (s72, 16, 128), (6144, 128, 1), 4096), reinterpret_tensor(buf433, (s72, 16, 128), (2048, 128, 1), 0), arg193_1, None, None, buf436)
            del arg193_1
            del buf431
            del buf432
            del buf436
            assert_size_stride(arg194_1, (2048, 2048), (2048, 1), 'input')
            buf439 = reinterpret_tensor(buf434, (s72, 2048), (2048, 1), 0); del buf434  # reuse
            # Topologically Sorted Source Nodes: [unified_attention_with_output_23, output_parallel_48], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf433, reinterpret_tensor(arg194_1, (2048, 2048), (1, 2048), 0), out=buf439)
            del arg194_1
            assert_size_stride(arg195_1, (2048, ), (1, ), 'input')
            buf440 = buf410; del buf410  # reuse
            buf442 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [x_179, x_residual_43, to_316, x_180, result_22, x_183, x_residual_44, to_321, x_184, x_187, x_residual_45, to_330, x_188, pow_48, variance_47, add_165, rsqrt_47, x_189, to_332, x_190], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf440, buf439, buf427, buf428, buf421, arg195_1, buf442, s72, 2048, stream=raw_stream0)
            del arg195_1
            del buf421
            del buf427
            del buf428
            del buf439
            assert_size_stride(arg196_1, (60, 2048), (2048, 1), 'input')
            buf443 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [output_119], Original ATen: [aten.t, aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mm_t_12.run(arg196_1, buf443, 131072, stream=raw_stream0)
            del arg196_1
            buf444 = empty_strided_cuda((s72, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output_119], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf442, buf443, out=buf444)
            del buf443
            # Topologically Sorted Source Nodes: [output_119, moe_forward_shared_23], Original ATen: [aten.mm, vllm.moe_forward_shared]
            buf445 = torch.ops.vllm.moe_forward_shared.default(buf442, reinterpret_tensor(buf444, (s72, 60), (64, 1), 0), buf442, None, arg197_1, 0)
            del arg197_1
            del buf442
            del buf444
            buf446 = buf445[0]
            assert_size_stride(buf446, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf446, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf447 = buf445[1]
            assert_size_stride(buf447, (s72, 2048), (2048, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf447, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf445
            assert_size_stride(arg198_1, (2048, ), (1, ), 'input')
            buf449 = buf446; del buf446  # reuse
            # Topologically Sorted Source Nodes: [result_23, x_191, x_residual_46, to_335, x_192, pow_49, variance_48, add_168, rsqrt_48, x_193, to_337, x_194], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf449, buf447, buf440, arg198_1, s72, 2048, stream=raw_stream0)
            del arg198_1
            del buf440
            del buf447
        return (buf449, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 16384
    arg1_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg2_1 = rand_strided((151936, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = 16384
    arg8_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg9_1 = None
    arg10_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = None
    arg14_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = None
    arg18_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = None
    arg22_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = None
    arg26_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = None
    arg30_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = None
    arg34_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = None
    arg38_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = None
    arg42_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = None
    arg46_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = None
    arg50_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = None
    arg54_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = None
    arg58_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = None
    arg62_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = None
    arg66_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = None
    arg70_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = None
    arg74_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = None
    arg78_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = None
    arg82_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = None
    arg86_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = None
    arg90_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = None
    arg94_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = None
    arg98_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = None
    arg102_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = None
    arg106_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = None
    arg110_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = None
    arg114_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = None
    arg118_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = None
    arg122_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = None
    arg126_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = None
    arg130_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = None
    arg134_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = None
    arg138_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = None
    arg142_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = None
    arg146_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = None
    arg150_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = None
    arg154_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = None
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = None
    arg162_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = None
    arg166_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = None
    arg170_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = None
    arg174_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = None
    arg178_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = None
    arg182_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = None
    arg186_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = None
    arg190_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = None
    arg194_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((60, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = None
    arg198_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
