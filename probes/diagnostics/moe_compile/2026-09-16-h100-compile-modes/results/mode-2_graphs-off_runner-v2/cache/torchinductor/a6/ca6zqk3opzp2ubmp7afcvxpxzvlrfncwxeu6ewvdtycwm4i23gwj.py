
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
