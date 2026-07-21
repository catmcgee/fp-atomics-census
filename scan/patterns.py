"""Pattern tables for the candidate scanner.

Each entry is (name, compiled regex, kind, class_hint). ``class_hint`` is a
coarse pre-classification for the triage reader and is never copied into the
inventory without being checked:

* ``A?``  floating-point accumulation if the operand is a float type
* ``B``   exact by construction (max, min, lock, integer-only op)
* ``?``   cannot be told from the pattern alone
* ``C``   call into an opaque library
"""
from __future__ import annotations

import re

# PTX suffix vocabulary. Order matters: longer suffixes first.
_PTX_TYPES = r"(f16x2|bf16x2|f32|f64|f16|bf16|s32|u32|s64|u64|b32|b64|u16|s16|b16)"
_PTX_FLOAT_TYPES = {"f32", "f64", "f16", "f16x2", "bf16", "bf16x2"}

CUDA_PATTERNS: list[tuple[str, re.Pattern[str], str, str]] = [
    ("atomicAdd", re.compile(r"(?<![A-Za-z0-9_])atomicAdd(?:_block|_system)?\s*\("), "atomicAdd", "A?"),
    ("atomicSub", re.compile(r"(?<![A-Za-z0-9_])atomicSub(?:_block|_system)?\s*\("), "atomicSub", "A?"),
    ("atomicCAS", re.compile(r"(?<![A-Za-z0-9_])atomicCAS(?:_block|_system)?\s*\("), "atomicCAS", "?"),
    ("atomicExch", re.compile(r"(?<![A-Za-z0-9_])atomicExch(?:_block|_system)?\s*\("), "atomicExch", "B"),
    ("atomicMax", re.compile(r"(?<![A-Za-z0-9_])atomicMax(?:_block|_system)?\s*\("), "atomicMax", "B"),
    ("atomicMin", re.compile(r"(?<![A-Za-z0-9_])atomicMin(?:_block|_system)?\s*\("), "atomicMin", "B"),
    ("atomicInc", re.compile(r"(?<![A-Za-z0-9_])atomicInc(?:_block|_system)?\s*\("), "atomicInc", "B"),
    ("atomicDec", re.compile(r"(?<![A-Za-z0-9_])atomicDec(?:_block|_system)?\s*\("), "atomicDec", "B"),
    ("atomicOr", re.compile(r"(?<![A-Za-z0-9_])atomicOr(?:_block|_system)?\s*\("), "atomicOr", "B"),
    ("atomicAnd", re.compile(r"(?<![A-Za-z0-9_])atomicAnd(?:_block|_system)?\s*\("), "atomicAnd", "B"),
    ("atomicXor", re.compile(r"(?<![A-Za-z0-9_])atomicXor(?:_block|_system)?\s*\("), "atomicXor", "B"),
    # libcu++ atomics and atomic_ref; the dtype is in the template argument
    ("cuda_atomic", re.compile(r"\bcuda::(?:std::)?atomic(?:_ref)?\s*<"), "atomic_ref", "?"),
    ("fetch_add", re.compile(r"\.fetch_(?:add|sub)\s*\("), "atomic_ref", "?"),
    # inline PTX reductions. The suffix decides the dtype.
    ("ptx_red", re.compile(
        r"(?<![A-Za-z0-9_.])red(?:\.(?:relaxed|release|acquire|acq_rel|weak|async|gpu|cta|sys|cluster|global|shared(?:::cta|::cluster)?|mbarrier(?:::complete_tx)?(?:::bytes)?))*"
        r"\.(add|inc|dec|min|max|and|or|xor)(?:\.noftz)?(?:\.v[248])?\." + _PTX_TYPES), "ptx-red", "?"),
    ("ptx_atom", re.compile(
        r"(?<![A-Za-z0-9_.])atom(?:\.(?:relaxed|release|acquire|acq_rel|weak|gpu|cta|sys|cluster|global|shared(?:::cta|::cluster)?))*"
        r"\.(add|inc|dec|min|max|and|or|xor|cas|exch)(?:\.noftz)?(?:\.v[248])?\." + _PTX_TYPES), "ptx-atom", "?"),
    ("ptx_multimem_red", re.compile(r"\bmultimem\.red\.[a-z_.:0-9]*"), "ptx-multimem-red", "?"),
    ("ptx_multimem_ld_reduce", re.compile(r"\bmultimem\.ld_reduce\.[a-z_.:0-9]*"), "ptx-multimem-ld_reduce", "C"),
    # TMA / bulk reductions into global memory
    ("ptx_tma_reduce", re.compile(r"\bcp\.reduce\.async\.bulk(?:\.tensor)?[a-z_.:0-9]*"), "tma-reduce", "?"),
    ("cute_tma_reduce", re.compile(
        r"\b(?:SM90_TMA_REDUCE_(?:ADD|MIN|MAX|INC|DEC|AND|OR|XOR)(?:_[1-5]D)?|SM90_BULK_REDUCE_ADD|SM100_TMA_[A-Z0-9_]*REDUCE[A-Z0-9_]*|TMA_REDUCE_ADD|BULK_REDUCE_ADD)\b"),
        "tma-reduce", "?"),
    # wrappers named like atomics: atomicMaxFloat(...), atomic_add_bf16x2(...), red_add_f32(...)
    ("atomic_wrapper", re.compile(
        r"(?<![A-Za-z0-9_])(?:atomic(?:Add|Sub|Max|Min|Cas|CAS|Exch)[A-Za-z0-9_]+|atomic_[a-z0-9_]+|atom_[a-z0-9_]+|red_(?:add|relaxed|release)[a-z0-9_]*)\s*\("),
        "other", "?"),
]

# Names that the wrapper pattern must not report because a primitive pattern
# already covers them or because they are not atomics.
WRAPPER_EXCLUDE = {
    "atomic_thread_fence", "atomic_ref", "atomic_load", "atomic_store", "atomic_fence",
}

# Opaque library entry points (class C). Reported with kind library-call.
OPAQUE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("cublasLt", re.compile(r"\bcublasLt[A-Za-z0-9_]*\s*\("), "cuBLASLt"),
    ("cublas", re.compile(r"\bcublas(?!Lt)[A-Za-z0-9_]*\s*\("), "cuBLAS"),
    ("cudnn", re.compile(r"\bcudnn[A-Za-z0-9_]*\s*\("), "cuDNN"),
    ("nccl", re.compile(r"\bnccl(?:AllReduce|ReduceScatter|Reduce|AllGather|AllToAll|Send|Recv|Broadcast)\s*\("), "NCCL"),
    ("nvshmem", re.compile(r"\bnvshmem(?:x)?_[A-Za-z0-9_]*(?:atomic|signal|reduce|fetch)[A-Za-z0-9_]*\s*\("), "NVSHMEM"),
    ("cubin", re.compile(r"\b(?:cuModuleLoad(?:Data|FatBinary)?|cuLibraryLoad(?:Data|FromFile)|cudaLibraryLoad(?:Data|FromFile)|cuLaunchKernel(?:Ex)?)\s*\("), "cubin"),
]

# C++ type spellings to inventory dtype names.
CXX_TYPE_TO_DTYPE: dict[str, str] = {
    "float": "float32",
    "double": "float64",
    "half": "float16", "__half": "float16", "fp16_t": "float16", "half_t": "float16", "cutlass::half_t": "float16",
    "nv_half": "float16", "__nv_half": "float16",
    "__nv_bfloat16": "bfloat16", "nv_bfloat16": "bfloat16", "bf16_t": "bfloat16", "bfloat16_t": "bfloat16",
    "cutlass::bfloat16_t": "bfloat16", "__hip_bfloat16": "bfloat16",
    "__half2": "half2", "half2": "half2", "nv_half2": "half2",
    "__nv_bfloat162": "bfloat162", "nv_bfloat162": "bfloat162", "__hip_bfloat162": "bfloat162",
    "float2": "float2", "float4": "float4",
    "int": "int32", "int32_t": "int32", "signed int": "int32", "int32": "int32",
    "unsigned": "uint32", "unsigned int": "uint32", "uint32_t": "uint32", "uint": "uint32", "uint32": "uint32",
    "long": "int64", "long long": "int64", "int64_t": "int64", "int64": "int64", "long int": "int64",
    "unsigned long long": "uint64", "unsigned long": "uint64", "uint64_t": "uint64", "size_t": "uint64", "uint64": "uint64",
    "bool": "bool",
    "int8_t": "int8", "uint8_t": "int8", "char": "int8", "unsigned char": "int8",
    "int16_t": "int16", "uint16_t": "int16", "short": "int16", "unsigned short": "int16",
}

FLOAT_DTYPES = {"float32", "float64", "float16", "bfloat16", "half2", "bfloat162", "float2", "float4", "fp8"}
INT_DTYPES = {"int8", "int16", "int32", "uint32", "int64", "uint64", "bool"}

# Template-ish type names that mean "depends on instantiation".
TEMPLATE_TYPE_NAMES = {"T", "scalar_t", "DType", "DTypeOut", "DTypeO", "DTypeIn", "Element", "ElementC", "ElementD",
                       "ElementAccumulator", "AccumT", "acc_t", "OutT", "out_t", "Dtype", "dtype", "TypeName",
                       "ElementOutput", "value_type", "ValueType", "TOut", "TIn", "TAcc", "T_OUT", "T_ACC", "OutType", "InType",
                       "AccType", "OutDtype", "AccumType", "ElementAccum", "Tacc", "TC", "TD"}


def ptx_dtype(suffix: str) -> str:
    return {
        "f32": "float32", "f64": "float64", "f16": "float16", "f16x2": "half2",
        "bf16": "bfloat16", "bf16x2": "bfloat162",
        "s32": "int32", "u32": "uint32", "s64": "int64", "u64": "uint64",
        "b32": "uint32", "b64": "uint64", "u16": "int16", "s16": "int16", "b16": "int16",
    }.get(suffix, "unknown")


def ptx_is_float(suffix: str) -> bool:
    return suffix in _PTX_FLOAT_TYPES
