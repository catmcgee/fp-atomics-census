"""Scanner tests on the synthetic CUDA fixtures.

Each fixture exercises one taxonomy class or one false-positive pattern. The
assertions are about what the scanner reports (kind, dtype hint, flags), not
about the final class, which is a triage decision.
"""
from pathlib import Path

import pytest

from scan.scanner import scan_cxx_source, scan_wrapper_calls, wrapper_definitions

FIX = Path(__file__).parent / "fixtures"


def scan(name: str):
    p = FIX / name
    return scan_cxx_source(p.read_bytes(), name, "fixture", "0" * 40)


def live(cands):
    return [c for c in cands if not c.excluded_reason]


def by_kind(cands, kind):
    return [c for c in cands if c.kind == kind and not c.excluded_reason]


def test_float_atomic_add_is_found_with_dtype():
    c = live(scan("class_a_float_atomic.cu"))
    adds = [x for x in c if x.kind == "atomicAdd"]
    assert {x.dtype_hint for x in adds} == {"float32", "half2", "bfloat16"}
    assert all(x.class_hint == "A?" for x in adds)
    assert {x.function for x in adds} == {"split_k_accumulate", "packed_half_accumulate", "bf16_accumulate"}
    sub = by_kind(c, "atomicSub")
    assert len(sub) == 1 and sub[0].dtype_hint == "float32"  # cast to float* wins over the double declaration
    assert all(x.memory_space_hint == "global" for x in adds)


def test_cas_loop_is_flagged_as_float_add():
    c = live(scan("class_a_cas_loop.cu"))
    cas = by_kind(c, "atomicCAS")
    assert len(cas) == 1
    assert cas[0].class_hint == "A?"
    assert cas[0].function == "atomic_add_float_cas"
    assert any("float add" in n for n in cas[0].notes)


def test_ptx_red_and_atom_and_multimem():
    c = live(scan("class_a_ptx_red.cu"))
    red = by_kind(c, "ptx-red")
    assert {x.dtype_hint for x in red} == {"float32", "bfloat162"}
    assert all(x.class_hint == "A?" and x.memory_space_hint == "global" for x in red)
    atom = by_kind(c, "ptx-atom")
    assert len(atom) == 1 and atom[0].dtype_hint == "float32" and atom[0].class_hint == "A?"
    mm = by_kind(c, "ptx-multimem-red")
    assert len(mm) == 1 and mm[0].dtype_hint == "float32" and mm[0].class_hint == "A?"
    # PTX inside asm strings must not be reported as "in string"
    assert all(not x.in_string for x in red + atom + mm)


def test_tma_reduce_add():
    c = live(scan("class_a_tma_reduce.hpp"))
    tma = by_kind(c, "tma-reduce")
    kinds = {x.pattern for x in tma}
    assert "ptx_tma_reduce" in kinds and "cute_tma_reduce" in kinds
    ptx = [x for x in tma if x.pattern == "ptx_tma_reduce"][0]
    assert ptx.dtype_hint == "float32" and ptx.class_hint == "A?"
    assert all(x.class_hint == "A?" for x in tma)


def test_shared_memory_atomic():
    c = live(scan("class_a_shared.cu"))
    adds = by_kind(c, "atomicAdd")
    assert len(adds) == 1
    assert adds[0].memory_space_hint == "shared" and adds[0].dtype_hint == "float32"


def test_a1_still_reported():
    c = live(scan("class_a1_unique_index.cu"))
    assert len(by_kind(c, "atomicAdd")) == 1  # the scanner does not decide A1


def test_a2_semaphore_reports_only_integer_lock_ops():
    c = live(scan("class_a2_semaphore.cu"))
    assert by_kind(c, "atomicAdd") == []
    red = by_kind(c, "ptx-red")
    assert len(red) == 1 and red[0].dtype_hint == "int32" and red[0].class_hint == "B"


def test_a3_gate_conditions_are_captured():
    c = live(scan("class_a3_gated.cu"))
    adds = by_kind(c, "atomicAdd")
    float_adds = [x for x in adds if x.dtype_hint == "float32"]
    assert len(float_adds) == 2
    conds = [x.conditions for x in float_adds]
    assert any(any("Deterministic" in s and s.startswith("else of") for s in cs) for cs in conds)
    assert any(any(s.startswith("after return-guard") and "use_atomics" in s for s in cs) for cs in conds)
    assert any("Deterministic" in x.template_params for x in float_adds)
    int_adds = [x for x in adds if x.dtype_hint == "int32"]
    assert len(int_adds) == 2 and all(x.class_hint == "B" for x in int_adds)


def test_integer_counters_are_class_b():
    c = live(scan("class_b_int_counter.cu"))
    assert all(x.class_hint == "B" for x in c)
    kinds = {x.kind for x in c}
    assert kinds == {"atomicAdd", "atomicOr", "atomicInc"}
    assert {x.dtype_hint for x in by_kind(c, "atomicAdd")} == {"int32", "uint64"}


def test_float_max_min_are_class_b():
    c = live(scan("class_b_float_max.cu"))
    mm = [x for x in c if x.kind in ("atomicMax", "atomicMin")]
    assert len(mm) == 2 and all(x.class_hint == "B" for x in mm)
    assert all(x.dtype_hint == "float32" for x in mm)
    # the wrapper call site is also reported and linked
    defs = wrapper_definitions(_with_ids(c))
    assert "atomicMaxFloat" in defs
    calls = scan_wrapper_calls((FIX / "class_b_float_max.cu").read_bytes(), "class_b_float_max.cu", "fixture", "0" * 40, defs)
    assert len(calls) == 1 and calls[0].function == "absmax" and calls[0].kind == "wrapper-call"


def test_lock_exch_cas_are_class_b():
    c = live(scan("class_b_lock_exch.cu"))
    assert {x.kind for x in c} == {"atomicCAS", "atomicExch"}
    assert all(x.class_hint == "B" for x in c)


def test_comments_strings_and_dead_blocks_are_flagged():
    c = scan("fp_comment_and_string.cu")
    assert c, "matches in comments and strings are kept as candidates, flagged"
    assert all(x.excluded_reason in ("comment", "string", "dead-block") for x in c)
    assert live(c) == []
    reasons = {x.excluded_reason for x in c}
    assert reasons == {"comment", "string", "dead-block"}


def test_macro_wrapper_sites():
    c = live(scan("fp_macro_wrapper.cu"))
    adds = by_kind(c, "atomicAdd")
    # only the #define body contains the primitive; expansions are found as wrapper calls
    assert len(adds) == 1 and adds[0].function == "ATOMIC_ADD (macro)"
    defs = wrapper_definitions(_with_ids(c))
    assert "ATOMIC_ADD" in defs
    calls = scan_wrapper_calls((FIX / "fp_macro_wrapper.cu").read_bytes(), "fp_macro_wrapper.cu", "fixture", "0" * 40, defs)
    assert [x.function for x in calls] == ["via_macro", "via_macro"]
    assert [x.dtype_hint for x in calls] == ["float32", "int32"]


def test_host_atomic_excluded():
    c = scan("fp_host_atomic.cpp")
    assert c and all(x.excluded_reason == "host" for x in c)


def test_name_collisions_not_reported():
    c = live(scan("fp_name_collision.cu"))
    assert by_kind(c, "atomicAdd") == []
    assert c == []


def _with_ids(cands):
    for i, c in enumerate(cands, 1):
        c.id = f"fixture-c{i:05d}"
    return cands
