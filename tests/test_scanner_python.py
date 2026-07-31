"""Python scanner tests on the Triton and torch fixtures."""
from pathlib import Path

from scan.py import scan_python_source

FIX = Path(__file__).parent / "fixtures"


def scan(name):
    return scan_python_source((FIX / name).read_text(), name, "fixture", "0" * 40)


def test_triton_float_atomic_add_and_float_max():
    c = scan("triton_atomic_float.py")
    adds = [x for x in c if x.kind == "tl.atomic_add"]
    assert len(adds) == 1
    a = adds[0]
    assert a.function == "splitk_kernel" and "triton.jit" in a.qualifiers
    assert a.class_hint == "A?" and a.dtype_hint == "float32"  # acc is tl.zeros(..., dtype=tl.float32)? no: inferred from value text
    assert any("sem=relaxed" in n for n in a.notes)
    mx = [x for x in c if x.kind == "tl.atomic_max"]
    assert len(mx) == 1 and mx[0].class_hint == "B" and mx[0].function == "scale_kernel"


def test_triton_integer_atomics():
    c = scan("triton_atomic_int.py")
    adds = [x for x in c if x.kind == "tl.atomic_add"]
    assert len(adds) == 2 and all(x.class_hint == "B" and x.dtype_hint == "int32" for x in adds)
    assert {x.function for x in adds} == {"count_kernel", "slot_kernel"}
    kinds = {x.kind for x in c}
    assert {"tl.atomic_cas", "tl.atomic_xchg"} <= kinds
    cas = [x for x in c if x.kind == "tl.atomic_cas"][0]
    assert cas.conditions and cas.conditions[0].startswith("while ")


def test_torch_ops():
    c = scan("torch_ops.py")
    by = {}
    for x in c:
        by.setdefault(x.match, []).append(x)
    assert by["out.index_add_("][0].class_hint == "A?"
    assert by["out.scatter_add_("][0].class_hint == "A?"
    ip = by["out.index_put_("][0]
    assert ip.class_hint == "A?" and "accumulate=True" in ip.notes
    bc = by["torch.bincount("]
    assert {x.class_hint for x in bc} == {"A?", "B"}
    assert by["dst.scatter_("][0].class_hint == "?"
    assert by["torch.cumsum("][0].class_hint == "?"
    assert by["torch.use_deterministic_algorithms("][0].kind == "torch-op"
    # comments and strings never match
    assert all(x.line not in (33, 34) for x in c)
    assert all(x.function != "not_sites" or x.match == "torch.use_deterministic_algorithms(" for x in c)
