"""Python scanner for Triton atomics and PyTorch operators that use atomics.

Uses the ``ast`` module rather than regexes so that comments and strings
never match, the enclosing function is known, and keyword arguments such as
``accumulate=True`` or ``sem="relaxed"`` are read structurally.

Reported patterns:

* ``tl.atomic_add`` / ``tl.atomic_max`` / ``tl.atomic_min`` / ``tl.atomic_cas``
  / ``tl.atomic_xchg`` / ``tl.atomic_and`` / ``tl.atomic_or`` / ``tl.atomic_xor``
  inside any function (a ``@triton.jit`` decorator is noted when present);
* PyTorch operators listed in docs/TAXONOMY.md: ``index_add_``,
  ``index_add``, ``scatter_add_``, ``scatter_add``, ``scatter_reduce_``,
  ``scatter_reduce``, ``index_put_`` and ``index_put`` with
  ``accumulate=True``, ``put_`` with ``accumulate=True``, ``bincount``,
  ``histc``, ``cumsum``, ``scatter_`` and ``scatter`` with a tensor source,
  ``index_copy_``, ``embedding_bag``, and ``use_deterministic_algorithms``
  (recorded so that the triage can see where an engine opts in);
* opaque library calls: ``torch._scaled_mm``, ``torch.distributed`` collectives,
  ``cublas``/``cudnn``-named bindings, and FlashInfer cubin loaders.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .scanner import Candidate

TRITON_ATOMICS = {
    "atomic_add": ("tl.atomic_add", "A?"),
    "atomic_max": ("tl.atomic_max", "B"),
    "atomic_min": ("tl.atomic_min", "B"),
    "atomic_cas": ("tl.atomic_cas", "B"),
    "atomic_xchg": ("tl.atomic_xchg", "B"),
    "atomic_and": ("tl.atomic_and", "B"),
    "atomic_or": ("tl.atomic_or", "B"),
    "atomic_xor": ("tl.atomic_xor", "B"),
}

# method or function name -> (class hint, note)
TORCH_OPS = {
    "index_add_": ("A?", "float index_add_ on CUDA accumulates with atomics (PyTorch deterministic-algorithms list)"),
    "index_add": ("A?", "float index_add on CUDA accumulates with atomics"),
    "scatter_add_": ("A?", "float scatter_add_ on CUDA accumulates with atomics"),
    "scatter_add": ("A?", "float scatter_add on CUDA accumulates with atomics"),
    "scatter_reduce_": ("A?", "scatter_reduce sum/mean/prod on CUDA accumulates with atomics; amax/amin are exact"),
    "scatter_reduce": ("A?", "scatter_reduce sum/mean/prod on CUDA accumulates with atomics; amax/amin are exact"),
    "index_put_": ("?", "order-dependent only with accumulate=True or duplicate indices"),
    "index_put": ("?", "order-dependent only with accumulate=True or duplicate indices"),
    "put_": ("?", "order-dependent only with accumulate=True or duplicate indices"),
    "bincount": ("?", "integer counts are exact; float weights accumulate with atomics"),
    "histc": ("B", "integer-valued counts; exact"),
    "cumsum": ("?", "PyTorch lists CUDA float cumsum as non-deterministic; integer cumsum is exact"),
    "scatter_": ("?", "store race if indices repeat and src is a tensor"),
    "scatter": ("?", "store race if indices repeat and src is a tensor"),
    "index_copy_": ("?", "undefined for duplicate indices; plain store otherwise"),
    "embedding_bag": ("?", "backward accumulates with atomics; forward is a plain reduce per bag"),
    "use_deterministic_algorithms": ("B", "opt-in to PyTorch deterministic mode; not a site, recorded for context"),
}

OPAQUE_CALLS = {
    "_scaled_mm": "cuBLASLt",
    "all_reduce": "NCCL", "reduce_scatter": "NCCL", "reduce_scatter_tensor": "NCCL", "all_to_all": "NCCL",
    "all_to_all_single": "NCCL", "all_gather": "NCCL", "all_gather_into_tensor": "NCCL",
    "cublas_gemm": "cuBLAS", "cudnn_batch_prefill_with_kv_cache": "cuDNN", "cudnn_batch_decode_with_kv_cache": "cuDNN",
    "load_cubin": "TensorRT-LLM cubin", "get_cubin": "TensorRT-LLM cubin", "download_cubin": "TensorRT-LLM cubin",
}

_TORCH_MATMUL_OPS = {"mm", "matmul", "bmm", "addmm", "baddbmm", "linear", "einsum"}

# CuTe DSL / MLIR-level atomics (FlashAttention-4 and similar Python-authored kernels)
_CUTE_ATOMIC_RE = re.compile(r"^(?:atomic_add|atomic_sub|atomic_fadd|atomicrmw|atomic_add_fp32|atomic_add_fp32x\d|atomic_max|atomic_min|atomic_cas|atomic_exch|red_add|red_release)[A-Za-z0-9_]*$")
_PTX_STR_RE = re.compile(r"(?<![A-Za-z0-9_.])(cp\.reduce\.async\.bulk[a-z_.:0-9]*|multimem\.(?:red|ld_reduce)\.[a-z_.:0-9]*|(?:red|atom)(?:\.[a-z_]+)*\.(?:add|inc|dec|min|max|and|or|xor|cas|exch)(?:\.noftz)?(?:\.v[248])?\.(?:f32|f64|f16x2|f16|bf16x2|bf16|s32|u32|s64|u64|b32|b64))")


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append(_dotted(node.func) + "()")
    else:
        parts.append("<expr>")
    return ".".join(reversed(parts))


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _const(node: ast.AST | None):
    if isinstance(node, ast.Constant):
        return node.value
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, src: str, rel: str, engine: str, sha: str):
        self.src_lines = src.split("\n")
        self.rel, self.engine, self.sha = rel, engine, sha
        self.stack: list[ast.AST] = []
        self.func_stack: list[tuple[str, bool]] = []
        self.out: list[Candidate] = []
        self.ptr_dtypes: dict[str, str] = {}

    # -- helpers ---------------------------------------------------------
    def _func(self) -> tuple[str, bool]:
        return self.func_stack[-1] if self.func_stack else ("<module>", False)

    def _snippet(self, line: int) -> tuple[int, str]:
        a = max(1, line - 2)
        b = min(len(self.src_lines), line + 3)
        return a, "\n".join(self.src_lines[a - 1:b])

    def _conditions(self) -> list[str]:
        conds = []
        for n in reversed(self.stack):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
            if isinstance(n, ast.If):
                conds.append("if " + ast.unparse(n.test)[:160])
            if isinstance(n, ast.While):
                conds.append("while " + ast.unparse(n.test)[:160])
        return conds

    def _emit(self, node: ast.AST, pattern: str, kind: str, match: str, hint: str, **extra) -> Candidate:
        fn, is_jit = self._func()
        c = Candidate(engine=self.engine, sha=self.sha, file=self.rel, line=node.lineno, col=node.col_offset + 1,
                      language="python", pattern=pattern, kind=kind, match=match, class_hint=hint,
                      function=fn, qualifiers=["triton.jit"] if is_jit else [], conditions=self._conditions())
        c.snippet_start, c.snippet = self._snippet(node.lineno)
        for k, v in extra.items():
            setattr(c, k, v)
        self.out.append(c)
        return c

    # -- visitors --------------------------------------------------------
    def generic_visit(self, node):
        self.stack.append(node)
        super().generic_visit(node)
        self.stack.pop()

    def _collect_dtypes(self, node: ast.AST) -> dict[str, str]:
        """Map local names to Triton dtypes from assignments like
        ``acc = tl.zeros(..., dtype=tl.float32)`` or ``x = y.to(tl.bfloat16)``."""
        found: dict[str, str] = {}
        for n in ast.walk(node):
            targets = []
            if isinstance(n, ast.Assign):
                targets = [t for t in n.targets if isinstance(t, ast.Name)]
                value = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                targets, value = [n.target], n.value
            else:
                continue
            txt = ast.unparse(value)
            m = re.search(r"tl\.(float32|float16|bfloat16|float64|int32|int64|uint32|int8|uint8|int16)\b", txt)
            if m and ("dtype=" in txt or ".to(" in txt or "tl.full" in txt or "tl.zeros" in txt):
                for t in targets:
                    found[t.id] = m.group(1)
        return found

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.ptr_dtypes = self._collect_dtypes(node)
        is_jit = any(_dotted(d if not isinstance(d, ast.Call) else d.func).split(".")[-1] in ("jit", "autotune", "heuristics", "constexpr")
                     or _dotted(d if not isinstance(d, ast.Call) else d.func) in ("triton.jit", "jit")
                     for d in node.decorator_list)
        self.func_stack.append((node.name, is_jit))
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and len(node.value) < 4000:
            for m in _PTX_STR_RE.finditer(node.value):
                txt = m.group(1)
                if txt.startswith("cp.reduce"):
                    kind, hint = "tma-reduce", ("A?" if ".add.f" in txt or txt.endswith(".add.f32") else "?")
                elif txt.startswith("multimem.ld_reduce"):
                    kind, hint = "ptx-multimem-ld_reduce", "C"
                elif txt.startswith("multimem.red"):
                    kind, hint = "ptx-multimem-red", ("A?" if re.search(r"\.(f32|f16|bf16|f16x2|bf16x2|f64)$", txt) else "B")
                else:
                    kind = "ptx-red" if txt.startswith("red") else "ptx-atom"
                    op = re.search(r"\.(add|inc|dec|min|max|and|or|xor|cas|exch)\.", txt + ".")
                    is_float = bool(re.search(r"\.(f32|f16|bf16|f16x2|bf16x2|f64)$", txt))
                    hint = "A?" if (op and op.group(1) == "add" and is_float) else "B"
                dtype = {"f32": "float32", "f64": "float64", "f16": "float16", "f16x2": "half2", "bf16": "bfloat16",
                         "bf16x2": "bfloat162", "s32": "int32", "u32": "uint32", "s64": "int64", "u64": "uint64",
                         "b32": "uint32", "b64": "uint64"}.get(txt.rsplit(".", 1)[-1], "unknown")
                c = self._emit(node, "ptx_string", kind, txt, hint, dtype_hint=dtype, memory_space_hint="global" if ".global" in txt else "unknown")
                c.notes.append("inline PTX in a Python string (CuTe DSL / Triton inline_asm)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = _dotted(node.func)
        last = name.split(".")[-1]
        base = name.rsplit(".", 1)[0] if "." in name else ""
        if last in TRITON_ATOMICS and (base.endswith("tl") or base == "triton.language" or base == "language"):
            kind, hint = TRITON_ATOMICS[last]
            args = [ast.unparse(a)[:120] for a in node.args]
            sem = _const(_kw(node, "sem"))
            dtype = "unknown"
            notes = []
            if kind == "tl.atomic_add" and len(node.args) >= 2:
                v = node.args[1]
                vs = ast.unparse(v)
                if isinstance(v, ast.Constant) and isinstance(v.value, int):
                    dtype, hint = "int32", "B"
                elif ".to(tl.int" in vs or ".to(tl.uint" in vs:
                    dtype, hint = "int32", "B"
                elif ".to(tl.float32" in vs or ".to(tl.float" in vs or "tl.float32" in vs:
                    dtype = "float32"
                elif ".to(tl.bfloat16" in vs:
                    dtype = "bfloat16"
                elif ".to(tl.float16" in vs:
                    dtype = "float16"
                elif "dtype.element_ty" in vs:
                    dtype = "template:pointer element type"
                    notes.append("value cast to the pointer element type; dtype follows the caller's tensor")
                elif isinstance(v, ast.Name) and v.id in self.ptr_dtypes:
                    t = self.ptr_dtypes[v.id]
                    dtype = t
                    if t.startswith(("int", "uint")):
                        hint = "B"
                    notes.append(f"dtype from assignment of {v.id} in the same function")
                else:
                    notes.append("dtype not inferable from the call; read the kernel and its launch")
            if sem:
                notes.append(f"sem={sem}")
            fn, _ = self._func()
            c = self._emit(node, kind, kind, name + "(", hint, args=args, dtype_hint=dtype, memory_space_hint="global")
            c.notes.extend(notes)
        elif last in TORCH_OPS and not base.endswith("tl"):
            hint, note = TORCH_OPS[last]
            args = [ast.unparse(a)[:120] for a in node.args]
            acc = _const(_kw(node, "accumulate"))
            reduce = _const(_kw(node, "reduce"))
            weights = _kw(node, "weights")
            notes = [note]
            if last in ("index_put_", "index_put", "put_"):
                if acc is True:
                    hint = "A?"; notes.append("accumulate=True")
                elif acc is None and len(node.args) >= 3 and last != "put_":
                    acc = _const(node.args[2]); hint = "A?" if acc is True else "?"
                else:
                    hint = "?"; notes.append("accumulate not True: plain store; order-dependent only with duplicate indices")
            if last in ("scatter_reduce_", "scatter_reduce"):
                if reduce in ("amax", "amin"):
                    hint = "B"
                elif reduce is None and len(node.args) >= 4:
                    reduce = _const(node.args[3])
                    hint = "B" if reduce in ("amax", "amin") else "A?"
                notes.append(f"reduce={reduce}")
            if last == "bincount":
                hint = "A?" if weights is not None else "B"
                notes.append("weights given" if weights is not None else "no weights: integer counts")
            if last in ("scatter_", "scatter"):
                if _kw(node, "value") is not None or (len(node.args) >= 3 and isinstance(node.args[2], ast.Constant)):
                    hint = "B"; notes.append("scalar value: race is benign (all writers store the same value)")
            if last == "use_deterministic_algorithms":
                notes.append("mode=" + (ast.unparse(node.args[0]) if node.args else "?"))
            c = self._emit(node, "torch_op", "torch-op", name + "(", hint, args=args)
            c.notes.extend(notes)
        elif _CUTE_ATOMIC_RE.match(last) and not base.endswith("tl"):
            kind = "atomic_ref" if "rmw" in last else ("atomicMax" if "max" in last else "atomicMin" if "min" in last else "atomicCAS-lock" if "cas" in last else "atomicExch" if "exch" in last else "other")
            hint = "B" if kind in ("atomicMax", "atomicMin", "atomicCAS-lock", "atomicExch") else "A?"
            dtype = "float32" if ("fp32" in last or "f32" in last or "FADD" in ast.unparse(node)) else "unknown"
            args = [ast.unparse(a)[:120] for a in node.args]
            if len(node.args) >= 2 and isinstance(node.args[-1], ast.Constant) and isinstance(node.args[-1].value, int):
                dtype, hint = "int32", "B"
            c = self._emit(node, "cute_atomic", kind if kind != "other" else "other", name + "(", hint, args=args, dtype_hint=dtype, memory_space_hint="global")
            c.notes.append("CuTe DSL / MLIR atomic helper; read the helper body for the exact instruction")
        elif last in OPAQUE_CALLS and ("torch" in name or "dist" in name or "flashinfer" in name or last.startswith(("cublas", "cudnn", "load_cubin", "get_cubin", "download_cubin"))):
            c = self._emit(node, "opaque_call", "library-call", name + "(", "C")
            c.notes.append(f"opaque library: {OPAQUE_CALLS[last]}")
        self.generic_visit(node)


def scan_python_source(source: str, rel_path: str, engine: str, sha: str) -> list[Candidate]:
    if not any(t in source for t in ("atomic", "index_add", "scatter", "index_put", "bincount", "histc", "cumsum",
                                    "put_(", "index_copy", "embedding_bag", "use_deterministic", "_scaled_mm",
                                    "all_reduce", "reduce_scatter", "all_to_all", "all_gather", "cubin", "cublas", "cudnn",
                                    "cp.reduce", "multimem", "red.", "atom.")):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        c = Candidate(engine=engine, sha=sha, file=rel_path, line=e.lineno or 1, col=1, language="python",
                      pattern="parse_error", kind="other", match="", class_hint="?", excluded_reason="parse-error",
                      notes=[f"could not parse: {e.msg}"])
        return [c]
    v = _Visitor(source, rel_path, engine, sha)
    v.visit(tree)
    return v.out


def scan_python_path(path: Path, rel_path: str, engine: str, sha: str) -> list[Candidate]:
    return scan_python_source(path.read_text(errors="replace"), rel_path, engine, sha)
