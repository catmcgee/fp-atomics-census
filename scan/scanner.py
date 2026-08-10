"""Core candidate scanner for CUDA, C++ and inline-PTX sources.

The scanner emits one candidate per pattern match with enough context for a
human to classify it: enclosing function and its qualifiers, template
parameters, enclosing ``if`` conditions, call arguments, a dtype hint and a
memory-space hint. Nothing here decides the taxonomy class.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cxx import ParsedFile
from .patterns import (
    CUDA_PATTERNS,
    CXX_TYPE_TO_DTYPE,
    FLOAT_DTYPES,
    INT_DTYPES,
    OPAQUE_PATTERNS,
    TEMPLATE_TYPE_NAMES,
    WRAPPER_EXCLUDE,
    ptx_dtype,
    ptx_is_float,
)

CUDA_EXTENSIONS = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".inc", ".inl", ".hh", ".hxx"}
PYTHON_EXTENSIONS = {".py"}

_PRIMITIVE_KINDS_FLOAT_CAPABLE = {"atomicAdd", "atomicSub", "atomicCAS", "ptx-red", "ptx-atom", "tma-reduce",
                                  "atomic_ref", "ptx-multimem-red", "atomicMax", "atomicMin"}
_DEVICE_QUALIFIERS = {"__device__", "__global__", "CUTLASS_DEVICE", "CUTE_DEVICE", "CUTLASS_GLOBAL",
                      "CUTE_HOST_DEVICE", "CUTLASS_HOST_DEVICE"}


@dataclass
class Candidate:
    engine: str
    sha: str
    file: str
    line: int
    col: int
    language: str
    pattern: str
    kind: str
    match: str
    class_hint: str
    dtype_hint: str = "unknown"
    memory_space_hint: str = "unknown"
    function: str = "<unknown>"
    qualifiers: list[str] = field(default_factory=list)
    template_params: str = ""
    function_lines: int = 0
    conditions: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    in_comment: bool = False
    in_string: bool = False
    in_dead_block: bool = False
    excluded_reason: str = ""
    snippet_start: int = 0
    snippet: str = ""
    notes: list[str] = field(default_factory=list)
    wrapper_of: str = ""
    in_scope: bool = True
    id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # id first for readability
        return {"id": d.pop("id"), **d}


# ---------------------------------------------------------------------------
# dtype and memory-space inference
# ---------------------------------------------------------------------------

_CAST_RE = re.compile(
    r"(?:reinterpret_cast|static_cast|const_cast)\s*<\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*(?:\s+[a-z]+)*)\s*\*|"
    r"\(\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*(?:\s+(?:int|long|char|short))*)\s*\*\s*\)")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE_WORDS = "|".join(sorted((re.escape(t) for t in CXX_TYPE_TO_DTYPE), key=len, reverse=True))
_TEMPLATE_WORDS = "|".join(sorted((re.escape(t) for t in TEMPLATE_TYPE_NAMES), key=len, reverse=True))


def _base_identifier(expr: str) -> str | None:
    """Leading identifier of a pointer expression: ``&out[i % 4]`` -> ``out``."""
    s = expr.strip()
    s = re.sub(r"^\(*\s*&?\s*", "", s)
    s = re.sub(r"^(?:reinterpret_cast|static_cast|const_cast)\s*<[^>]*>\s*\(", "", s)
    s = re.sub(r"^\(\s*[A-Za-z_][A-Za-z0-9_:\s]*\*+\s*\)\s*", "", s)
    s = re.sub(r"^\(*\s*&?\s*", "", s)
    m = _IDENT_RE.match(s)
    return m.group(0) if m else None


def _declared_dtype(name: str, func_text: str) -> tuple[str, str]:
    """(dtype, memory_space) for ``name`` from declarations in the function text."""
    if not name or not func_text:
        return "unknown", "unknown"
    # __shared__ T name[...]
    m = re.search(r"__shared__\s+(?:__align__\(\d+\)\s+)?(?:const\s+)?(?:volatile\s+)?(%s|%s)\s+(?:\*\s*)?%s\b"
                  % (_TYPE_WORDS, _TEMPLATE_WORDS, re.escape(name)), func_text)
    if m:
        return _type_to_dtype(m.group(1)), "shared"
    # T* name   /  T *__restrict__ name  / T name[]
    m = re.search(r"(?<![A-Za-z0-9_])(?:const\s+)?(?:volatile\s+)?(%s|%s)\s*(?:const\s*)?(?:\*+\s*(?:const\s*)?(?:__restrict__\s*)?|\s)\s*%s\s*(?:[\[,;)=]|$)"
                  % (_TYPE_WORDS, _TEMPLATE_WORDS, re.escape(name)), func_text, flags=re.M)
    if m:
        return _type_to_dtype(m.group(1)), "global" if "*" in m.group(0) else "unknown"
    return "unknown", "unknown"


def _type_to_dtype(t: str) -> str:
    t = re.sub(r"\s+", " ", t.strip())
    if t in CXX_TYPE_TO_DTYPE:
        return CXX_TYPE_TO_DTYPE[t]
    if t in TEMPLATE_TYPE_NAMES:
        return "template:" + t
    return "unknown"


def _literal_dtype(val: str) -> str:
    v = val.strip()
    if re.fullmatch(r"-?\d+[uUlL]*", v):
        return "int32" if not re.search(r"[lL]", v) else "int64"
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]", v):
        return "float32"
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", v):
        return "float64"
    return "unknown"


def infer_cuda_dtype_and_space(args: list[str], func_text: str) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    if not args:
        return "unknown", "unknown", ["no call arguments parsed"]
    ptr = args[0]
    m = _CAST_RE.search(ptr)
    if m:
        t = (m.group(1) or m.group(2) or "").strip()
        dt = _type_to_dtype(t)
        if dt != "unknown":
            base = _base_identifier(ptr)
            _, space = _declared_dtype(base or "", func_text)
            if "__shared__" in func_text and base and re.search(r"__shared__[^;]*\b%s\b" % re.escape(base), func_text):
                space = "shared"
            return dt, space, [f"dtype from cast to {t}*"]
    base = _base_identifier(ptr)
    dt, space = _declared_dtype(base or "", func_text)
    if dt != "unknown":
        notes.append(f"dtype from declaration of {base}")
    elif len(args) > 1:
        lit = _literal_dtype(args[1])
        if lit != "unknown":
            dt = lit
            notes.append("dtype from value literal")
    if base and space == "unknown":
        if base.startswith(("smem", "s_", "sh_", "shared")):
            space = "shared"
            notes.append("memory space guessed from identifier prefix")
    return dt, space, notes


def _class_hint_for_dtype(kind: str, dtype: str, default: str) -> str:
    if kind in ("atomicMax", "atomicMin", "atomicExch", "atomicInc", "atomicDec", "atomicOr", "atomicAnd", "atomicXor"):
        return "B"
    if dtype in FLOAT_DTYPES:
        return "A?"
    if dtype in INT_DTYPES:
        return "B" if kind != "atomicCAS" else "?"
    return default


# ---------------------------------------------------------------------------
# per-file scan
# ---------------------------------------------------------------------------

_ANY_CUDA = re.compile("|".join(f"(?:{p.pattern})" for _, p, _, _ in CUDA_PATTERNS))
_ANY_OPAQUE = re.compile("|".join(f"(?:{p.pattern})" for _, p, _ in OPAQUE_PATTERNS))


def _snippet(lines: list[str], line: int, before: int = 2, after: int = 3) -> tuple[int, str]:
    a = max(1, line - before)
    b = min(len(lines), line + after)
    return a, "\n".join(lines[a - 1:b])


def _byte_to_linecol(source: bytes, offset: int) -> tuple[int, int]:
    line = source.count(b"\n", 0, offset) + 1
    last_nl = source.rfind(b"\n", 0, offset)
    col = offset - (last_nl + 1) + 1
    return line, col


def scan_cxx_source(source: bytes, rel_path: str, engine: str, sha: str, language: str = "cuda",
                    include_opaque: bool = True) -> list[Candidate]:
    text = source.decode("utf-8", errors="replace")
    if not _ANY_CUDA.search(text) and not (include_opaque and _ANY_OPAQUE.search(text)):
        return []
    pf = ParsedFile.parse(source)
    out: list[Candidate] = []

    for name, rx, kind, hint in CUDA_PATTERNS:
        for m in rx.finditer(text):
            off = len(text[: m.start()].encode("utf-8")) if not text.isascii() else m.start()
            line, col = _byte_to_linecol(source, off)
            fn_name, fn_node = pf.enclosing_function(off)
            quals = pf.function_qualifiers(fn_node)
            fn_text = pf.function_text(fn_node)
            cand = Candidate(
                engine=engine, sha=sha, file=rel_path, line=line, col=col, language=language,
                pattern=name, kind=kind, match=m.group(0).strip(), class_hint=hint,
                function=fn_name, qualifiers=quals,
                template_params=pf.template_parameters(fn_node),
                function_lines=pf.function_line_count(fn_node),
                conditions=pf.enclosing_conditions(off) + pf.early_return_guards(off),
                in_comment=pf.in_comment(off), in_string=pf.in_string(off),
                in_dead_block=pf.in_dead_block(line),
            )
            if pf.in_declarator(off) and not name.startswith("ptx_"):
                cand.excluded_reason = "declaration"
                cand.notes.append("function declarator or parameter, not a call")
            if fn_node is None:
                macro = pf.macro_definition_name(off)
                if macro:
                    cand.function = f"{macro} (macro)"
                    cand.notes.append("inside a #define; call sites are reported as wrapper-call")
            cand.snippet_start, cand.snippet = _snippet(pf.lines, line)
            _enrich(cand, name, m, pf, off, fn_text, fn_node)
            out.append(cand)

    if include_opaque:
        for name, rx, lib in OPAQUE_PATTERNS:
            for m in rx.finditer(text):
                off = len(text[: m.start()].encode("utf-8")) if not text.isascii() else m.start()
                line, col = _byte_to_linecol(source, off)
                fn_name, fn_node = pf.enclosing_function(off)
                cand = Candidate(
                    engine=engine, sha=sha, file=rel_path, line=line, col=col, language=language,
                    pattern=name, kind="library-call", match=m.group(0).strip(), class_hint="C",
                    function=fn_name, qualifiers=pf.function_qualifiers(fn_node),
                    in_comment=pf.in_comment(off), in_string=pf.in_string(off), in_dead_block=pf.in_dead_block(line),
                    notes=[f"opaque library: {lib}"],
                )
                cand.snippet_start, cand.snippet = _snippet(pf.lines, line)
                if cand.in_comment or cand.in_string:
                    cand.excluded_reason = "comment" if cand.in_comment else "string"
                out.append(cand)
    return out


def _enrich(cand: Candidate, name: str, m: re.Match, pf: ParsedFile, off: int, fn_text: str, fn_node) -> None:
    kind = cand.kind
    if name.startswith("ptx_"):
        # PTX lives in asm strings; in_string is expected and not a false-positive signal.
        cand.in_string = False
        cand.notes.append("inline PTX")
        if cand.in_comment:
            cand.excluded_reason = "comment"
        elif cand.in_dead_block:
            cand.excluded_reason = "dead-block"
        elif not _in_asm_context(pf, off):
            cand.excluded_reason = "string"
            cand.notes.append("PTX-looking text outside an asm statement")
        if name in ("ptx_red", "ptx_atom"):
            op, suffix = m.group(1), m.group(2)
            cand.dtype_hint = ptx_dtype(suffix)
            cand.memory_space_hint = "shared" if ".shared" in m.group(0) else ("global" if ".global" in m.group(0) else "unknown")
            if op in ("min", "max", "and", "or", "xor", "inc", "dec", "cas", "exch"):
                cand.class_hint = "B"
            elif op == "add":
                cand.class_hint = "A?" if ptx_is_float(suffix) else "B"
        elif name == "ptx_multimem_red":
            suf = m.group(0).rsplit(".", 1)[-1]
            cand.dtype_hint = ptx_dtype(suf)
            cand.memory_space_hint = "global"
            cand.class_hint = "A?" if ptx_is_float(suf) else "B"
            cand.notes.append("multicast reduction across NVLink; the switch performs the add")
        elif name == "ptx_multimem_ld_reduce":
            suf = m.group(0).rsplit(".", 1)[-1]
            cand.dtype_hint = ptx_dtype(suf)
            cand.memory_space_hint = "global"
            cand.notes.append("hardware multicast load-reduce; in-switch reduction order is undocumented")
        elif name == "ptx_tma_reduce":
            suf = re.search(r"\.(f32|f64|f16|bf16|s32|u32|s64|u64|b32|b64)\b", m.group(0))
            cand.dtype_hint = ptx_dtype(suf.group(1)) if suf else "unknown"
            cand.memory_space_hint = "global"
            cand.class_hint = "A?" if (".add" in m.group(0) and (not suf or ptx_is_float(suf.group(1)))) else ("B" if re.search(r"\.(min|max|and|or|xor|inc|dec)", m.group(0)) else "?")
            cand.notes.append("TMA bulk reduction into global memory")
        return

    if name == "cute_tma_reduce":
        cand.in_string = False
        cand.memory_space_hint = "global"
        cand.class_hint = "A?" if "ADD" in m.group(0) else "B"
        cand.notes.append("TMA reduction wrapper; dtype is the tensor element type at the call site")
        return

    if name == "cuda_atomic":
        targ = re.search(r"atomic(?:_ref)?\s*<\s*([^,>]+)", pf.function_text(fn_node) or "")
        seg = pf.lines[cand.line - 1]
        t = re.search(r"atomic(?:_ref)?\s*<\s*([^,>]+)", seg)
        if t:
            cand.dtype_hint = _type_to_dtype(t.group(1).strip())
            cand.class_hint = _class_hint_for_dtype("atomic_ref", cand.dtype_hint, "?")
        return

    if name == "fetch_add":
        if not (set(cand.qualifiers) & _DEVICE_QUALIFIERS) and "__device__" not in fn_text and cand.file.endswith((".cpp", ".cc", ".cxx")):
            cand.excluded_reason = "host"
            cand.notes.append("fetch_add outside device code (std::atomic on the host)")
        return

    if name == "atomic_wrapper":
        ident = m.group(0).rstrip("( \t")
        if ident in WRAPPER_EXCLUDE:
            cand.excluded_reason = "not-an-atomic"
        cand.notes.append(f"call to atomic-named helper {ident}; see the wrapper definition")
        cand.args = pf.call_arguments(off)
        return

    # CUDA intrinsic atomics
    cand.args = pf.call_arguments(off)
    dt, space, notes = infer_cuda_dtype_and_space(cand.args, fn_text)
    cand.dtype_hint, cand.memory_space_hint = dt, space
    cand.notes.extend(notes)
    if kind in ("atomicMax", "atomicMin") and re.search(r"__float_as_(?:int|uint)|__double_as_longlong", " ".join(cand.args)):
        cand.notes.append("float max/min through integer reinterpretation; exact")
        cand.dtype_hint = "float32" if "__float_as" in " ".join(cand.args) else "float64"
    if kind == "atomicCAS":
        reinterp = re.search(r"__float_as_(?:int|uint)|__int_as_float|__uint_as_float|__double_as_longlong|__longlong_as_double|__half_as_ushort|__ushort_as_half|__bfloat16_as_ushort|__ushort_as_bfloat16|__half2_as_uint|__bfloat162_as_uint", fn_text)
        if reinterp:
            cand.notes.append("CAS loop with float reinterpretation in the same function: likely a float add")
            cand.class_hint = "A?"
            if cand.dtype_hint in INT_DTYPES or cand.dtype_hint == "unknown":
                for pat, dt in (("__float_as", "float32"), ("__double_as", "float64"), ("__half2_as", "half2"),
                                ("__bfloat162_as", "bfloat162"), ("__half_as", "float16"), ("__ushort_as_half", "float16"),
                                ("__bfloat16_as", "bfloat16"), ("__ushort_as_bfloat16", "bfloat16")):
                    if pat in fn_text:
                        cand.dtype_hint = dt
                        break
        else:
            cand.notes.append("CAS without float reinterpretation nearby: likely a lock or flag")
            cand.class_hint = "B"
    cand.class_hint = _class_hint_for_dtype(kind, cand.dtype_hint, cand.class_hint) if kind != "atomicCAS" else cand.class_hint
    if cand.in_comment or cand.in_dead_block:
        cand.excluded_reason = "comment" if cand.in_comment else "dead-block"
    elif cand.in_string:
        cand.excluded_reason = "string"
    if kind in ("atomicAdd", "atomicSub") and cand.dtype_hint.startswith("template:"):
        cand.notes.append("dtype is a template parameter; list the launched instantiations")


def _in_asm_context(pf: ParsedFile, off: int) -> bool:
    n = pf.node_at(off)
    while n is not None:
        if n.type in ("gnu_asm_expression", "asm_statement") or (n.type == "expression_statement" and n.text.lstrip().startswith(b"asm")):
            return True
        n = n.parent
    # tree-sitter-cpp sometimes parses asm volatile(...) as a call_expression
    n = pf.node_at(off)
    while n is not None:
        if n.type == "call_expression" and n.text.lstrip().startswith(b"asm"):
            return True
        n = n.parent
    return False


# ---------------------------------------------------------------------------
# wrapper linking: definitions of small device helpers that contain a float
# atomic, and the call sites of those helpers across the repo
# ---------------------------------------------------------------------------

_PRIMITIVE_NAMES = {"atomicAdd", "atomicSub", "atomicCAS", "atomicExch", "atomicMax", "atomicMin", "atomicInc",
                    "atomicDec", "atomicOr", "atomicAnd", "atomicXor", "atomicAdd_block", "atomicAdd_system",
                    "atomic_add", "atomic_max", "atomic_min", "atomic_cas", "atomic_exch"}


def wrapper_definitions(cands: list[Candidate], max_lines: int = 80) -> dict[str, Candidate]:
    """Map helper-function name -> defining candidate.

    A helper counts when it is a small device function or a macro whose body
    contains an atomic on a floating-point (or template/unknown) operand.
    Integer-only helpers are not linked, and a function that overloads a
    primitive name (``__device__ half atomicAdd(half*, half)``) is reported
    as a site but never used as a wrapper, since that would link every call
    of the primitive in the repository to it.
    """
    defs: dict[str, Candidate] = {}
    for c in cands:
        if c.excluded_reason or c.kind not in _PRIMITIVE_KINDS_FLOAT_CAPABLE:
            continue
        fn = c.function
        if fn in _PRIMITIVE_NAMES or fn.replace(" (macro)", "") in _PRIMITIVE_NAMES:
            c.notes.append("defines an overload of a primitive atomic name; not used as a wrapper")
            continue
        if c.dtype_hint in INT_DTYPES:
            continue
        if fn.endswith(" (macro)"):
            defs.setdefault(fn[: -len(" (macro)")], c)
            continue
        if "__global__" in c.qualifiers or "CUTLASS_GLOBAL" in c.qualifiers:
            continue
        if not (set(c.qualifiers) & _DEVICE_QUALIFIERS):
            continue
        if fn.startswith("<") or "(struct scope)" in fn or fn in defs:
            continue
        if fn in ("operator()", "copy", "store", "run", "invoke", "apply", "step", "epilogue"):
            continue
        if c.function_lines and c.function_lines > max_lines:
            continue
        defs[fn] = c
    return defs


def scan_wrapper_calls(source: bytes, rel_path: str, engine: str, sha: str, defs: dict[str, Candidate],
                       language: str = "cuda") -> list[Candidate]:
    if not defs:
        return []
    text = source.decode("utf-8", errors="replace")
    rx = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in sorted(defs, key=len, reverse=True)) + r")\s*\(")
    if not rx.search(text):
        return []
    pf = ParsedFile.parse(source)
    out: list[Candidate] = []
    for m in rx.finditer(text):
        name = m.group(1)
        d = defs[name]
        off = len(text[: m.start()].encode("utf-8")) if not text.isascii() else m.start()
        line, col = _byte_to_linecol(source, off)
        if d.file == rel_path and d.function == name:
            fn_name, fn_node = pf.enclosing_function(off)
            if fn_name == name:
                continue  # inside the definition itself (recursion or the declarator)
        fn_name, fn_node = pf.enclosing_function(off)
        if fn_name == name and d.file == rel_path:
            continue
        # skip the definition line itself (function or macro)
        if d.file == rel_path and line == d.line and d.function.endswith(" (macro)"):
            continue
        if d.file == rel_path and abs(line - d.line) < 2 and pf.lines[line - 1].lstrip().startswith(("__device__", "template", "static", "inline", "CUTLASS", "CUTE")):
            continue
        cand = Candidate(
            engine=engine, sha=sha, file=rel_path, line=line, col=col, language=language,
            pattern=f"wrapper_call:{name}", kind="wrapper-call", match=m.group(0).strip(),
            class_hint=d.class_hint, dtype_hint=d.dtype_hint, memory_space_hint="unknown",
            function=fn_name, qualifiers=pf.function_qualifiers(fn_node),
            template_params=pf.template_parameters(fn_node), conditions=pf.enclosing_conditions(off),
            in_comment=pf.in_comment(off), in_string=pf.in_string(off), in_dead_block=pf.in_dead_block(line),
            wrapper_of=d.id, notes=[f"calls {name} defined at {d.file}:{d.line}"],
        )
        cand.args = pf.call_arguments(off)
        fn_text = pf.function_text(fn_node)
        dt, space, notes = infer_cuda_dtype_and_space(cand.args, fn_text)
        if dt != "unknown":
            cand.dtype_hint = dt
            cand.notes.extend(notes)
        cand.memory_space_hint = space
        cand.snippet_start, cand.snippet = _snippet(pf.lines, line)
        if cand.in_comment or cand.in_string or cand.in_dead_block:
            cand.excluded_reason = "comment" if cand.in_comment else ("string" if cand.in_string else "dead-block")
        out.append(cand)
    return out


def scan_path(path: Path, rel_path: str, engine: str, sha: str, include_opaque: bool = True) -> list[Candidate]:
    ext = path.suffix.lower()
    if ext in CUDA_EXTENSIONS:
        language = "cuda" if ext in (".cu", ".cuh") else "cpp"
        source = path.read_bytes()
        cands = scan_cxx_source(source, rel_path, engine, sha, language=language, include_opaque=include_opaque)
        return cands
    return []
