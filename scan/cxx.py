"""tree-sitter helpers for C++ and CUDA sources.

tree-sitter-cpp does not know CUDA, but it parses ``__global__``,
``__device__`` and ``__shared__`` as attributes or identifiers and only
produces localised error nodes around ``<<<...>>>`` launch syntax. That is
enough to answer the three questions the scanner asks: is this byte inside a
comment or string, which function encloses it, and which ``if`` conditions
enclose it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

_LANG = Language(tree_sitter_cpp.language())

CONTAINER_NODE_TYPES = ("struct_specifier", "class_specifier", "union_specifier")


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(_LANG)


@dataclass
class ParsedFile:
    source: bytes
    root: Node
    lines: list[str] = field(default_factory=list)
    dead_ranges: list[tuple[int, int]] = field(default_factory=list)  # 1-based inclusive line ranges under #if 0

    @classmethod
    def parse(cls, source: bytes) -> "ParsedFile":
        tree = _parser().parse(source)
        text = source.decode("utf-8", errors="replace")
        pf = cls(source=source, root=tree.root_node, lines=text.split("\n"))
        pf.dead_ranges = _dead_ranges(pf.lines)
        return pf

    # -- position helpers -------------------------------------------------
    def node_at(self, byte_offset: int) -> Node:
        return self.root.descendant_for_byte_range(byte_offset, byte_offset + 1)

    def in_comment(self, byte_offset: int) -> bool:
        n = self.node_at(byte_offset)
        while n is not None:
            if n.type == "comment":
                return True
            n = n.parent
        return False

    def in_string(self, byte_offset: int) -> bool:
        n = self.node_at(byte_offset)
        while n is not None:
            if n.type in ("string_literal", "raw_string_literal", "char_literal", "concatenated_string", "string_content"):
                return True
            n = n.parent
        return False

    def in_dead_block(self, line: int) -> bool:
        return any(a <= line <= b for a, b in self.dead_ranges)

    # -- enclosing context -------------------------------------------------
    def enclosing_function(self, byte_offset: int) -> tuple[str, Node | None]:
        """Return (name, node) of the innermost function definition.

        Falls back to the innermost struct/class name (CUTLASS-style functors
        with static ``copy`` methods) and finally to ``<file-scope>``.
        """
        n = self.node_at(byte_offset)
        container = None
        while n is not None:
            if n.type == "function_definition":
                return _function_name(n), n
            if container is None and n.type in CONTAINER_NODE_TYPES:
                container = n
            n = n.parent
        if container is not None:
            name_node = container.child_by_field_name("name")
            if name_node is not None:
                return name_node.text.decode() + " (struct scope)", container
        return "<file-scope>", None

    def enclosing_conditions(self, byte_offset: int, limit: int = 4) -> list[str]:
        """Text of the ``if``/``else`` conditions enclosing the site, innermost first."""
        out: list[str] = []
        n = self.node_at(byte_offset)
        prev = None
        while n is not None and len(out) < limit:
            if n.type == "if_statement":
                cond = n.child_by_field_name("condition")
                is_else = prev is not None and prev.type == "else_clause"
                if cond is not None:
                    txt = _squash(cond.text.decode("utf-8", errors="replace"))
                    is_constexpr = any(c.type == "constexpr" or c.text == b"constexpr" for c in n.children)
                    prefix = "if constexpr " if is_constexpr else "if "
                    out.append(("else of " if is_else else "") + prefix + txt)
            if n.type == "function_definition":
                break
            prev = n
            n = n.parent
        return out

    def early_return_guards(self, byte_offset: int, limit: int = 4) -> list[str]:
        """Conditions of preceding sibling ``if`` statements whose body returns.

        ``if (!use_atomics) { ...; return; } atomicAdd(...)`` guards the atomic
        just as an enclosing ``if`` would, so these are reported alongside the
        enclosing conditions, prefixed with ``after return-guard``.
        """
        out: list[str] = []
        n = self.node_at(byte_offset)
        while n is not None and n.type != "function_definition" and len(out) < limit:
            parent = n.parent
            if parent is not None and parent.type == "compound_statement":
                for sib in parent.children:
                    if sib.start_byte >= n.start_byte:
                        break
                    if sib.type == "if_statement" and b"return" in sib.text:
                        cond = sib.child_by_field_name("condition")
                        if cond is not None:
                            out.append("after return-guard if " + _squash(cond.text.decode("utf-8", errors="replace")))
            n = parent
        return out

    def macro_definition_name(self, byte_offset: int) -> str | None:
        """Name of the ``#define NAME(...)`` whose body contains the offset, if any."""
        n = self.node_at(byte_offset)
        while n is not None:
            if n.type in ("preproc_function_def", "preproc_def"):
                name = n.child_by_field_name("name")
                return name.text.decode() if name is not None else None
            n = n.parent
        # tree-sitter may leave a multi-line #define as preproc_arg text; fall back to the line
        line = self.source.count(b"\n", 0, byte_offset)
        i = line
        while i >= 0:
            m = re.match(r"\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", self.lines[i])
            if m:
                return m.group(1)
            if not self.lines[i].rstrip().endswith("\\"):
                break
            i -= 1
        return None

    def template_parameters(self, func_node: Node | None) -> str:
        """Template parameter list text of the enclosing template, if any."""
        n = func_node
        while n is not None:
            if n.type == "template_declaration":
                params = n.child_by_field_name("parameters")
                if params is not None:
                    return _squash(params.text.decode("utf-8", errors="replace"))
            n = n.parent
        return ""

    def function_qualifiers(self, func_node: Node | None) -> list[str]:
        """CUDA execution-space qualifiers found on the function definition."""
        if func_node is None:
            return []
        head = func_node.text[: max(0, func_node.text.find(b"{"))].decode("utf-8", errors="replace")
        return [q for q in ("__global__", "__device__", "__host__", "CUTLASS_DEVICE", "CUTE_DEVICE",
                            "CUTLASS_GLOBAL", "CUTE_HOST_DEVICE", "CUTLASS_HOST_DEVICE", "__forceinline__")
                if q in head]

    def function_text(self, func_node: Node | None) -> str:
        if func_node is None:
            return ""
        return func_node.text.decode("utf-8", errors="replace")

    def function_line_count(self, func_node: Node | None) -> int:
        if func_node is None:
            return 0
        return func_node.end_point[0] - func_node.start_point[0] + 1

    def call_arguments(self, byte_offset: int) -> list[str]:
        """Arguments of the call expression that starts at or contains byte_offset."""
        n = self.node_at(byte_offset)
        while n is not None and n.type != "call_expression":
            n = n.parent
        if n is None:
            return []
        args = n.child_by_field_name("arguments")
        if args is None:
            return []
        return [_squash(c.text.decode("utf-8", errors="replace")) for c in args.named_children]


def _function_name(fn: Node) -> str:
    decl = fn.child_by_field_name("declarator")
    while decl is not None and decl.type != "function_declarator":
        inner = decl.child_by_field_name("declarator")
        if inner is None:
            break
        decl = inner
    if decl is None:
        return "<unknown>"
    name = decl.child_by_field_name("declarator")
    if name is None:
        return "<unknown>"
    txt = name.text.decode("utf-8", errors="replace")
    txt = re.sub(r"<.*$", "", txt, flags=re.S)
    return txt.strip() or "<unknown>"


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_IF_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def _dead_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Line ranges under ``#if 0`` (nesting aware; the ``#else`` branch is live)."""
    ranges: list[tuple[int, int]] = []
    stack: list[tuple[bool, int]] = []
    for i, line in enumerate(lines, start=1):
        m = _IF_RE.match(line)
        if not m:
            continue
        kw, rest = m.group(1), m.group(2).strip()
        if kw in ("if", "ifdef", "ifndef"):
            dead = kw == "if" and rest.split("//")[0].strip() in ("0", "false")
            stack.append((dead, i))
        elif kw in ("else", "elif") and stack:
            dead, start = stack.pop()
            if dead:
                ranges.append((start, i))
            stack.append((False, i))
        elif kw == "endif" and stack:
            dead, start = stack.pop()
            if dead:
                ranges.append((start, i))
    return ranges
