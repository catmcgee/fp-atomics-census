"""Move the census from one upstream commit to a newer one without re-triaging by hand.

    python -m scan.repin plan --out <dir> [--only vllm ...] [--before 2026-09-15T00:00:00Z]
    python -m scan.repin apply --only cutlass [--resolutions reviewed.jsonl] [--before ...]

``plan`` picks, for each repository in scan-manifest.json, the last commit on
its default branch before the cut-off, and maps every inventory location from
the pinned sha to that commit with git's own diff: rename detection over the
whole tree (``git diff -M -C -l0 --name-status``) and zero-context hunks per
file (``git diff -U0`` of the two blobs, Myers algorithm). A cited range whose
lines all lie in unchanged lines is shifted exactly and then verified by
comparing its source text at both commits. The scanner is re-run at the new
commit through the Makefile flow (checkout in repos/, ``scan.run --sha``), and
candidate links are carried across by location and pattern, never by position,
because candidate ids are positional. Rows that cannot be carried mechanically
go on a worklist with the reason, the old and new locations and the diff of
the relevant hunks. ``plan`` leaves the tree unchanged and restores each
checkout to its pinned sha.

``apply`` writes one repository's migration into the tree: its inventory rows,
references into it from other repositories' rows, its candidates, its manifest
entry, its disposition ledger and its checkout. It refuses while any worklist
row lacks a reviewed replacement at the new sha (``--resolutions``). Run
``make docs validate check-derived test`` afterwards.

Location fields handled: the row's ``file``/``line_start``/``line_end``,
``evidence[].ref``, ``path.entry_points[]`` and ``gate.read_at`` (unless the
gate's scope is external_runtime); a ``repos/<engine>/`` prefix maps through
that engine's diff. Prose fields are never rewritten: line numbers in
``notes``, ``path.description``, ``evidence[].claim``, ``gate.flag``,
``gate.note``, ``gate.default``, ``downstream.argument``,
``opaque_target.note``, ``default_path_condition`` and ``function`` must be
invariant (unchanged line, zero shift) in every file they could refer to, or
the row goes on the worklist.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUTOFF = "2026-09-15T00:00:00Z"

_REF = re.compile(r"^(?P<file>[^:\s]+):(?P<start>\d+)(?:-(?P<end>\d+))?$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# Scanner fields that describe a candidate's context. A linked candidate whose
# line is unchanged but whose context differs goes to review.
CONTEXT_FIELDS = ("language", "kind", "match", "class_hint", "dtype_hint", "memory_space_hint", "function",
                  "qualifiers", "template_params", "conditions", "args", "in_comment", "in_string",
                  "in_dead_block", "excluded_reason", "in_scope")

PROSE_FIELDS = ("function", "notes", "path.description", "evidence[].claim", "gate.flag", "gate.note",
                "gate.default", "downstream.argument", "opaque_target.note", "default_path_condition")

_EXT = r"(?:cu|cuh|h|hh|hpp|hxx|c|cc|cpp|cxx|inl|inc|py)"
_NUMLIST = r"\d+(?:\s*-\s*\d+)?(?:(?:\s*,\s*|\s+and\s+)\d+(?:\s*-\s*\d+)?)*"
PROSE_FILE_REF = re.compile(rf"(?P<file>[A-Za-z0-9_][A-Za-z0-9_./-]*\.{_EXT}):(?P<nums>{_NUMLIST})")
PROSE_LINE_WORD = re.compile(rf"\b(?:lines?|at)\s+(?P<nums>{_NUMLIST})")
PROSE_PAREN = re.compile(rf"\((?:also\s+|lines?\s+)?(?P<nums>{_NUMLIST})\s*[;),]")


# ---------------------------------------------------------------------------
# hunks and line mapping (pure)
# ---------------------------------------------------------------------------

Hunk = tuple[int, int, int, int]  # old_start, old_count, new_start, new_count


def parse_hunks(diff_text: str) -> list[Hunk] | None:
    """Hunks of a zero-context diff; None when git reports a binary difference."""
    hunks: list[Hunk] = []
    for line in diff_text.splitlines():
        m = _HUNK.match(line)
        if m:
            hunks.append((int(m[1]), 1 if m[2] is None else int(m[2]), int(m[3]), 1 if m[4] is None else int(m[4])))
        elif line.startswith("Binary files "):
            return None
    return sorted(hunks)


def map_line(hunks: list[Hunk], line: int) -> int | None:
    """New number of an old line, or None when the line was changed or deleted."""
    offset = 0
    for old_start, old_count, new_start, new_count in hunks:
        if old_count == 0:  # insertion after old line old_start
            if line <= old_start:
                break
            offset += new_count
            continue
        if line < old_start:
            break
        if line < old_start + old_count:
            return None
        offset += new_count - old_count
    return line + offset


@dataclass
class RangeMap:
    new_start: int | None
    new_end: int | None
    changed: list[int] = field(default_factory=list)  # old lines changed or deleted
    inserted_inside: bool = False  # lines added between two cited lines

    @property
    def ok(self) -> bool:
        return not self.changed and not self.inserted_inside and self.new_start is not None


def map_range(hunks: list[Hunk] | None, start: int, end: int) -> RangeMap:
    if hunks is None:  # binary: nothing can be carried
        return RangeMap(None, None, list(range(start, end + 1)))
    changed = [n for n in range(start, end + 1) if map_line(hunks, n) is None]
    inserted = any(oc == 0 and start <= os_ < end for os_, oc, _, _ in hunks)
    return RangeMap(map_line(hunks, start), map_line(hunks, end), changed, inserted)


def hunk_window(hunks: list[Hunk] | None, start: int, end: int) -> tuple[int, int] | None:
    """Approximate new-side window for a changed old range: from the mapped line before it to the one after it."""
    if hunks is None:
        return None
    before = next((map_line(hunks, n) for n in range(start - 1, 0, -1) if map_line(hunks, n) is not None), 0)
    probe = end + 1
    after = None
    for _ in range(100000):
        after = map_line(hunks, probe)
        if after is not None:
            break
        probe += 1
    return before + 1, (after - 1 if after else before + 1)


def format_ref(prefix: str, path: str, start: int, end: int, as_range: bool) -> str:
    return f"{prefix}{path}:{start}" if not as_range else f"{prefix}{path}:{start}-{end}"


# ---------------------------------------------------------------------------
# git access for one engine
# ---------------------------------------------------------------------------

def git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=off", *args], check=check,
                          capture_output=True, text=True).stdout


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


@dataclass
class FileStatus:
    status: str  # M (modified or type change), R (renamed), D (deleted)
    new_path: str | None
    similarity: int | None = None
    ambiguous: bool = False
    copies: list[str] = field(default_factory=list)


def parse_name_status(z_output: str) -> dict[str, FileStatus]:
    """Old path -> status from ``git diff --name-status -z -M -C``. Unlisted old paths are unchanged."""
    tokens = z_output.split("\0")
    out: dict[str, FileStatus] = {}
    copies: dict[str, list[str]] = defaultdict(list)
    i = 0
    while i < len(tokens) and tokens[i]:
        code = tokens[i]
        kind = code[0]
        if kind in "RC":
            src, dst = tokens[i + 1], tokens[i + 2]
            i += 3
            if kind == "R":
                out[src] = FileStatus("R", dst, int(code[1:]) if code[1:].isdigit() else None)
            else:
                copies[src].append(dst)
            continue
        path = tokens[i + 1]
        i += 2
        if kind == "D":
            out[path] = FileStatus("D", None)
        elif kind in "MT":
            out[path] = FileStatus("M", path)
    for src, dsts in copies.items():
        st = out.get(src)
        if st is not None:
            st.copies = dsts
            # The old path's content went to two places and the old path is gone: git chose one.
            st.ambiguous = st.status == "R"
    return out


@dataclass
class RefMap:
    old_path: str
    start: int
    end: int
    new_path: str | None
    new_start: int | None
    new_end: int | None
    problem: str = ""  # "", file-deleted, rename-ambiguous, lines-changed, insertion-inside-range, text-mismatch
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem

    @property
    def moved(self) -> bool:
        return self.ok and (self.new_path, self.new_start, self.new_end) != (self.old_path, self.start, self.end)


class EngineDiff:
    """Diff between two commits of one repository, with lazily computed per-file hunks and blobs."""

    def __init__(self, repo: Path, old: str, new: str):
        self.repo, self.old, self.new = repo, old, new
        proc = subprocess.run(["git", "-C", str(repo), "diff", "--name-status", "-z", "-M", "-C", "-l0", old, new],
                              check=True, capture_output=True, text=True)
        if "rename detection was skipped" in proc.stderr:
            raise RuntimeError(f"{repo}: git skipped inexact rename detection: {proc.stderr.strip()}")
        self.statuses = parse_name_status(proc.stdout)
        self._hunks: dict[str, list[Hunk] | None] = {}
        self._lines: dict[tuple[str, str], list[str] | None] = {}
        self._old_tree: list[str] | None = None

    def status(self, path: str) -> FileStatus | None:
        return self.statuses.get(path)

    def hunks(self, path: str) -> list[Hunk] | None:
        st = self.status(path)
        if st is None:
            return []
        if path not in self._hunks:
            out = git(self.repo, "diff", "-U0", "--no-color", "--no-ext-diff", "--no-textconv", "--diff-algorithm=myers",
                      f"{self.old}:{path}", f"{self.new}:{st.new_path}")
            self._hunks[path] = parse_hunks(out)
        return self._hunks[path]

    def lines(self, sha: str, path: str) -> list[str] | None:
        key = (sha, path)
        if key not in self._lines:
            try:
                self._lines[key] = git_bytes(self.repo, "show", f"{sha}:{path}").decode(errors="replace").split("\n")
            except subprocess.CalledProcessError:
                self._lines[key] = None
        return self._lines[key]

    def old_tree(self) -> list[str]:
        if self._old_tree is None:
            self._old_tree = git(self.repo, "ls-tree", "-r", "--name-only", self.old).splitlines()
        return self._old_tree

    def map_ref(self, path: str, start: int, end: int, verify: bool = True) -> RefMap:
        st = self.status(path)
        if st is not None and st.status == "D":
            return RefMap(path, start, end, None, None, None, "file-deleted", f"{path} is deleted at {self.new[:12]}")
        if st is not None and st.ambiguous:
            return RefMap(path, start, end, None, None, None, "rename-ambiguous",
                          f"{path} renamed to {st.new_path} and also copied to {', '.join(st.copies)}")
        new_path = st.new_path if st is not None else path
        rm = map_range(self.hunks(path), start, end)
        if rm.changed:
            return RefMap(path, start, end, new_path, None, None, "lines-changed",
                          f"old lines {_compact(rm.changed)} changed or deleted")
        if rm.inserted_inside:
            return RefMap(path, start, end, new_path, None, None, "insertion-inside-range",
                          "lines were inserted between cited lines")
        ref = RefMap(path, start, end, new_path, rm.new_start, rm.new_end)
        if verify:
            old_lines, new_lines = self.lines(self.old, path), self.lines(self.new, new_path)
            if old_lines is None or new_lines is None or rm.new_end > len(new_lines) or end > len(old_lines) \
                    or old_lines[start - 1:end] != new_lines[rm.new_start - 1:rm.new_end]:
                ref.problem, ref.detail = "text-mismatch", "source text of the mapped range differs"
        return ref

    def invariant(self, path: str, start: int, end: int) -> bool:
        """True when lines start..end keep their numbers and text (a prose line number stays correct)."""
        st = self.status(path)
        if st is None:
            return True
        if st.status != "M":
            return False
        rm = map_range(self.hunks(path), start, end)
        return rm.ok and rm.new_start == start and rm.new_end == end

    def relevant_diff(self, path: str, ranges: list[tuple[int, int]], context: int = 3, max_hunk_lines: int = 300) -> str:
        st = self.status(path)
        if st is None:
            return ""
        if st.status == "D":
            old = self.lines(self.old, path) or []
            out = [f"--- a/{path}", "+++ /dev/null"]
            for a, b in sorted(set(ranges)):
                lo, hi = max(1, a - context), min(len(old), b + context)
                out.append(f"@@ -{lo},{hi - lo + 1} +0,0 @@ (file deleted; cited lines {a}-{b} and context shown)")
                out.extend("-" + s for s in old[lo - 1:hi])
            return "\n".join(out) + "\n"
        text = git(self.repo, "diff", f"-U{context}", "--no-color", "--no-ext-diff", "--no-textconv", "--diff-algorithm=myers",
                   f"{self.old}:{path}", f"{self.new}:{st.new_path}")
        hunks: list[list[str]] = []
        for line in text.splitlines():
            if _HUNK.match(line):
                hunks.append([line])
            elif hunks:
                hunks[-1].append(line)
        keep = []
        for h in hunks:
            m = _HUNK.match(h[0])
            os_, oc = int(m[1]), 1 if m[2] is None else int(m[2])
            lo, hi = os_, os_ + max(oc, 1) - 1
            if any(lo <= b + context and a - context <= hi for a, b in ranges):
                if len(h) > max_hunk_lines + 1:
                    h = h[:max_hunk_lines + 1] + [f"... ({len(h) - 1 - max_hunk_lines} more lines of this hunk omitted)"]
                keep.extend(h)
        if not keep:
            return ""
        header = [f"--- a/{path}", f"+++ b/{st.new_path}"]
        if st.status == "R":
            header.insert(0, f"rename {path} -> {st.new_path} (similarity {st.similarity}%)")
        return "\n".join(header + keep) + "\n"


def _compact(nums: list[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


# ---------------------------------------------------------------------------
# inventory rows
# ---------------------------------------------------------------------------

@dataclass
class Loc:
    field: str  # range, evidence[i].ref, path.entry_points[i], gate.read_at
    engine: str
    path: str
    start: int
    end: int
    prefix: str  # "" or "repos/<engine>/"
    as_range: bool


def parse_ref(ref: str, own_engine: str, fld: str) -> Loc | None:
    m = _REF.match(ref)
    if not m:
        return None
    path, prefix, engine = m["file"], "", own_engine
    if path.startswith("repos/") and path.count("/") >= 2:
        engine, path = path[len("repos/"):].split("/", 1)
        prefix = f"repos/{engine}/"
    start = int(m["start"])
    return Loc(fld, engine, path, start, int(m["end"] or start), prefix, m["end"] is not None)


def locations(row: dict) -> list[Loc]:
    eng = row["engine"]
    locs = [Loc("range", eng, row["file"], row["line_start"], row["line_end"], "", True)]
    for i, ev in enumerate(row.get("evidence", [])):
        loc = parse_ref(ev.get("ref", ""), eng, f"evidence[{i}].ref")
        if loc:
            locs.append(loc)
    for i, ep in enumerate((row.get("path") or {}).get("entry_points", [])):
        loc = parse_ref(ep, eng, f"path.entry_points[{i}]")
        if loc:
            locs.append(loc)
    gate = row.get("gate") or {}
    if gate and gate.get("scope") != "external_runtime":
        loc = parse_ref(gate.get("read_at", ""), eng, "gate.read_at")
        if loc:
            locs.append(loc)
    return locs


def set_location(row: dict, loc: Loc, new_path: str, start: int, end: int) -> None:
    if loc.field == "range":
        row["file"], row["line_start"], row["line_end"] = new_path, start, end
        return
    ref = format_ref(loc.prefix, new_path, start, end, loc.as_range)
    if loc.field.startswith("evidence["):
        row["evidence"][int(loc.field[9:loc.field.index("]")])]["ref"] = ref
    elif loc.field.startswith("path.entry_points["):
        row["path"]["entry_points"][int(loc.field[18:loc.field.index("]")])] = ref
    elif loc.field == "gate.read_at":
        row["gate"]["read_at"] = ref


def prose_texts(row: dict) -> list[tuple[str, str]]:
    gate, down, opaque, path = row.get("gate") or {}, row.get("downstream") or {}, row.get("opaque_target") or {}, row.get("path") or {}
    out = [("function", row.get("function", "")), ("notes", row.get("notes", "")), ("path.description", path.get("description", ""))]
    out += [(f"evidence[{i}].claim", ev.get("claim", "")) for i, ev in enumerate(row.get("evidence", []))]
    out += [("gate.flag", gate.get("flag", "")), ("gate.note", gate.get("note", "")), ("gate.default", gate.get("default", "")),
            ("downstream.argument", down.get("argument", "")), ("opaque_target.note", opaque.get("note", "")),
            ("default_path_condition", row.get("default_path_condition", ""))]
    return [(k, v) for k, v in out if v]


def _ranges(nums: str) -> list[tuple[int, int]]:
    out = []
    for part in re.split(r"\s*,\s*|\s+and\s+", nums.strip()):
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if m:
            a, b = int(m[1]), int(m[2] or m[1])
            if 1 <= a <= b:
                out.append((a, b))
    return out


def prose_refs(text: str) -> tuple[list[tuple[str, str, list[tuple[int, int]]]], list[tuple[str, list[tuple[int, int]]]]]:
    """(file references [(matched text, file, ranges)], bare line numbers [(matched text, ranges)]) in prose."""
    files, bare = [], []
    masked = text
    for m in PROSE_FILE_REF.finditer(text):
        files.append((m[0], m["file"], _ranges(m["nums"])))
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    seen = set()
    for rx in (PROSE_LINE_WORD, PROSE_PAREN):
        for m in rx.finditer(masked):
            if m.span("nums") in seen:
                continue
            seen.add(m.span("nums"))
            bare.append((m[0].strip(), _ranges(m["nums"])))
    return files, bare


def resolve_prose_file(ref_path: str, tree: list[str], engine: str) -> list[str]:
    """Paths in one engine's old tree that a prose file reference can name (exact, or a suffix on a path boundary)."""
    if ref_path.startswith("repos/"):
        parts = ref_path.split("/", 2)
        if len(parts) < 3 or parts[1] != engine:
            return []
        ref_path = parts[2]
    return [p for p in tree if p == ref_path or p.endswith("/" + ref_path)]


@dataclass
class Issue:
    code: str
    field: str
    detail: str


@dataclass
class RowResult:
    row_id: str
    engine: str
    new_row: dict | None
    issues: list[Issue]
    refs: list[dict]
    touched: bool  # the row cites the moved engine (structured or prose)


def candidate_key(c: dict) -> tuple:
    return c["file"], c["line"], c["col"], c["pattern"], c.get("match", "")


def keyed(cands: list[dict]) -> dict[tuple, dict]:
    """Candidates by location, pattern and match text, with an occurrence index for the rare repeats
    (one Python string holding several PTX reductions); the scanner's stable sort fixes that order."""
    seen: Counter = Counter()
    out = {}
    for c in cands:
        k = candidate_key(c)
        out[(*k, seen[k])] = c
        seen[k] += 1
    return out


@dataclass
class CandidateMap:
    old_to_new: dict[str, str | None]
    old_by_id: dict[str, dict]
    new_by_id: dict[str, dict]
    new_only: list[dict]
    removed: list[dict]
    context_changes: dict[str, list[str]]  # old id -> changed fields


def map_candidates(old: list[dict], new: list[dict], diff: EngineDiff) -> CandidateMap:
    new_by_key = keyed(new)
    old_to_new: dict[str, str | None] = {}
    matched_new: set[str] = set()
    for (_, _, col, pattern, match, n), c in keyed(old).items():
        ref = diff.map_ref(c["file"], c["line"], c["line"], verify=True)
        hit = new_by_key.get((ref.new_path, ref.new_start, col, pattern, match, n)) if ref.ok else None
        old_to_new[c["id"]] = hit["id"] if hit else None
        if hit:
            matched_new.add(hit["id"])
    new_by_id = {c["id"]: c for c in new}
    old_by_id = {c["id"]: c for c in old}
    changes: dict[str, list[str]] = {}
    for oid, nid in old_to_new.items():
        if nid is None:
            continue
        o, n = old_by_id[oid], new_by_id[nid]
        diffs = [f for f in CONTEXT_FIELDS if o.get(f) != n.get(f)]
        if (o.get("wrapper_of") or None) != (n.get("wrapper_of") or None):
            if old_to_new.get(o.get("wrapper_of") or "") != (n.get("wrapper_of") or None):
                diffs.append("wrapper_of")
        if diffs:
            changes[oid] = diffs
    return CandidateMap(old_to_new, old_by_id, new_by_id,
                        [c for c in new if c["id"] not in matched_new],
                        [c for c in old if old_to_new[c["id"]] is None], changes)


def migrate_row(row: dict, engine: str, diff: EngineDiff, new_sha: str, cmap: CandidateMap | None = None,
                own_tree: list[str] | None = None) -> RowResult:
    """Carry one row across ``engine``'s move. The new row is returned only when nothing needs review.

    ``own_tree`` lists the files of the row's own repository at its pin when that is not the moved one: a
    prose file reference that resolves there is taken to name that repository's file.
    """
    new = copy.deepcopy(row)
    issues: list[Issue] = []
    refs: list[dict] = []
    own = row["engine"] == engine
    cited: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for loc in locations(row):
        if loc.engine != engine:
            continue
        cited[loc.path].append((loc.start, loc.end))
        rm = diff.map_ref(loc.path, loc.start, loc.end)
        entry = {"field": loc.field, "old": format_ref(loc.prefix, loc.path, loc.start, loc.end, loc.as_range or loc.field == "range"),
                 "status": rm.problem or ("shifted" if rm.moved else "unchanged")}
        if rm.ok:
            entry["new"] = format_ref(loc.prefix, rm.new_path, rm.new_start, rm.new_end, loc.as_range or loc.field == "range")
            set_location(new, loc, rm.new_path, rm.new_start, rm.new_end)
        else:
            entry["detail"] = rm.detail
            if rm.new_path and rm.problem in ("lines-changed", "insertion-inside-range"):
                win = hunk_window(diff.hunks(loc.path), loc.start, loc.end)
                if win:
                    entry["new_window"] = f"{rm.new_path}:{win[0]}-{win[1]}"
            issues.append(Issue(rm.problem, loc.field, f"{entry['old']}: {rm.detail}"))
        refs.append(entry)
    touched = own or bool(cited)

    if own:
        new["sha"] = new_sha
        if cmap is not None and row.get("candidate_ids"):
            new_ids = []
            for cid in row["candidate_ids"]:
                nid = cmap.old_to_new.get(cid)
                if nid is None:
                    issues.append(Issue("candidate-unmatched", "candidate_ids",
                                        f"{cid} has no candidate at its mapped location with the same pattern at {new_sha[:12]}"))
                    continue
                if cid in cmap.context_changes:
                    issues.append(Issue("candidate-context-changed", "candidate_ids",
                                        f"{cid} -> {nid}: scanner context changed in {', '.join(cmap.context_changes[cid])}"))
                new_ids.append(nid)
            new["candidate_ids"] = new_ids
        if cmap is not None and not issues:
            ranges = [(new["file"], new["line_start"], new["line_end"])]
            for ev in new.get("evidence", []):
                m = _REF.match(ev["ref"])
                if m and not m["file"].startswith("repos/"):
                    ranges.append((m["file"], int(m["start"]), int(m["end"] or m["start"])))
            for c in cmap.new_only:
                if any(c["file"] == f and a <= c["line"] <= b for f, a, b in ranges):
                    issues.append(Issue("new-candidate-in-cited-range", "candidate_ids",
                                        f"new candidate {c['id']} ({c['pattern']}, {c['class_hint']}) at {c['file']}:{c['line']} has no old counterpart"))

    # Prose line numbers: never rewritten, so they must keep their numbers and text.
    tree = diff.old_tree()
    for fld, text in prose_texts(row):
        files, bare = prose_refs(text)
        for matched, ref_path, ranges in files:
            if own_tree is not None and resolve_prose_file(ref_path, own_tree, row["engine"]):
                continue
            # A file too short at the pinned sha for the cited lines cannot be the one the prose names.
            paths = [p for p in resolve_prose_file(ref_path, tree, engine)
                     if all(b <= len(diff.lines(diff.old, p) or []) for _, b in ranges)]
            if not paths:
                continue
            touched = True
            bad = [(p, a, b) for p in paths for a, b in ranges if not diff.invariant(p, a, b)]
            if bad:
                hints = []
                for p, a, b in bad:
                    rm = diff.map_ref(p, a, b)
                    hints.append(f"{p}:{a}-{b} -> " + (f"{rm.new_path}:{rm.new_start}-{rm.new_end} (text unchanged)" if rm.ok else rm.problem))
                issues.append(Issue("prose-line-reference", fld, f"'{matched}' no longer names the same lines: " + "; ".join(hints)))
            for p in paths:
                cited[p].extend(ranges)
        if not bare:
            continue
        for matched, ranges in bare:
            bad = []
            for p in cited:
                old_lines = diff.lines(diff.old, p)
                n = len(old_lines) if old_lines is not None else 0
                bad += [f"{p}:{a}-{b}" for a, b in ranges if b <= n and not diff.invariant(p, a, b)]
            if bad:
                touched = True
                issues.append(Issue("prose-line-reference", fld,
                                    f"'{matched}' is not invariant in a cited file ({'; '.join(bad[:4])}{' ...' if len(bad) > 4 else ''})"))
    return RowResult(row["id"], row["engine"], None if issues else new, issues, refs, touched)


# ---------------------------------------------------------------------------
# scope drift
# ---------------------------------------------------------------------------

def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def scope_drift(diff: EngineDiff, entry: dict, new_tree: list[str]) -> list[dict]:
    """For each scan path and exclude of the manifest entry: files renamed out of it (by destination
    prefix), files deleted under it, and whether it still exists at the new commit. Scope is the owner's
    decision; this only reports where the files went."""
    out = []
    for kind, prefixes in (("scan_paths", entry.get("scan_paths", [])), ("exclude", entry.get("exclude", []))):
        for pre in prefixes:
            old_files = [p for p in diff.old_tree() if _under(p, pre)]
            moved: Counter = Counter()
            deleted = 0
            for p in old_files:
                st = diff.status(p)
                if st is None or st.status == "M":
                    continue
                if st.status == "D":
                    deleted += 1
                elif not _under(st.new_path, pre):
                    rest = p[len(pre.rstrip("/")):]
                    if rest and st.new_path.endswith(rest):  # same layout under a new directory
                        moved[st.new_path[:-len(rest)]] += 1
                    else:  # reorganised: group by directory at one level below the old path's depth
                        moved["/".join(st.new_path.split("/")[:pre.rstrip("/").count("/") + 2])] += 1
            exists = any(_under(p, pre) for p in new_tree)
            if moved or not exists:
                top = moved.most_common(1)
                out.append({"field": kind, "path": pre, "files_at_old": len(old_files), "renamed_out": dict(moved.most_common()),
                            "deleted": deleted, "exists_at_new": exists,
                            "dominant_destination": top[0][0] if top and top[0][1] * 2 > len(old_files) else None,
                            "major_destinations": [d for d, n in moved.most_common() if n >= 20 or n * 10 >= len(old_files)]})
    return out


def proposed_scope(entry: dict, drift: list[dict]) -> dict:
    """Mechanical proposal: a path most of whose files moved to one place is replaced by that place (kept as
    well while it still exists). Offered for the owner to accept or edit, never applied by default."""
    moved = {(d["field"], d["path"]): d for d in drift}
    scope = {}
    for kind in ("scan_paths", "exclude"):
        paths: list[str] = []
        for pre in entry.get(kind, []):
            d = moved.get((kind, pre))
            if d is None:
                paths.append(pre)
                continue
            if d["exists_at_new"] or not d["dominant_destination"]:
                paths.append(pre)
            # a scan path also follows any directory that took a large share of its files; an exclude only its main one
            for dest in (d["major_destinations"] if kind == "scan_paths" else [d["dominant_destination"]] if d["dominant_destination"] else []):
                if dest not in paths:
                    paths.append(dest)
        scope[kind] = paths
    return scope


# ---------------------------------------------------------------------------
# per-engine plan
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def default_branch(repo: Path) -> str:
    ref = git(repo, "symbolic-ref", "-q", "refs/remotes/origin/HEAD", check=False).strip()
    if ref:
        return ref.rsplit("/", 1)[-1]
    for b in ("main", "master"):
        if git(repo, "rev-parse", "-q", "--verify", f"refs/remotes/origin/{b}", check=False).strip():
            return b
    raise RuntimeError(f"{repo}: no default branch")


def commit_date_utc(repo: Path, sha: str) -> str:
    ts = int(git(repo, "log", "-1", "--format=%ct", sha).strip())
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_ancestor(repo: Path, a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", a, b], capture_output=True).returncode == 0


def ensure_history(repo: Path, old: str, branch: str) -> bool:
    """Deepen a shallow clone until the pinned sha is an ancestor of the branch, so commit counts are exact."""
    if is_ancestor(repo, old, f"origin/{branch}"):
        return False
    if not (repo / ".git" / "shallow").exists():
        return False
    since = (dt.datetime.fromtimestamp(int(git(repo, "log", "-1", "--format=%ct", old).strip()), dt.timezone.utc)
             - dt.timedelta(days=2)).strftime("%Y-%m-%d")
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", f"--shallow-since={since}", "origin", branch], check=True)
    return True


def checkout(repo: Path, sha: str, url: str, name: str, dest: Path) -> None:
    """Detached checkout of sha, fetching it through scan.clone when it is not present."""
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError(f"{repo}: working tree has local changes; refusing to switch commits")
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0:
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach", sha], check=True)
    else:
        from scan.clone import ensure
        ensure(name, url, sha, dest)


@dataclass
class EnginePlan:
    name: str
    old: str
    new: str
    old_date: str
    new_date: str
    branch: str
    commit_count: int | None
    results: list[RowResult] = field(default_factory=list)
    cmap: CandidateMap | None = None
    new_candidates_dir: Path | None = None
    diff: EngineDiff | None = None
    notes: list[str] = field(default_factory=list)
    drift: list[dict] = field(default_factory=list)
    scope: dict | None = None  # scan_paths/exclude override used for the rescan

    @property
    def moved(self) -> bool:
        return self.old != self.new

    def worklist(self) -> list[RowResult]:
        return [r for r in self.results if r.issues]

    def auto_rows(self) -> dict[str, dict]:
        return {r.row_id: r.new_row for r in self.results if r.new_row is not None and r.touched}


def choose_new(repo: Path, branch: str, cutoff: str) -> str:
    return git(repo, "rev-list", "-1", f"--before={cutoff}", f"origin/{branch}").strip()


def plan_engine(entry: dict, rows: list[dict], root: Path, cutoff: str, work: Path, keep_checkout: bool = False,
                scope: dict | None = None) -> EnginePlan:
    name, old = entry["name"], entry["sha"]
    repos_dir = root / "repos"
    repo = repos_dir / name
    branch = default_branch(repo)
    notes = []
    if ensure_history(repo, old, branch):
        notes.append("clone was shallow; history deepened to the pinned commit to count commits")
    new = choose_new(repo, branch, cutoff)
    if not new:
        raise RuntimeError(f"{name}: no commit on origin/{branch} before {cutoff}")
    count = int(git(repo, "rev-list", "--count", f"{old}..{new}").strip()) if is_ancestor(repo, old, new) else None
    if count is None and old != new:
        notes.append(f"pinned sha {old[:12]} is not an ancestor of {new[:12]}; commit count unknown")
    plan = EnginePlan(name, old, new, entry["commit_date"], commit_date_utc(repo, new), branch, count if old != new else 0, notes=notes)
    if not plan.moved:
        return plan
    diff = EngineDiff(repo, old, new)
    plan.diff = diff
    plan.drift = scope_drift(diff, entry, git(repo, "ls-tree", "-r", "--name-only", new).splitlines())
    plan.scope = scope

    # Rescan at the new sha through the Makefile flow, then put the pinned checkout back.
    cand_dir = work / "staged" / name / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "scan-manifest.json"
    if scope:
        manifest = json.loads(manifest_path.read_text())
        for e in manifest["repos"]:
            if e["name"] == name:
                e.update({k: scope[k] for k in ("scan_paths", "exclude") if k in scope})
        manifest_path = work / "staged" / name / "scan-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    head = git(repo, "rev-parse", "HEAD").strip()
    checkout(repo, new, entry["url"], name, repos_dir)
    try:
        from scan.run import main as scan_main
        rc = scan_main(["--manifest", str(manifest_path), "--repos-dir", str(repos_dir), "--only", name,
                        "--sha", new, "--out", str(cand_dir)])
        if rc:
            raise RuntimeError(f"{name}: scanner exited with {rc}")
    finally:
        if not keep_checkout:
            checkout(repo, head, entry["url"], name, repos_dir)
    plan.new_candidates_dir = cand_dir
    old_cands = load_jsonl(root / "candidates" / f"{name}.jsonl")
    stale = {c["sha"] for c in old_cands} - {old}
    if stale:
        raise RuntimeError(f"{name}: candidates/{name}.jsonl is not at the pinned sha ({stale})")
    plan.cmap = map_candidates(old_cands, load_jsonl(cand_dir / f"{name}.jsonl"), diff)
    pins = {e["name"]: e["sha"] for e in manifest_entries(root)}
    trees: dict[str, list[str]] = {}
    for row in rows:
        other = row["engine"]
        if other != name and other not in trees:
            trees[other] = git(repos_dir / other, "ls-tree", "-r", "--name-only", pins[other]).splitlines()
        plan.results.append(migrate_row(row, name, diff, new, plan.cmap if other == name else None,
                                        None if other == name else trees[other]))
    return plan


def manifest_entries(root: Path) -> list[dict]:
    return json.loads((root / "scan-manifest.json").read_text())["repos"]


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def class_a_question(c: dict) -> bool:
    """The scanner marks it as a floating-point accumulation (atomic or reduction) inside the scan paths."""
    return c.get("class_hint") == "A?" and bool(c.get("in_scope")) and not c.get("excluded_reason")


def _cand_summary(c: dict) -> dict:
    return {k: c.get(k) for k in ("id", "line", "col", "kind", "match", "class_hint", "dtype_hint", "memory_space_hint",
                                  "function", "in_scope", "excluded_reason")}


def group_candidates(cands: list[dict], extra=None) -> dict:
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in sorted(cands, key=lambda c: (c["file"], c["pattern"], c["line"], c["col"])):
        d = _cand_summary(c)
        d["class_a_question"] = class_a_question(c)
        if extra:
            d.update(extra(c))
        groups[c["file"]][c["pattern"]].append(d)
    return {f: dict(p) for f, p in groups.items()}


def worklist_entry(plan: EnginePlan, res: RowResult, row: dict) -> dict:
    diff = plan.diff
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for loc in locations(row):
        if loc.engine == plan.name:
            by_file[loc.path].append((loc.start, loc.end))
    for issue in res.issues:
        if issue.code == "prose-line-reference":
            for m in re.finditer(r"([^\s:;'()]+):(\d+)-(\d+)", issue.detail):
                by_file[m[1]].append((int(m[2]), int(m[3])))
    diffs = "".join(diff.relevant_diff(p, rs) for p, rs in sorted(by_file.items()) if diff.status(p) is not None)
    hints = {}
    if plan.cmap:
        for cid in row.get("candidate_ids", []) if row["engine"] == plan.name else []:
            if plan.cmap.old_to_new.get(cid) is None and cid in plan.cmap.old_by_id:
                o = plan.cmap.old_by_id[cid]
                st = diff.status(o["file"])
                path = st.new_path if st and st.new_path else o["file"]
                hints[cid] = [{"id": c["id"], "file": c["file"], "line": c["line"], "function": c["function"]}
                              for c in plan.cmap.new_only if c["file"] == path and c["pattern"] == o["pattern"]
                              and c["function"] == o["function"]]
    rng = next((r for r in res.refs if r["field"] == "range"), None)
    new_loc = {"sha": plan.new}
    if rng:
        new_loc["location"] = rng.get("new") or rng.get("new_window") or None
        new_loc["exact"] = "new" in rng
    return {
        "id": res.row_id, "engine": res.engine,
        "reasons": sorted({i.code for i in res.issues}),
        "issues": [asdict(i) for i in res.issues],
        "old_location": {"sha": row["sha"], "file": row["file"], "line_start": row["line_start"], "line_end": row["line_end"]},
        "new_location": new_loc if row["engine"] == plan.name else {"note": f"row stays at {row['sha'][:12]}; only its references into {plan.name} move"},
        "references": res.refs,
        "possible_candidate_counterparts": hints,
        "diff": diffs,
        "old_row": row,
    }


def write_plan_outputs(plan: EnginePlan, rows_by_id: dict[str, dict], out: Path) -> dict:
    for sub in ("worklist", "new_candidates", "removed_candidates"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    work = plan.worklist()
    (out / "worklist" / f"{plan.name}.json").write_text(json.dumps({
        "engine": plan.name, "old_sha": plan.old, "new_sha": plan.new, "rows": len(work),
        "entries": [worklist_entry(plan, r, rows_by_id[r.row_id]) for r in work]}, indent=2, ensure_ascii=False) + "\n")
    stats = {"auto": 0, "auto_shifted": 0, "worklist_by_reason": Counter(), "cross_engine_rows_rewritten": 0}
    for r in plan.results:
        if r.engine != plan.name:
            if r.new_row is not None and r.new_row != rows_by_id[r.row_id]:
                stats["cross_engine_rows_rewritten"] += 1
            continue
        if r.new_row is not None:
            stats["auto"] += 1
            stats["auto_shifted"] += any(x["status"] == "shifted" for x in r.refs)
            stats["auto_gated_or_default"] = stats.get("auto_gated_or_default", 0) + bool(r.new_row.get("gate") or r.new_row.get("default_path") is True)
    for r in work:
        for code in {i.code for i in r.issues}:
            stats["worklist_by_reason"][code] += 1
    cm = plan.cmap
    linked = {cid for row in rows_by_id.values() if row["engine"] == plan.name for cid in row.get("candidate_ids", [])}
    if cm is not None:
        inv = defaultdict(list)
        for row in rows_by_id.values():
            if row["engine"] == plan.name:
                for cid in row.get("candidate_ids", []):
                    inv[cid].append(row["id"])
        new_only, removed = cm.new_only, cm.removed
        (out / "new_candidates" / f"{plan.name}.json").write_text(json.dumps({
            "engine": plan.name, "new_sha": plan.new, "count": len(new_only),
            "eligible": sum(1 for c in new_only if c.get("in_scope") and not c.get("excluded_reason")),
            "class_a_question": sum(map(class_a_question, new_only)),
            "note": "class_a_question: the scanner hints a floating-point accumulation (class_hint A?) at an in-scope, non-excluded site",
            "class_a_question_sites": [{k: c.get(k) for k in ("id", "file", "line", "pattern", "match", "dtype_hint", "function")}
                                       for c in new_only if class_a_question(c)],
            "files": group_candidates(new_only)}, indent=2, ensure_ascii=False) + "\n")
        (out / "removed_candidates" / f"{plan.name}.json").write_text(json.dumps({
            "engine": plan.name, "old_sha": plan.old, "count": len(removed),
            "linked": sum(1 for c in removed if c["id"] in linked),
            "files": group_candidates(removed, lambda c: {"inventory_ids": sorted(inv.get(c["id"], []))})},
            indent=2, ensure_ascii=False) + "\n")
        stats.update(candidates_old=len(cm.old_by_id), candidates_new=len(cm.new_by_id),
                     candidates_matched=sum(v is not None for v in cm.old_to_new.values()),
                     linked_old=len(linked), linked_carried=sum(1 for c in linked if cm.old_to_new.get(c)),
                     new_only=len(new_only), new_only_class_a=sum(map(class_a_question, new_only)),
                     removed=len(removed), removed_linked=sum(1 for c in removed if c["id"] in linked),
                     context_changed=len(cm.context_changes),
                     eligible_old=sum(1 for c in cm.old_by_id.values() if c.get("in_scope") and not c.get("excluded_reason")),
                     eligible_new=sum(1 for c in cm.new_by_id.values() if c.get("in_scope") and not c.get("excluded_reason")))
    staged = out / "staged" / plan.name
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "auto_rows.jsonl").write_text(dump_jsonl(list(plan.auto_rows().values())))
    (staged / "worklist_rows_old_sha.jsonl").write_text(dump_jsonl([rows_by_id[r.row_id] for r in work]))
    return stats


def summary_markdown(plans: list[EnginePlan], stats: dict[str, dict], cutoff: str, rows_by_engine: Counter) -> str:
    out = [f"# Repin plan: last commit before {cutoff}", "",
           "Generated by `python -m scan.repin plan`. Rows are carried only when every cited line is unchanged and its text is equal at both commits; everything else is on the worklist. Candidate links are carried by location and pattern.", "",
           "| Repository | Old sha (date) | New sha (date) | Commits | Rows | Auto-carried (shifted) | Worklist | Candidates old/new | Linked carried | New | New class A? | Removed (linked) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in plans:
        s = stats.get(p.name, {})
        rows = rows_by_engine[p.name]
        if not p.moved:
            out.append(f"| {p.name} | `{p.old[:12]}` ({p.old_date}) | unchanged | 0 | {rows} | {rows} (0) | 0 | n/a | n/a | 0 | 0 | 0 |")
            continue
        wl = len(p.worklist())
        out.append(f"| {p.name} | `{p.old[:12]}` ({p.old_date}) | `{p.new[:12]}` ({p.new_date}) | {p.commit_count if p.commit_count is not None else 'unknown'} | {rows} | "
                   f"{s['auto']} ({s['auto_shifted']}) | {wl} | {s['candidates_old']}/{s['candidates_new']} | {s['linked_carried']}/{s['linked_old']} | "
                   f"{s['new_only']} | {s['new_only_class_a']} | {s['removed']} ({s['removed_linked']}) |")
    out += ["", "## Worklist by reason", "",
            "A row can have several reasons; counts are rows. Rows of other repositories appear when they cite the moved repository.", ""]
    reasons = sorted({k for s in stats.values() for k in s.get("worklist_by_reason", {})})
    out.append("| Repository | Worklist rows | Other-repository rows | " + " | ".join(reasons) + " |")
    out.append("|---|---|---|" + "---|" * len(reasons))
    for p in plans:
        if not p.moved:
            continue
        s = stats[p.name]
        other = sum(1 for r in p.worklist() if r.engine != p.name)
        out.append(f"| {p.name} | {len(p.worklist())} | {other} | " + " | ".join(str(s['worklist_by_reason'].get(k, 0)) for k in reasons) + " |")
    out += ["", "## Per repository", ""]
    for p in plans:
        s = stats.get(p.name, {})
        out.append(f"### {p.name}")
        out.append("")
        out.append(f"- Default branch `{p.branch}`; old `{p.old}` ({p.old_date}); new `{p.new}` ({p.new_date}).")
        if not p.moved:
            out.append("- No commit after the pin before the cut-off; the sha is kept.")
            out.append("")
            continue
        out.append(f"- Commits between the pins: {p.commit_count if p.commit_count is not None else 'unknown'}; files changed: "
                   f"{sum(1 for st in p.diff.statuses.values() if st.status == 'M')} modified, "
                   f"{sum(1 for st in p.diff.statuses.values() if st.status == 'R')} renamed, "
                   f"{sum(1 for st in p.diff.statuses.values() if st.status == 'D')} deleted.")
        out.append(f"- Rows: {rows_by_engine[p.name]}; carried and text-verified: {s['auto']} ({s['auto_shifted']} with shifted lines); on the worklist: {len(p.worklist())}"
                   f" ({sum(1 for r in p.worklist() if r.engine != p.name)} from other repositories); other repositories' rows rewritten for references into {p.name}: {s['cross_engine_rows_rewritten']}.")
        out.append(f"- Candidates: {s['candidates_old']} old, {s['candidates_new']} new, {s['candidates_matched']} matched by location and pattern; "
                   f"linked candidates carried {s['linked_carried']} of {s['linked_old']}; {s['context_changed']} matched candidates changed scanner context.")
        out.append(f"- Eligible candidates (in scope, not excluded): {s['eligible_old']} old, {s['eligible_new']} new. New without an old counterpart: {s['new_only']} "
                   f"({s['new_only_class_a']} class A?); removed: {s['removed']} ({s['removed_linked']} linked to rows).")
        out.append(f"- Carried rows with a gate or default_path true, whose claims also rest on code they do not cite: {s.get('auto_gated_or_default', 0)}.")
        if p.scope:
            out.append(f"- Rescanned with a scope override: `{json.dumps(p.scope)}`.")
        for d in p.drift:
            dest = ", ".join(f"`{k}` ({v})" for k, v in d["renamed_out"].items()) or "none"
            out.append(f"- Scope drift: {d['field']} `{d['path']}` ({d['files_at_old']} files at the old commit): renamed out to {dest}; "
                       f"{d['deleted']} deleted; {'still exists' if d['exists_at_new'] else 'no longer exists'} at the new commit.")
        for n in p.notes:
            out.append(f"- Note: {n}.")
        out.append("")
    out += ["## Limitations", "", *[f"- {x}" for x in LIMITATIONS], ""]
    return "\n".join(out) + "\n"


LIMITATIONS = [
    "Verification is textual and covers the cited lines only. A carried row's reachability, default-path, gate-default and "
    "dispatch claims can depend on code it does not cite (callers, defaults, selection logic); those were not re-read.",
    "Prose is never rewritten. Line numbers in prose are found by pattern (`file:N`, `line(s) N`, `at N`, parenthesised number "
    "lists); other phrasings are not detected. A bare number is checked against every file the row cites or names, and a "
    "partial path is checked against every file it suffix-matches, so some flags are needless (for example a number that is a "
    "shape, or a reference to another release such as `q_gemm.cu:319-320 in vLLM 0.11.2`).",
    "A prose file reference in another repository's row is tied to the moved repository only when it does not resolve in the "
    "row's own tree; a file that resolves nowhere is ignored.",
    "Git's Myers diff decides which lines are unchanged. Moved blocks appear as deletion plus insertion and go to review. "
    "Rename detection uses git's 50% similarity threshold, so a file moved and heavily edited appears as deleted.",
    "An ambiguous rename is detected only as a renamed source that git also reports as copied (`-C`, modified files as copy "
    "sources); a split that git reports as rename plus new file is not flagged beyond its changed lines.",
    "Candidate context comparison ignores snippet, notes and function length; any other scanner field that differs on an "
    "unchanged line sends the linked row to review.",
    "The plan keeps each manifest's scan_paths and exclude unless `--scope` is given; scope drift is reported, not decided.",
    "Outside the inventory the tool changes nothing: the README pinning sentence, docs/SHAPE_HYPOTHESIS.md (which names the "
    "vLLM, FlashInfer and SGLang pins and cites line numbers at them) and the note in triage/attach_runtime.py that the "
    "moe_wna16 kernel is byte-identical between the vLLM 0.28.0 wheel and the pinned sha need updating by hand when those "
    "repositories move. Runtime evidence is attached by row id and moves with its row.",
]


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply_engine(plan: EnginePlan, root: Path, manifest: dict, resolutions: dict[str, dict], generated: str) -> list[str]:
    """Write one engine's migration into the tree. Returns the paths written."""
    missing = [r.row_id for r in plan.worklist() if r.row_id not in resolutions]
    if missing:
        raise SystemExit(f"{plan.name}: {len(missing)} worklist rows have no reviewed replacement: {', '.join(missing[:10])}")
    extra = set(resolutions) - {r.row_id for r in plan.worklist()}
    if extra:
        raise SystemExit(f"{plan.name}: resolutions for rows not on the worklist: {', '.join(sorted(extra))}")
    for rid, row in resolutions.items():
        if row["engine"] == plan.name and row.get("sha") != plan.new:
            raise SystemExit(f"{rid}: reviewed replacement is not at {plan.new}")
    replace = {**plan.auto_rows(), **resolutions}
    written = []
    for path in sorted((root / "inventory").glob("*.jsonl")):
        rows = load_jsonl(path)
        new_rows = [replace.get(r["id"], r) for r in rows]
        if new_rows != rows:
            path.write_text(dump_jsonl(new_rows), encoding="utf-8")
            written.append(str(path.relative_to(root)))
    for suffix in (".jsonl", ".coverage.json"):
        src = plan.new_candidates_dir / f"{plan.name}{suffix}"
        dst = root / "candidates" / f"{plan.name}{suffix}"
        dst.write_bytes(src.read_bytes())
        written.append(str(dst.relative_to(root)))
    for entry in manifest["repos"]:
        if entry["name"] == plan.name:
            entry["sha"], entry["commit_date"] = plan.new, plan.new_date
            if plan.scope:
                entry.update({k: plan.scope[k] for k in ("scan_paths", "exclude") if k in plan.scope})
    manifest["generated"] = generated
    (root / "scan-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    written.append("scan-manifest.json")
    from triage.dispositions import records
    rows = load_jsonl(root / "inventory" / f"{plan.name}.jsonl")
    ledger = records(rows, load_jsonl(root / "candidates" / f"{plan.name}.jsonl"))
    (root / "triage" / "dispositions" / f"{plan.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    written.append(f"triage/dispositions/{plan.name}.jsonl")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "apply"])
    ap.add_argument("--only", action="append", help="engine name (repeatable); apply takes exactly one")
    ap.add_argument("--before", default=DEFAULT_CUTOFF, help="cut-off for the new commit (git --before)")
    ap.add_argument("--out", type=Path, help="plan: output directory for the worklist, candidate lists and summary")
    ap.add_argument("--resolutions", type=Path, help="apply: JSONL of reviewed rows replacing every worklist row")
    ap.add_argument("--scope", type=Path, help='JSON {"<engine>": {"scan_paths": [...], "exclude": [...]}} to rescan (and, for apply, '
                                              "record in the manifest) with a different scope")
    ap.add_argument("--generated", default=dt.date.today().isoformat(), help="apply: manifest 'generated' date")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    manifest = json.loads((root / "scan-manifest.json").read_text())
    rows: list[dict] = []
    for path in sorted((root / "inventory").glob("*.jsonl")):
        rows += load_jsonl(path)
    rows_by_id = {r["id"]: r for r in rows}
    entries = [e for e in manifest["repos"] if not args.only or e["name"] in args.only]
    scopes = json.loads(args.scope.read_text()) if args.scope else {}

    if args.command == "plan":
        if not args.out:
            ap.error("plan needs --out")
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        plans, stats = [], {}
        for entry in entries:
            plan = plan_engine(entry, rows, root, args.before, out, scope=scopes.get(entry["name"]))
            plans.append(plan)
            if plan.moved:
                stats[plan.name] = write_plan_outputs(plan, rows_by_id, out)
            print(f"{plan.name}: {plan.old[:12]} -> {plan.new[:12]}; worklist {len(plan.worklist())}")
        (out / "summary.md").write_text(summary_markdown(plans, stats, args.before, Counter(r["engine"] for r in rows)))
        (out / "summary.json").write_text(json.dumps({p.name: {"old": p.old, "new": p.new, "old_date": p.old_date, "new_date": p.new_date,
                                                              "commits": p.commit_count, "worklist": len(p.worklist()),
                                                              "scope_drift": p.drift, "scope_used": p.scope,
                                                              "proposed_scope": proposed_scope(next(e for e in entries if e["name"] == p.name), p.drift) if p.drift else None,
                                                              **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.get(p.name, {}).items()}}
                                                      for p in plans}, indent=2) + "\n")
        return 0

    if not args.only or len(args.only) != 1:
        ap.error("apply takes exactly one --only")
    resolutions = {r["id"]: r for r in load_jsonl(args.resolutions)} if args.resolutions else {}
    with __import__("tempfile").TemporaryDirectory(prefix="repin-") as tmp:
        plan = plan_engine(entries[0], rows, root, args.before, Path(tmp), keep_checkout=True,
                           scope=scopes.get(entries[0]["name"]))
        if not plan.moved:
            print(f"{plan.name}: no commit after {plan.old[:12]} before {args.before}; nothing to apply")
            return 0
        try:
            written = apply_engine(plan, root, manifest, resolutions, args.generated)
        except BaseException:
            checkout(root / "repos" / plan.name, plan.old, entries[0]["url"], plan.name, root / "repos")
            raise
    print(f"{plan.name}: {plan.old[:12]} -> {plan.new[:12]}; wrote {', '.join(written)}")
    print("next: make docs && make validate && make check-derived && make test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
