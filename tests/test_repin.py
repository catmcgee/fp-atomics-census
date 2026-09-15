"""Tests for scan/repin.py: line mapping from zero-context hunks, reference rewriting, candidate
matching by location and pattern, and the review conditions, on a temporary git repository."""
import subprocess
from pathlib import Path

import pytest

from scan.repin import (EngineDiff, format_ref, hunk_window, map_candidates, map_line, map_range, migrate_row,
                        parse_hunks, parse_name_status, parse_ref, prose_refs, resolve_prose_file)


def test_map_line_through_insertion_deletion_and_change():
    # insert 2 lines after old 3; replace old 6-7 by 1 line; delete old 10
    hunks = parse_hunks("@@ -3,0 +4,2 @@\n+a\n+b\n@@ -6,2 +8 @@\n-x\n-y\n+z\n@@ -10 +11,0 @@\n-q\n")
    assert hunks == [(3, 0, 4, 2), (6, 2, 8, 1), (10, 1, 11, 0)]
    assert [map_line(hunks, n) for n in range(1, 13)] == [1, 2, 3, 6, 7, None, None, 9, 10, None, 11, 12]


def test_insertion_at_start_of_file_shifts_every_line():
    hunks = parse_hunks("@@ -0,0 +1,3 @@\n+a\n+b\n+c\n")
    assert map_line(hunks, 1) == 4


def test_map_range_requires_every_line_unchanged_and_contiguous():
    hunks = [(3, 0, 4, 2), (6, 2, 8, 1)]
    ok = map_range(hunks, 4, 5)
    assert ok.ok and (ok.new_start, ok.new_end) == (6, 7)
    assert map_range(hunks, 5, 6).changed == [6]
    inside = map_range(hunks, 2, 4)  # insertion after line 3 splits the range
    assert inside.inserted_inside and not inside.ok
    assert map_range(hunks, 1, 3).ok  # insertion right after the range does not touch it
    assert map_range(None, 1, 2).changed == [1, 2]  # binary difference


def test_hunk_window_brackets_a_changed_range():
    assert hunk_window([(3, 0, 4, 2), (6, 2, 8, 1)], 6, 7) == (8, 8)


def test_refs_keep_their_form_and_prefix():
    loc = parse_ref("repos/cutlass/include/a.h:10", "sglang", "evidence[0].ref")
    assert (loc.engine, loc.path, loc.start, loc.end, loc.prefix, loc.as_range) == ("cutlass", "include/a.h", 10, 10, "repos/cutlass/", False)
    assert format_ref(loc.prefix, "include/b.h", 12, 12, loc.as_range) == "repos/cutlass/include/b.h:12"
    assert format_ref("", "a.cu", 3, 5, True) == "a.cu:3-5"
    assert parse_ref("PyTorch runtime", "vllm", "gate.read_at") is None


def test_name_status_marks_a_renamed_and_copied_source_ambiguous():
    z = "M\0a.cu\0R095\0b.cu\0c.cu\0C080\0b.cu\0d.cu\0D\0e.cu\0C070\0a.cu\0f.cu\0A\0g.cu\0"
    st = parse_name_status(z)
    assert st["a.cu"].status == "M" and not st["a.cu"].ambiguous  # still present: copies do not matter
    assert st["b.cu"].status == "R" and st["b.cu"].new_path == "c.cu" and st["b.cu"].ambiguous
    assert st["e.cu"].status == "D"
    assert "g.cu" not in st


def test_prose_references_are_found_with_their_numbers():
    files, bare = prose_refs("see flash_api.cpp:1539-1544 and 1600 (also 12, 30-31); at 534, 973-984 on lines 5 and 9")
    assert files == [("flash_api.cpp:1539-1544 and 1600", "flash_api.cpp", [(1539, 1544), (1600, 1600)])]
    assert ("(also 12, 30-31)", [(12, 12), (30, 31)]) in bare
    assert ("at 534, 973-984", [(534, 534), (973, 984)]) in bare
    assert ("lines 5 and 9", [(5, 5), (9, 9)]) in bare
    tree = ["csrc/flash_api.cpp", "hopper/flash_api.cpp", "x/myflash_api.cpp"]
    assert resolve_prose_file("flash_api.cpp", tree, "fa") == ["csrc/flash_api.cpp", "hopper/flash_api.cpp"]
    assert resolve_prose_file("repos/fa/csrc/flash_api.cpp", tree, "fa") == ["csrc/flash_api.cpp"]
    assert resolve_prose_file("repos/other/csrc/flash_api.cpp", tree, "fa") == []


# ---------------------------------------------------------------------------
# a real repository
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str | None], msg: str) -> str:
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _body(tag: str, n: int = 40) -> str:
    return "".join(f"// {tag} line {i} with enough distinct text to keep rename similarity high\n" for i in range(1, n + 1))


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "eng"
    r.mkdir()
    _git(r, "init", "-q")
    kernel = _body("kernel").splitlines(keepends=True)
    kernel[19] = "atomicAdd(&out[i], v);\n"  # line 20
    kernel[29] = "atomicAdd(&out[j], w);\n"  # line 30
    old = _commit(r, {"k.cu": "".join(kernel), "moved.cu": _body("moved"), "gone.cu": _body("gone"),
                      "split.cu": _body("split"), "same.cu": _body("same")}, "old")
    new_kernel = ["// new header\n", "// another\n"] + kernel  # every line shifts by 2
    new_kernel[2 + 29] = "atomicAdd(&out[j], w * 2);\n"  # old line 30 changed
    new_kernel.insert(2 + 25, "// inserted between 25 and 26\n")
    split = _body("split")
    new = _commit(r, {"k.cu": "".join(new_kernel), "moved.cu": None, "renamed/moved.cu": _body("moved"),
                      "gone.cu": None, "split.cu": None, "split_a.cu": split, "split_b.cu": split}, "new")
    return r, old, new


def _row(**kw):
    row = {"id": "eng-0001", "engine": "eng", "sha": "0" * 40, "file": "k.cu", "line_start": 19, "line_end": 21,
           "function": "f", "path": {"description": "d", "entry_points": []}, "evidence": [{"ref": "k.cu:20", "claim": "c"}],
           "snippet": "", "notes": "", "provenance": "manual"}
    row.update(kw)
    return row


def test_unchanged_range_is_shifted_and_verified(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    res = migrate_row(_row(evidence=[{"ref": "k.cu:20", "claim": "c"}, {"ref": "moved.cu:3-4", "claim": "c"}]), "eng", diff, new)
    assert not res.issues
    row = res.new_row
    assert (row["sha"], row["file"], row["line_start"], row["line_end"]) == (new, "k.cu", 21, 23)
    assert [e["ref"] for e in row["evidence"]] == ["k.cu:22", "renamed/moved.cu:3-4"]


def test_changed_inserted_deleted_and_ambiguous_locations_go_to_review(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    cases = {
        "lines-changed": _row(line_start=29, line_end=31, evidence=[{"ref": "k.cu:30", "claim": "c"}]),
        "insertion-inside-range": _row(evidence=[{"ref": "k.cu:24-27", "claim": "c"}]),
        "file-deleted": _row(path={"description": "d", "entry_points": ["gone.cu:5"]}),
        "rename-ambiguous": _row(gate={"flag": "F", "read_at": "split.cu:7", "default": "0"}),
    }
    for code, row in cases.items():
        res = migrate_row(row, "eng", diff, new)
        assert res.new_row is None and code in {i.code for i in res.issues}, code
    # an external runtime gate is not a location
    external = _row(gate={"flag": "F", "read_at": "gone.cu:5", "default": "0", "scope": "external_runtime"})
    assert migrate_row(external, "eng", diff, new).new_row is not None


def test_cross_repository_reference_moves_and_the_row_keeps_its_sha(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    row = _row(id="other-0001", engine="other", sha="1" * 40, file="x.py",
               evidence=[{"ref": "x.py:1", "claim": "c"}, {"ref": "repos/eng/k.cu:20", "claim": "c"}])
    res = migrate_row(row, "eng", diff, new, own_tree=["x.py"])
    assert res.touched and res.new_row["sha"] == "1" * 40 and res.new_row["file"] == "x.py"
    assert [e["ref"] for e in res.new_row["evidence"]] == ["x.py:1", "repos/eng/k.cu:22"]


def test_prose_line_numbers_must_stay_correct(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    stale = migrate_row(_row(notes="the add at line 20 feeds k.cu:30"), "eng", diff, new)
    assert {i.code for i in stale.issues} == {"prose-line-reference"}
    # moved.cu is renamed, so a prose path naming it is stale even though its lines are unchanged
    assert migrate_row(_row(notes="see moved.cu:3"), "eng", diff, new).issues
    # a number the diff leaves alone and a file the diff does not touch are fine
    fine = migrate_row(_row(notes="same.cu:3 is untouched; 123 tokens (n = 2048)"), "eng", diff, new)
    assert fine.issues == [] and fine.new_row["line_start"] == 21
    # k.cu has 40 lines at the old commit, so k.cu:300 cannot name it
    assert migrate_row(_row(notes="see k.cu:300"), "eng", diff, new).issues == []
    # another repository's row naming a file of its own is not tied to this repository
    other = _row(id="other-0001", engine="other", sha="1" * 40, file="x.py", evidence=[{"ref": "x.py:1", "claim": "c"}],
                 notes="k.cu:30 in our own tree")
    assert migrate_row(other, "eng", diff, new, own_tree=["x.py", "k.cu"]).issues == []
    assert migrate_row(other, "eng", diff, new, own_tree=["x.py"]).issues


def _cand(cid, file, line, pattern="atomicAdd", col=1, **kw):
    c = {"id": cid, "file": file, "line": line, "col": col, "pattern": pattern, "kind": "atomicAdd", "class_hint": "A?",
         "dtype_hint": "float32", "function": "f", "in_scope": True, "excluded_reason": "", "wrapper_of": ""}
    c.update(kw)
    return c


def test_candidates_are_matched_by_location_and_pattern_not_position(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    old_c = [_cand("eng-c00001", "k.cu", 20), _cand("eng-c00002", "k.cu", 30), _cand("eng-c00003", "moved.cu", 5, "atomicMax")]
    # the new scan renumbers everything: a new candidate sorts first
    new_c = [_cand("eng-c00001", "a_new.cu", 1), _cand("eng-c00002", "k.cu", 22), _cand("eng-c00003", "k.cu", 32),
             _cand("eng-c00004", "renamed/moved.cu", 5, "atomicAdd", dtype_hint="float16")]
    cm = map_candidates(old_c, new_c, diff)
    assert cm.old_to_new == {"eng-c00001": "eng-c00002", "eng-c00002": None, "eng-c00003": None}
    assert [c["id"] for c in cm.new_only] == ["eng-c00001", "eng-c00003", "eng-c00004"]  # changed line and different pattern
    assert [c["id"] for c in cm.removed] == ["eng-c00002", "eng-c00003"]
    row = _row(provenance="scanner", candidate_ids=["eng-c00001"])
    res = migrate_row(row, "eng", diff, new, cm)
    assert not res.issues and res.new_row["candidate_ids"] == ["eng-c00002"]
    lost = migrate_row(_row(provenance="scanner", candidate_ids=["eng-c00001", "eng-c00003"],
                            evidence=[{"ref": "k.cu:20", "claim": "c"}, {"ref": "moved.cu:5", "claim": "c"}]), "eng", diff, new, cm)
    assert "candidate-unmatched" in {i.code for i in lost.issues} and lost.new_row is None


def test_context_change_on_an_unchanged_line_goes_to_review(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    cm = map_candidates([_cand("eng-c00001", "k.cu", 20)], [_cand("eng-c00001", "k.cu", 22, conditions=["if (x)"])], diff)
    assert cm.context_changes == {"eng-c00001": ["conditions"]}
    res = migrate_row(_row(provenance="scanner", candidate_ids=["eng-c00001"]), "eng", diff, new, cm)
    assert {i.code for i in res.issues} == {"candidate-context-changed"}


def test_new_candidate_inside_a_cited_range_goes_to_review(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    cm = map_candidates([_cand("eng-c00001", "k.cu", 20)],
                        [_cand("eng-c00001", "k.cu", 21, "atomic_wrapper"), _cand("eng-c00002", "k.cu", 22)], diff)
    res = migrate_row(_row(provenance="scanner", candidate_ids=["eng-c00001"]), "eng", diff, new, cm)
    assert {i.code for i in res.issues} == {"new-candidate-in-cited-range"}


def test_repeated_matches_at_one_position_pair_up_in_order(repo):
    r, old, new = repo
    diff = EngineDiff(r, old, new)
    old_c = [_cand("eng-c00001", "k.cu", 20, "ptx_string", match="red.add.f32"),
             _cand("eng-c00002", "k.cu", 20, "ptx_string", match="red.add.f32"),
             _cand("eng-c00003", "k.cu", 20, "ptx_string", match="red.add.bf16")]
    new_c = [_cand("eng-c00007", "k.cu", 22, "ptx_string", match="red.add.f32"),
             _cand("eng-c00008", "k.cu", 22, "ptx_string", match="red.add.bf16"),
             _cand("eng-c00009", "k.cu", 22, "ptx_string", match="red.add.f32")]
    cm = map_candidates(old_c, new_c, diff)
    assert cm.old_to_new == {"eng-c00001": "eng-c00007", "eng-c00002": "eng-c00009", "eng-c00003": "eng-c00008"}


def test_scope_drift_reports_where_a_scan_path_went(repo, tmp_path):
    from scan.repin import proposed_scope, scope_drift
    r = tmp_path / "moves"
    r.mkdir()
    _git(r, "init", "-q")
    old = _commit(r, {"kern/csrc/a.cu": _body("a"), "kern/tests/t.py": _body("t"), "srt/x.py": _body("x"),
                      "srt/layers/ops/k.py": _body("k"), "srt/y.py": _body("y")}, "old")
    new = _commit(r, {"kern/csrc/a.cu": None, "kern/tests/t.py": None, "py/kernels/aot/csrc/a.cu": _body("a"),
                      "py/kernels/aot/tests/t.py": _body("t"), "srt/layers/ops/k.py": None, "py/kernels/act/k.py": _body("k"),
                      "srt/y.py": None}, "new")
    diff = EngineDiff(r, old, new)
    entry = {"scan_paths": ["kern", "srt"], "exclude": ["kern/tests"]}
    drift = scope_drift(diff, entry, _git(r, "ls-tree", "-r", "--name-only", new).splitlines())
    assert [(d["field"], d["path"], d["renamed_out"], d["deleted"], d["dominant_destination"], d["exists_at_new"]) for d in drift] == [
        ("scan_paths", "kern", {"py/kernels/aot": 2}, 0, "py/kernels/aot", False),
        ("scan_paths", "srt", {"py/kernels": 1}, 1, None, True),  # reorganised move, grouped one level down
        ("exclude", "kern/tests", {"py/kernels/aot/tests": 1}, 0, "py/kernels/aot/tests", False)]
    assert proposed_scope(entry, drift) == {"scan_paths": ["py/kernels/aot", "srt", "py/kernels"], "exclude": ["py/kernels/aot/tests"]}


def test_target_resolves_a_missing_annotated_tag_to_its_commit(tmp_path):
    from scan.repin import resolve_target
    up = tmp_path / "up"
    up.mkdir()
    _git(up, "-c", "init.defaultBranch=main", "init", "-q")
    old = _commit(up, {"a.cu": _body("a")}, "old")
    tagged = _commit(up, {"a.cu": _body("a") + "// release\n"}, "release")
    _git(up, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", "v1.0", "-m", "v1.0")
    _commit(up, {"a.cu": _body("b")}, "after")
    down = tmp_path / "down"
    down.mkdir()
    _git(down, "init", "-q")
    _git(down, "remote", "add", "origin", str(up))
    _git(down, "fetch", "-q", "--no-tags", "origin", "main")
    assert resolve_target(down, "v1.0", old) == (tagged, "tag v1.0")  # fetched, then dereferenced to the commit
    assert resolve_target(down, old, old) == (old, f"ref {old}")


def test_manifest_records_the_rule_next_to_the_sha():
    from scan.repin import PIN_RULE_NOTE, EnginePlan, record_pin
    manifest = {"generated": "2026-07-17", "notes": "Pinned.", "repos": [
        {"name": "a", "url": "u", "sha": "1" * 40, "commit_date": "2026-07-06T00:00:00Z", "scan_paths": ["x"], "exclude": []},
        {"name": "b", "url": "u", "sha": "3" * 40, "commit_date": "2024-09-04T15:35:00+02:00", "scan_paths": ["y"], "exclude": []}]}
    record_pin(manifest, EnginePlan("a", "1" * 40, "2" * 40, "", "2026-09-09T00:00:00Z", "main", 5, rule="tag v0.29.0"), "2026-09-15")
    record_pin(manifest, EnginePlan("b", "3" * 40, "3" * 40, "", "2024-09-04T13:35:00Z", "master", 0, rule="master before X"), "2026-09-15")
    a, b = manifest["repos"]
    assert list(a) == ["name", "url", "sha", "commit_date", "pin_rule", "scan_paths", "exclude"]
    assert (a["sha"], a["commit_date"], a["pin_rule"]) == ("2" * 40, "2026-09-09T00:00:00Z", "tag v0.29.0")
    assert (b["sha"], b["commit_date"], b["pin_rule"]) == ("3" * 40, "2024-09-04T15:35:00+02:00", "master before X")  # date kept
    assert manifest["notes"] == "Pinned. " + PIN_RULE_NOTE and manifest["generated"] == "2026-09-15"
    record_pin(manifest, EnginePlan("a", "2" * 40, "2" * 40, "", "", "main", 0, rule="tag v0.29.0"), "2026-09-15")
    assert manifest["notes"].count(PIN_RULE_NOTE) == 1
