#!/usr/bin/env python3
"""Stage verified C2 handoffs and reproduce every completed comparison on CPU.

Default mode is read-only and audits this public directory.  ``--stage-from``
is the maintainer path: it accepts only labels with a verified ACK, verifies
every original handoff byte, and adds their evidence to this directory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
FROZEN_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
CACHE_ROOTS = {
    "CUDA_CACHE_PATH",
    "DG_JIT_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TMPDIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "VLLM_CACHE_ROOT",
    "XDG_CACHE_HOME",
}
EXPECTED_VBIOS = {"tp1": "96.00.DA.00.0C", "tp2": "96.00.89.00.01"}
EXPECTED_LABELS = {
    "BI_b", "BI_c", "MIX_default_negative", "DET_MIX_record", "DET_MIX_warm",
    "DET_MIX_a", "DET_MIX_b", "DET_MIX_c", "MOE_EAGER_record",
    "MOE_EAGER_replay", "TP2_plain_c",
}
SOURCE_SUFFIXES = {
    ".best_config", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp",
    ".json", ".kernel_perf", ".llir", ".log", ".ninja", ".ptx",
    ".py", ".source", ".ttgir", ".ttir", ".txt",
}
def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def safe_relative(value: str) -> Path:
    path = Path(value)
    assert value and not path.is_absolute() and ".." not in path.parts, value
    return path


def public_relative(path: Path, scope: str) -> Path:
    if scope == "tp1":
        return path
    assert scope == "tp2", scope
    heads = {
        "e6": Path("results_tp2/e6"),
        "logs": Path("logs_tp2"),
        "handoff": Path("handoff_tp2"),
        "evidence": Path("evidence_tp2"),
    }
    assert path.parts and path.parts[0] in heads, path
    return heads[path.parts[0]].joinpath(*path.parts[1:])


def write_bytes(path: Path, data: bytes, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        assert path.read_bytes() == data, f"staged immutable file changed: {path}"
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def deterministic_gzip(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as handle:
        handle.write(data)
    return output.getvalue()


def gunzip(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {"schema": manifest["schema"], "label": manifest["label"], "files": manifest["files"]}
    return sha256_bytes(canonical_json(payload))


def parse_command(data: bytes) -> list[str]:
    first = data.decode(errors="replace").splitlines()[0]
    assert first.startswith("COMMAND "), first
    command = json.loads(first.removeprefix("COMMAND "))
    assert isinstance(command, list) and all(isinstance(item, str) for item in command)
    return command


def source_archive(full_archive: bytes) -> tuple[bytes, int]:
    """Return a deterministic tar.gz containing generated text/source/IR only."""
    selected: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(full_archive), mode="r:gz") as source:
        for member in source.getmembers():
            if not member.isfile() or Path(member.name).suffix not in SOURCE_SUFFIXES:
                continue
            handle = source.extractfile(member)
            assert handle is not None
            selected.append((member.name, handle.read()))
    selected.sort()
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as output:
            for name, data in selected:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                output.addfile(info, io.BytesIO(data))
    return raw.getvalue(), len(selected)


def verify_source_archive(path: Path) -> int:
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        names = []
        for member in archive.getmembers():
            assert member.isfile(), (path, member.name)
            assert Path(member.name).suffix in SOURCE_SUFFIXES, (path, member.name)
            assert member.mtime == member.uid == member.gid == 0, (path, member.name)
            assert member.mode == 0o644, (path, member.name)
            names.append(member.name)
            count += 1
        assert names == sorted(names) and len(names) == len(set(names)), path
    return count


def verify_original_handoff(work: Path, label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    results = work / "c2-results"
    ack_path = work / f"{label}.verified-ack.json"
    manifest_path = results / "handoff" / label / "manifest.json"
    ack, manifest = load_json(ack_path), load_json(manifest_path)
    assert ack == {
        "schema": 1,
        "label": label,
        "handoff_manifest_sha256": manifest["sha256"],
        "verified_manifest_sha256": manifest["sha256"],
    }, ack_path
    assert manifest["schema"] == 1 and manifest["label"] == label
    assert manifest["sha256"] == manifest_digest(manifest), manifest_path
    payloads: dict[str, bytes] = {}
    metadata: dict[str, tuple[int, str]] = {}
    for row in manifest["files"]:
        relative = safe_relative(row["path"])
        data = (results / relative).read_bytes()
        actual = (len(data), sha256_bytes(data))
        expected = (row["size"], row["sha256"])
        assert actual == expected, (label, relative, actual, expected)
        if row["path"] in metadata:
            assert metadata[row["path"]] == expected, (label, relative)
        metadata[row["path"]] = expected
        payloads[row["path"]] = data
    return ack, manifest, payloads


def stage_provenance(work: Path) -> None:
    files = {
        work / "logs/c2_setup.log": ROOT / "provenance/setup_attempt1.log.gz",
        work / "c2_setup/c2_setup.log": ROOT / "provenance/setup_success.log.gz",
        work / "c2_setup/c2_ready_cpu.log": ROOT / "provenance/setup_ready_cpu.log.gz",
        work / "c2_setup/readiness_cpu.json": ROOT / "provenance/readiness_cpu.json",
        work / "c2_setup/SHA256SUMS-final.txt": ROOT / "provenance/setup_SHA256SUMS.txt",
        work / "c2_setup/staged_records_sha256.txt": ROOT / "provenance/staged_records_sha256.txt",
        work / "c2_transfer_sha256.txt": ROOT / "provenance/c2_transfer_sha256.txt",
        work / "recover_bi_ack.py": ROOT / "provenance/recovery/recover_bi_ack.py",
        work / "tp2_collective_backend_audit.json": ROOT / "provenance/tp2_collective_backend_audit.json",
        work / "tp2_collective_backend_audit.md": ROOT / "provenance/tp2_collective_backend_audit.md",
        work / "lifecycle_summary.json": ROOT / "provenance/lifecycle/lifecycle_summary.json",
        work / "control.json": ROOT / "provenance/lifecycle/tp1/control.json",
        work / "teardown_requested.json": ROOT / "provenance/lifecycle/tp1/teardown_requested.json",
        work / "watchdog.jsonl": ROOT / "provenance/lifecycle/tp1/watchdog.jsonl",
        work / "watchdog-process.log": ROOT / "provenance/lifecycle/tp1/watchdog-process.log.gz",
        work / "tp2-any-region/control.json": ROOT / "provenance/lifecycle/tp2/control.json",
        work / "tp2-any-region/teardown_requested.json": ROOT / "provenance/lifecycle/tp2/teardown_requested.json",
        work / "tp2-any-region/watchdog.jsonl": ROOT / "provenance/lifecycle/tp2/watchdog.jsonl",
        work / "tp2-any-region/watchdog-process.log": ROOT / "provenance/lifecycle/tp2/watchdog-process.log.gz",
    }
    for source, target in files.items():
        if not source.is_file():
            continue
        data = source.read_bytes()
        write_bytes(target, deterministic_gzip(data) if target.suffix == ".gz" else data)
    recovery = work / "c2-recovery"
    if recovery.is_dir():
        for source in sorted(path for path in recovery.rglob("*") if path.is_file()):
            relative = source.relative_to(recovery)
            target = ROOT / "provenance/recovery" / relative
            data = source.read_bytes()
            if source.suffix == ".log":
                target = target.with_name(target.name + ".gz")
                data = deterministic_gzip(data)
            write_bytes(target, data)

    # These exact pinned sources bound what the backend audit can establish.
    source_audit = load_json(work / "tp2_collective_backend_audit.json")
    for row in source_audit["source_semantics"]["files"]:
        relative = safe_relative(row["path"])
        source = REPO / relative
        data = source.read_bytes()
        assert (len(data), sha256_bytes(data)) == (row["size"], row["sha256"]), source
        write_bytes(ROOT / "provenance/collective_source" / relative, data)


def stage_tp2_provenance(work: Path) -> None:
    setup = work / "c2_setup"
    if setup.is_dir():
        for source in sorted(path for path in setup.rglob("*") if path.is_file()):
            target = ROOT / "provenance/tp2_setup" / source.relative_to(setup)
            data = source.read_bytes()
            if source.suffix == ".log":
                target = target.with_name(target.name + ".gz")
                data = deterministic_gzip(data)
            write_bytes(target, data)
    failed = work / "failed-cache-layout-results"
    verified_path = work / "failed-cache-layout-results.verified.json"
    if failed.is_dir() and verified_path.is_file():
        verified = load_json(verified_path)
        for relative_text, expected in sorted(verified.items()):
            relative = safe_relative(relative_text)
            source = failed / relative
            data = source.read_bytes()
            assert sha256_bytes(data) == expected, source
            target = ROOT / "provenance/tp2_failed_cache_layout" / relative
            if source.suffix == ".log":
                target = target.with_name(target.name + ".gz")
                data = deterministic_gzip(data)
            write_bytes(target, data)
        write_bytes(
            ROOT / "provenance/tp2_failed_cache_layout/SHA256.json",
            canonical_json(verified),
        )


def stage_final_metadata(work: Path, scope: str) -> None:
    """Preserve the final immutable run ledger after the pod has stopped."""
    results = work / "c2-results"
    target_root = ROOT / "provenance" / f"{scope}_final"
    for relative in (
        Path("COMPLETE.json"), Path("STOP"), Path("immutable_records.json"),
        Path("runner_state.json"), Path("runner/run_followup.py"), Path("logs/smoke.log"),
    ):
        source = results / relative
        assert source.is_file(), source
        target = target_root / relative
        data = source.read_bytes()
        if source.suffix == ".log":
            target = target.with_name(target.name + ".gz")
            data = deterministic_gzip(data)
        write_bytes(target, data)


def stage(work: Path, scope: str) -> None:
    work = work.resolve()
    ack_paths = sorted(work.glob("*.verified-ack.json"))
    assert ack_paths or scope == "tp2", f"no verified ACKs below {work}"
    existing_completed = load_json(ROOT / "completed.json")["labels"] if (ROOT / "completed.json").is_file() else []
    completed: list[dict[str, Any]] = [row for row in existing_completed if row.get("scope", "tp1") != scope]
    omitted_path = ROOT / "provenance/full_cache_archives.json"
    existing_omitted = load_json(omitted_path)["archives"] if omitted_path.is_file() else []
    omitted: dict[tuple[str, str], dict[str, Any]] = {
        (row.get("scope", "tp1"), row["path"]): row
        for row in existing_omitted if row.get("scope", "tp1") != scope
    }
    handoff_root = "handoff" if scope == "tp1" else "handoff_tp2"
    generated_root = "generated_sources" if scope == "tp1" else "generated_sources_tp2"
    for ack_path in ack_paths:
        label = ack_path.name.removesuffix(".verified-ack.json")
        ack, manifest, payloads = verify_original_handoff(work, label)
        write_bytes(ROOT / handoff_root / label / "manifest.json", canonical_json(manifest), immutable=True)
        write_bytes(ROOT / handoff_root / label / "verified-ack.json", canonical_json(ack), immutable=True)
        for relative_text, data in sorted(payloads.items()):
            relative = safe_relative(relative_text)
            if relative.parts[:1] == ("archives",) and relative.name == "cache.tar.gz":
                public_name = f"{generated_root}/{label}.sources.tar.gz"
                light, count = source_archive(data)
                write_bytes(ROOT / public_name, light, immutable=True)
                row = {
                    "label": label,
                    "scope": scope,
                    "path": relative_text,
                    "size": len(data),
                    "sha256": sha256_bytes(data),
                    "public_source_archive": public_name,
                    "public_source_files": count,
                    "public_source_sha256": sha256_bytes(light),
                }
                prior = omitted.get((scope, relative_text))
                assert prior is None or prior == row
                omitted[(scope, relative_text)] = row
                continue
            target = ROOT / public_relative(relative, scope)
            if relative.suffix == ".log":
                target = target.with_name(target.name + ".gz")
                data = deterministic_gzip(data)
            write_bytes(target, data, immutable=True)

        compare_log = payloads.get(f"logs/{label}.compare.log")
        run_log = payloads.get(f"logs/{label}.run.log")
        assert run_log is not None, label
        run_command = parse_command(run_log)
        if compare_log is None:
            assert "--record" in run_command, (label, run_command)
            completed.append({"label": label, "scope": scope, "kind": "record", "manifest_sha256": manifest["sha256"]})
            continue
        command = parse_command(compare_log)
        index = command.index("--compare")
        record, replay = Path(command[index + 1]), Path(command[index + 2])
        assert record.name == "record" and replay.parent == record.parent, command
        arm, replay_name = record.parent.name, replay.name
        summary_name = "summary.json" if replay_name == "replay" else f"summary_{replay_name}.json"
        original_summary = Path("e6") / arm / summary_name
        assert original_summary.as_posix() in payloads, (label, original_summary)
        summary = public_relative(original_summary, scope)
        completed.append({
            "label": label,
            "scope": scope,
            "kind": "replay",
            "arm": arm,
            "replay": replay_name,
            "summary": summary.as_posix(),
            "manifest_sha256": manifest["sha256"],
        })

    if scope == "tp1":
        stage_provenance(work)
    else:
        stage_tp2_provenance(work)
    stage_final_metadata(work, scope)
    completed.sort(key=lambda row: (row.get("scope", "tp1"), row["label"]))
    write_bytes(ROOT / "completed.json", canonical_json({"schema": 1, "labels": completed}))
    write_bytes(ROOT / "provenance/full_cache_archives.json", canonical_json({
        "schema": 1,
        "note": "Full binary cache archives remain in the durable private work root; public archives retain generated source and IR only.",
        "archives": sorted(omitted.values(), key=lambda row: (row.get("scope", "tp1"), row["path"])),
    }))
    digest_rows = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path != ROOT / "SHA256.json" and "__pycache__" not in path.parts:
            digest_rows[path.relative_to(ROOT).as_posix()] = sha256_bytes(path.read_bytes())
    write_bytes(ROOT / "SHA256.json", canonical_json(digest_rows))


def staged_handoff_bytes(relative: Path, scope: str) -> bytes | None:
    if relative.parts[:1] == ("archives",):
        return None
    direct = ROOT / public_relative(relative, scope)
    if direct.is_file():
        return direct.read_bytes()
    compressed = direct.with_name(direct.name + ".gz")
    if compressed.is_file():
        return gunzip(compressed)
    return None


def verify_public_handoffs() -> tuple[list[dict[str, Any]], int]:
    completed = load_json(ROOT / "completed.json")["labels"]
    omitted_rows = load_json(ROOT / "provenance/full_cache_archives.json")["archives"]
    files_verified = 0
    for item in completed:
        label, scope = item["label"], item.get("scope", "tp1")
        handoff_root = "handoff" if scope == "tp1" else "handoff_tp2"
        manifest = load_json(ROOT / handoff_root / label / "manifest.json")
        ack = load_json(ROOT / handoff_root / label / "verified-ack.json")
        assert manifest["sha256"] == item["manifest_sha256"] == manifest_digest(manifest)
        assert ack["schema"] == 1 and ack["label"] == label
        assert ack["handoff_manifest_sha256"] == ack["verified_manifest_sha256"] == manifest["sha256"]
        for row in manifest["files"]:
            relative = safe_relative(row["path"])
            data = staged_handoff_bytes(relative, scope)
            if data is None:
                omission = next(
                    (entry for entry in omitted_rows if entry.get("scope", "tp1") == scope and entry["path"] == row["path"]),
                    None,
                )
                assert omission and omission["size"] == row["size"] and omission["sha256"] == row["sha256"], row
                source = ROOT / omission["public_source_archive"]
                assert sha256_bytes(source.read_bytes()) == omission["public_source_sha256"]
                assert verify_source_archive(source) == omission["public_source_files"]
            else:
                assert len(data) == row["size"] and sha256_bytes(data) == row["sha256"], (label, relative)
            files_verified += 1
    return completed, files_verified


def tree_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            })
    payload = {"schema": 1, "files": rows}
    return {**payload, "sha256": sha256_bytes(canonical_json(payload))}


def verify_cache_snapshot(path: Path, label: str, *, require_empty: bool) -> dict[str, Any]:
    snapshot = load_json(path)
    payload = {"schema": snapshot["schema"], "roots": snapshot["roots"]}
    assert snapshot["sha256"] == sha256_bytes(canonical_json(payload)), path
    assert {row["name"] for row in snapshot["roots"]} == CACHE_ROOTS, path
    assert len(snapshot["roots"]) == len(CACHE_ROOTS), path
    owners = {Path(row["path"]).parent.name for row in snapshot["roots"]}
    assert len(owners) == 1, path
    if require_empty:
        assert owners == {label}, (path, owners)
        assert all(row["file_count"] == 0 and row["files"] == [] for row in snapshot["roots"]), path
    for row in snapshot["roots"]:
        assert row["file_count"] == len(row["files"]), (path, row["name"])
    return snapshot


def verify_record(item: dict[str, Any]) -> None:
    arm, scope = item["arm"], item.get("scope", "tp1")
    result_root = ROOT if scope == "tp1" else ROOT / "results_tp2"
    handoff_root = ROOT / ("handoff" if scope == "tp1" else "handoff_tp2")
    record = result_root / "e6" / arm / "record"
    state = load_json(handoff_root / item["label"] / "state_before_ack.json")
    expected = state["records"][arm]
    expected = {key: value for key, value in expected.items() if key != "cache_owner"}
    assert tree_manifest(record) == expected, (item["label"], arm)
    schedule = load_json(record / "schedule.json")
    assert not schedule["errors"]
    assert schedule["forward_passes_at_scheduler"] == len(schedule["passes"])


def verify_preflight(completed: list[dict[str, Any]]) -> None:
    stable = ("gpu", "gpu_count", "driver", "cuda", "cudnn", "torch", "python", "sm", "sm_count", "installed_distributions")
    for scope in sorted({item.get("scope", "tp1") for item in completed}):
        evidence = ROOT / ("evidence" if scope == "tp1" else "evidence_tp2")
        result_root = ROOT if scope == "tp1" else ROOT / "results_tp2"
        report = load_json(evidence / "preflight_report.json")
        assert report["schema"] == 1 and report["compatible"] is True and report["errors"] == []
        assert report["expected_source"] == report["actual_source"] == FROZEN_SOURCE
        assert report["source_dirty"] is False
        actual = report["actual"]
        assert actual["vbios"] == [EXPECTED_VBIOS[scope]] * report["required_gpus"], (scope, actual["vbios"])
        for key, value in report["required_env"].items():
            assert actual["env"].get(key) == value, (scope, key)
        for item in completed:
            if item["kind"] != "replay" or item.get("scope", "tp1") != scope:
                continue
            env = load_json(result_root / "e6" / item["arm"] / "record/env.json")
            for key in stable:
                assert env.get(key) == actual.get(key), (item["label"], key)


def export_frozen_comparator(target: Path) -> Path:
    shape = target / "probes/shape"
    shape.mkdir(parents=True)
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", FROZEN_SOURCE, "probes/shape"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    roots = {Path("probes/shape"), Path("probes/shape/shape_hook_pkg/shape_hook")}
    frozen_files = [name for name in listing if Path(name).suffix == ".py" and Path(name).parent in roots]
    assert "probes/shape/run_e6.py" in frozen_files
    for relative in frozen_files:
        result = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{FROZEN_SOURCE}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        )
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_bytes(result.stdout)
    return shape / "run_e6.py"


def reproduce_rank1(comparator: Path, record: Path, replay: Path, label: str, summary: dict[str, Any]) -> dict[str, Any]:
    code = r'''import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from run_e6 import compare_replay
from shape_common import read_hook, real_steps
record, replay = Path(sys.argv[2]), Path(sys.argv[3])
def steps(path, rank):
    return real_steps(read_hook(path / "hook", rank))
rec1, rep1 = steps(record, 1), steps(replay, 1)
report = compare_replay(rec1, rep1)
cross = {"h": 0, "logits_h": 0, "argmax": 0}
for left, right in ((steps(record, 0), rec1), (steps(replay, 0), rep1)):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert [row["req"] for row in a["requests"]] == [row["req"] for row in b["requests"]]
        for x, y in zip(a["requests"], b["requests"]):
            assert x.get("new_token_ids") == y.get("new_token_ids")
            for field in cross:
                cross[field] += int(x.get(field) != y.get(field))
print(json.dumps({"rank1": report, "cross_rank_differing_rows": cross}, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(comparator.parent), str(record), str(replay)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reproduced = json.loads(result.stdout)
    saved = load_json(ROOT / f"evidence_tp2/rank1_{label}.json")
    assert reproduced["rank1"] == saved["rank1"], label
    assert reproduced["cross_rank_differing_rows"] == saved["cross_rank_differing_rows"], label
    report = reproduced["rank1"]
    assert report["validation_errors"] == [] and report["first_mismatch"] is None
    assert report["verdict_P2"] == summary["verdict_P2"]
    assert report["rows_differing"] == summary["rows_differing"]
    assert reproduced["cross_rank_differing_rows"]["logits_h"] == 0
    assert reproduced["cross_rank_differing_rows"]["argmax"] == 0
    return {
        "rank1_verdict_P2": report["verdict_P2"],
        "rank1_rows_differing": report["rows_differing"],
        "cross_rank_differing_rows": reproduced["cross_rank_differing_rows"],
    }


def reproduce_comparisons(completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    with tempfile.TemporaryDirectory(prefix="c2-followup-audit-") as temporary:
        temporary = Path(temporary)
        comparator = export_frozen_comparator(temporary / "source")
        for item in completed:
            if item["kind"] != "replay":
                continue
            verify_record(item)
            label, arm, replay_name, scope = item["label"], item["arm"], item["replay"], item.get("scope", "tp1")
            logs = ROOT / ("logs" if scope == "tp1" else "logs_tp2")
            result_root = ROOT if scope == "tp1" else ROOT / "results_tp2"
            before = verify_cache_snapshot(logs / f"{label}.cache_before.json", label, require_empty=False)
            after = verify_cache_snapshot(logs / f"{label}.cache_after.json", label, require_empty=False)
            assert [row["name"] for row in before["roots"]] == [row["name"] for row in after["roots"]]
            owners = {Path(row["path"]).parent.name for row in before["roots"]}
            if owners == {label}:
                assert all(row["file_count"] == 0 and row["files"] == [] for row in before["roots"]), label
            else:
                assert any(row["file_count"] > 0 for row in before["roots"]), (label, owners)
            source_arm = result_root / "e6" / arm
            target_arm = temporary / "runs" / scope / arm
            target_arm.mkdir(parents=True, exist_ok=True)
            if not (target_arm / "record").exists():
                shutil.copytree(source_arm / "record", target_arm / "record")
            shutil.copytree(source_arm / replay_name, target_arm / replay_name)
            subprocess.run(
                [sys.executable, str(comparator), "--compare", str(target_arm / "record"), str(target_arm / replay_name)],
                check=True,
                cwd=comparator.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            name = "summary.json" if replay_name == "replay" else f"summary_{replay_name}.json"
            generated = (target_arm / name).read_bytes()
            saved = (ROOT / item["summary"]).read_bytes()
            assert generated == saved, label
            summary = json.loads(saved)
            assert summary["requirements_met"] and summary["validation_errors"] == []
            assert summary["verdict_P2"] in {"IDENTICAL", "DIFFERS"}
            assert summary["forcing_log"]["calls"] == summary["passes_replayed"]
            comparison = {
                "label": label,
                "scope": scope,
                "arm": arm,
                "replay": replay_name,
                "verdict_P2": summary["verdict_P2"],
                "rows_compared": summary["rows_compared"],
                "rows_differing": summary["rows_differing"],
                "forcing_calls": summary["forcing_log"]["calls"],
            }
            if scope == "tp2":
                comparison.update(reproduce_rank1(
                    comparator, target_arm / "record", target_arm / replay_name, label, summary
                ))
            comparisons.append(comparison)
    return comparisons


def verify_moe_routing() -> list[dict[str, Any]]:
    arm = "Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs0_prefix0_routing_diag"
    result = []
    for name, evidence_name in (("record", "moe_routing_record.json"), ("replay_warm", "moe_routing_replay_warm.json")):
        report = load_json(ROOT / "evidence" / evidence_name)
        assert report["schema"] == 1 and report["valid"] is True and report["errors"] == []
        assert report["forward_passes"] == 37 and report["sources"] == {"actual_router": 37}
        assert report["num_experts"] == 60 and report["num_experts_per_tok"] == 4
        rows = []
        for line in (ROOT / "e6" / arm / name / "hook/rank0.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("event") != "forward":
                continue
            assert row.get("resolved_compile") == "0"
            assert row.get("resolved_cudagraph") == "NONE"
            assert row.get("moe_counts_source") == "actual_router"
            counts = row.get("moe_expert_counts_first_layer")
            assert isinstance(counts, list) and len(counts) == 60
            assert all(isinstance(value, int) and value >= 0 for value in counts)
            assert sum(counts) == int(row["total_scheduled"]) * 4
            rows.append(row)
        assert len(rows) == 37
        result.append({"run": name, "forward_passes": 37, "source": "actual_router", "experts": 60, "top_k": 4})
    return result


def verify_final_metadata(completed: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scope in ("tp1", "tp2"):
        root = ROOT / "provenance" / f"{scope}_final"
        expected = {row["label"] for row in completed if row.get("scope", "tp1") == scope}
        complete = load_json(root / "COMPLETE.json")
        state = load_json(root / "runner_state.json")
        immutable = load_json(root / "immutable_records.json")
        assert complete["schema"] == state["schema"] == immutable["schema"] == 1
        assert set(complete["completed_labels"]) == set(state["completed_labels"]) == expected
        runner = root / state["runner_copy"]
        assert runner.is_file() and sha256_bytes(runner.read_bytes()) == state["runner_sha256"]
        assert state["frozen_source"] == FROZEN_SOURCE
        assert immutable["records"] == state["records"]
        stop = json.loads((root / "STOP").read_text())
        assert "block further launches" in stop["reason"]
        result[scope] = {
            "completed_labels": len(expected),
            "runner_sha256": state["runner_sha256"],
            "stopped_utc": stop["utc"],
        }
    return result


def verify_lifecycle() -> dict[str, Any]:
    root = ROOT / "provenance/lifecycle"
    summary = load_json(root / "lifecycle_summary.json")
    assert len(summary["pods"]) == 2
    for scope, pod in zip(("tp1", "tp2"), summary["pods"], strict=True):
        control = load_json(root / scope / "control.json")
        teardown = load_json(root / scope / "teardown_requested.json")
        events = [json.loads(line) for line in (root / scope / "watchdog.jsonl").read_text().splitlines()]
        assert control["pod_id"] == pod["pod_id"]
        assert teardown.get("pod_id", teardown.get("pod", {}).get("id")) == pod["pod_id"]
        assert teardown["stdout"] and '"deleted": true' in teardown["stdout"]
        assert teardown["stderr"] == ""
        assert events[-1] == pod["absence_verified"]
        assert events[-1]["event"] == "absence_verified" and events[-1]["consecutive_reads"] >= 2
    assert summary["estimated_gpu_cost_total_usd"] <= summary["batch_cap_usd"]
    return {
        "pods_deleted_and_absence_verified": len(summary["pods"]),
        "estimated_gpu_cost_total_usd": summary["estimated_gpu_cost_total_usd"],
        "batch_cap_usd": summary["batch_cap_usd"],
    }


def audit_reference_bytes(path_text: str) -> bytes:
    relative = safe_relative(path_text)
    tp2_prefix = Path("tp2-any-region/c2-results")
    if relative.parts[:len(tp2_prefix.parts)] == tp2_prefix.parts:
        data = staged_handoff_bytes(Path(*relative.parts[len(tp2_prefix.parts):]), "tp2")
        assert data is not None, path_text
        return data
    if relative.parts[:1] == ("repos",):
        return (ROOT / "provenance/collective_source" / relative).read_bytes()
    return (REPO / relative).read_bytes()


def verify_collective_backends() -> dict[str, Any]:
    report = load_json(ROOT / "provenance/tp2_collective_backend_audit.json")
    assert report["schema"] == 1
    assert report["causal_status"] == "UNRESOLVED"
    assert report["evidence_state"] == "COLLECTIVE_SUBBACKEND_MISMATCH"
    assert report["held_fixed"]["resolved_fuse_allreduce_rms"] is False
    assert report["source_semantics"]["pinned_vllm"] == {
        "commit": "98dff2a81d747d1dba01a47f939f48c3526d4206", "tag": "v0.29.0"
    }
    for cell in report["cells"]:
        assert cell["per_call_kernel_launch_observed"] is False
        for key in ("log", "run", "summary", "preflight", "rank1_evidence"):
            reference = cell.get(key)
            if reference is None:
                continue
            data = audit_reference_bytes(reference["path"])
            assert (len(data), sha256_bytes(data)) == (reference["size"], reference["sha256"]), reference["path"]
            if key == "log":
                line_count = len(data.decode(errors="replace").splitlines())
                assert all(1 <= first <= last <= line_count for first, last in reference["line_ranges"])
    for reference in report["source_semantics"]["files"]:
        data = audit_reference_bytes(reference["path"])
        assert (len(data), sha256_bytes(data)) == (reference["size"], reference["sha256"]), reference["path"]
        line_count = len(data.decode(errors="replace").splitlines())
        assert all(1 <= first <= last <= line_count for first, last in reference["line_ranges"])
    cells = {cell["id"]: cell for cell in report["cells"]}
    assert cells["old_m3_record"]["workspace_backend"] == "mnnvl"
    assert cells["old_m3_warm_replay"]["workspace_backend"] == "mnnvl"
    assert [cells[name]["workspace_backend"] for name in ("old_m4_cold_a", "old_m4_cold_b", "new_cold_followup_c")] == ["trtllm"] * 3
    return {
        "historical_m3_record": "mnnvl",
        "historical_m4_plain_cold": ["trtllm", "trtllm"],
        "followup_tp2_plain_c": "trtllm",
        "causal_status": report["causal_status"],
        "per_call_kernel_launch_observed": False,
        "interpretation_limit": "The follow-up reproduces a cold TP=2 difference on the fallback trtllm backend; it is not a fixed-collective-kernel comparison with the historical M3 mnnvl record.",
    }


def audit() -> dict[str, Any]:
    checksums = load_json(ROOT / "SHA256.json")
    for relative, expected in checksums.items():
        assert sha256_bytes((ROOT / safe_relative(relative)).read_bytes()) == expected, relative
    completed, handoff_files = verify_public_handoffs()
    assert {item["label"] for item in completed} == EXPECTED_LABELS
    assert sum(item["kind"] == "record" for item in completed) == 2
    assert sum(item["kind"] == "replay" for item in completed) == 9
    verify_preflight(completed)
    comparisons = reproduce_comparisons(completed)
    assert len(comparisons) == sum(item["kind"] == "replay" for item in completed)
    return {
        "schema": 1,
        "frozen_source": FROZEN_SOURCE,
        "public_files_verified": len(checksums),
        "handoff_entries_verified": handoff_files,
        "completed_labels": len(completed),
        "completed_replays": len(comparisons),
        "comparisons": comparisons,
        "moe_routing": verify_moe_routing(),
        "collective_backends": verify_collective_backends(),
        "final_metadata": verify_final_metadata(completed),
        "lifecycle": verify_lifecycle(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-from", type=Path, help="durable TP=1 2026-09-16-followup work root")
    parser.add_argument("--stage-tp2", type=Path, help="durable TP=2 work root; staged into separate public namespaces")
    args = parser.parse_args(argv)
    if args.stage_from:
        stage(args.stage_from, "tp1")
    if args.stage_tp2:
        stage(args.stage_tp2, "tp2")
    print(json.dumps(audit(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
