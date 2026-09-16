#!/usr/bin/env python3
"""CPU-only audit and public staging for the factorial follow-up.

The raw results root is never written.  A label contributes observations only
after its ACK names the matching handoff manifest and every file named by that
manifest still has its recorded SHA-256.  This deliberately permits a live
campaign to be reported as partial.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


FROZEN_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
PHASES = ("record", "warm", "cold_0", "cold_1", "cold_2")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_sums(staging: Path) -> int:
    """Verify SHA256SUMS as an exact, safe inventory before importing code."""
    sums = staging / "SHA256SUMS.txt"
    if not sums.is_file() or sums.is_symlink():
        raise AuditError("public bundle lacks a regular SHA256SUMS.txt")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(sums.read_text().splitlines(), 1):
        try:
            digest, text = line.split("  ", 1)
        except ValueError as exc:
            raise AuditError(f"malformed SHA256SUMS row {line_number}") from exc
        relative = PurePosixPath(text)
        if (
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not text
            or relative.is_absolute()
            or relative.as_posix() != text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or text == "SHA256SUMS.txt"
            or text in expected
        ):
            raise AuditError(f"unsafe or duplicate SHA256SUMS row {line_number}: {text!r}")
        expected[text] = digest
    actual: dict[str, str] = {}
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise AuditError(f"public bundle contains a symlink: {path.relative_to(staging)}")
        if not path.is_file() or path == sums:
            continue
        relative = path.relative_to(staging).as_posix()
        actual[relative] = sha256_file(path)
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AuditError(f"SHA256SUMS coverage mismatch: missing={missing}, extra={extra}")
    mismatched = sorted(path for path in expected if expected[path] != actual[path])
    if mismatched:
        raise AuditError(f"SHA256SUMS digest mismatch: {mismatched}")
    return len(actual)


def cells() -> list[dict[str, Any]]:
    return [
        {
            "label": f"d{deterministic}_c{combo}_b{benchmark}",
            "deterministic": deterministic,
            "combo_kernels": bool(combo),
            "benchmark_combo_kernel": bool(benchmark),
        }
        for deterministic, combo, benchmark in itertools.product((0, 1), repeat=3)
    ]


class AuditError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON: {path}") from exc


def manifest_path(results: Path, label: str) -> Path:
    return results / "handoff" / label / "manifest.json"


def ack_path(results: Path, ack_dir: Path | None, label: str) -> Path | None:
    candidates = []
    if ack_dir is not None:
        candidates.extend((ack_dir / f"{label}.verified-ack.json", ack_dir / f"{label}.json"))
    candidates.extend((
        results / "acks" / f"{label}.verified-ack.json",
        results / "acks" / f"{label}.json",
        results / "handoff" / label / "verified-ack.json",
    ))
    return next((path for path in candidates if path.is_file()), None)


def verified_manifest(results: Path, ack_dir: Path | None, label: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return a verified manifest or the reason it cannot be admitted."""
    manifest_file = manifest_path(results, label)
    ack_file = ack_path(results, ack_dir, label)
    if not manifest_file.is_file():
        return None, "missing handoff manifest"
    if ack_file is None:
        return None, "missing SHA acknowledgement"
    try:
        manifest, ack = read_json(manifest_file), read_json(ack_file)
    except AuditError as exc:
        return None, str(exc)
    if not isinstance(manifest, dict) or not isinstance(ack, dict):
        return None, "manifest or acknowledgement is not an object"
    digest = manifest.get("sha256")
    if (
        manifest.get("schema") != 1
        or manifest.get("label") != label
        or not isinstance(digest, str)
        or digest != hashlib.sha256(canonical_json({k: v for k, v in manifest.items() if k != "sha256"})).hexdigest()
    ):
        return None, "manifest digest/schema/label is invalid"
    if (
        ack.get("schema") != 1
        or ack.get("label") != label
        or ack.get("handoff_manifest_sha256") != digest
        or ack.get("verified_manifest_sha256") != digest
    ):
        return None, "acknowledgement does not verify manifest"
    files = manifest.get("files")
    if not isinstance(files, list):
        return None, "manifest has no files list"
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            return None, "manifest has invalid file entry"
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            return None, f"unsafe manifest path: {row['path']}"
        path = results.joinpath(*relative.parts)
        if not path.is_file() or path.stat().st_size != row.get("size") or sha256_file(path) != row.get("sha256"):
            return None, f"manifest file failed verification: {row['path']}"
    return manifest, None


def expected_config(cell: dict[str, Any]) -> dict[str, bool]:
    return {key: cell[key] for key in ("combo_kernels", "benchmark_combo_kernel")}


def validate_run(run_dir: Path, cell: dict[str, Any], phase: str) -> list[str]:
    errors: list[str] = []
    try:
        run, env = read_json(run_dir / "run.json"), read_json(run_dir / "env.json")
    except AuditError as exc:
        return [str(exc)]
    args = run.get("args") if isinstance(run, dict) else None
    captured = env.get("env") if isinstance(env, dict) else None
    if run.get("status") != "configured":
        errors.append(f"run status is {run.get('status')!r}")
    if not isinstance(args, dict) or not args.get("mixed") or args.get("prefix_caching") != 0:
        errors.append("run arguments do not attest mixed schedule with prefix caching off")
    if run.get("resolved_compile") in (None, "0") or run.get("resolved_cudagraph") in (None, "NONE"):
        errors.append("run did not resolve compiled CUDA graphs")
    if not isinstance(captured, dict) or captured.get("TORCHINDUCTOR_DETERMINISTIC") != str(cell["deterministic"]):
        errors.append("captured deterministic setting differs from cell")
    resolved = run.get("resolved_config", "") if isinstance(run, dict) else ""
    for key, value in expected_config(cell).items():
        if f"'{key}': {value}" not in resolved:
            errors.append(f"resolved config lacks {key}={value}")
    if phase != "record" and run.get("recorded_run_id") is None:
        errors.append("replay does not name recorded run id")
    return errors


def forcing_check(summary: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        report = read_json(summary)
    except AuditError as exc:
        return None, [str(exc)]
    errors = []
    forcing = report.get("forcing_log") if isinstance(report, dict) else None
    if report.get("requirements_met") is not True:
        errors.append("frozen comparator requirements were not met")
    if report.get("verdict_P2") not in {"IDENTICAL", "DIFFERS"}:
        errors.append(f"invalid replay verdict: {report.get('verdict_P2')!r}")
    if not isinstance(forcing, dict):
        errors.append("missing forcing-log summary")
    elif forcing.get("calls") != report.get("passes_replayed") or forcing.get("forced_rows") != report.get("rows_compared"):
        errors.append("forcing calls/rows do not match replayed passes/rows")
    return report if isinstance(report, dict) else None, errors


def compiler_choice_summary(run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Retain a compact, non-cache summary of recorded compiler choices."""
    try:
        artefacts = read_json(run_dir / "artefacts.json")
    except AuditError as exc:
        return None, [str(exc)]
    if not isinstance(artefacts, dict):
        return None, ["artefacts.json is not an object"]
    best = artefacts.get("best_configs")
    if not isinstance(best, dict):
        return None, ["artefacts.json lacks best_configs"]
    choices: dict[str, int] = {}
    for value in best.values():
        if not isinstance(value, dict):
            continue
        # File paths and cache hashes remain private; kernel-launch choices do not.
        choice = {key: value.get(key) for key in ("XBLOCK", "R0_BLOCK", "RBLOCK", "num_warps", "num_stages") if key in value}
        signature = json.dumps(choice, sort_keys=True, separators=(",", ":"))
        choices[signature] = choices.get(signature, 0) + 1
    return {
        "best_config_entries": len(best),
        "best_config_choice_counts": choices,
        "triton_kernel_families": artefacts.get("triton_kernel_families"),
        "extern_calls": artefacts.get("extern_calls"),
    }, []


def run_dir_for(results: Path, cell: str, phase: str) -> Path | None:
    matches = list((results / "e6").glob(f"*_factorial_{cell}_mixed"))
    if len(matches) != 1:
        return None
    return matches[0] / ("record" if phase == "record" else f"replay_{phase}")


def audit(results: Path, ack_dir: Path | None = None) -> dict[str, Any]:
    """Return an eight-cell table without treating missing live arms as failure."""
    plan = read_json(results / "plan.json")
    planned = plan.get("cells") if isinstance(plan, dict) else None
    if not isinstance(planned, list):
        raise AuditError("plan.json lacks cells")
    plan_by_label = {cell.get("label"): cell for cell in planned if isinstance(cell, dict)}
    plan_errors = []
    if plan.get("cold_repeats") != 3:
        plan_errors.append(f"plan requests {plan.get('cold_repeats')!r} cold repeats, expected 3")
    if set(plan_by_label) != {cell["label"] for cell in cells()}:
        plan_errors.append("plan does not name the full eight-cell factorial matrix")
    table = []
    all_errors = list(plan_errors)
    for cell in cells():
        label = cell["label"]
        row: dict[str, Any] = {**cell, "planned": label in plan_by_label, "phases": {}}
        if row["planned"] and plan_by_label[label] != cell:
            row["planning_error"] = "plan cell settings differ from factorial matrix"
            all_errors.append(f"{label}: {row['planning_error']}")
        for phase in PHASES:
            handoff_label = f"{label}_{phase}"
            manifest, reason = verified_manifest(results, ack_dir, handoff_label)
            phase_row: dict[str, Any] = {"ack_verified": manifest is not None}
            if manifest is None:
                phase_row["state"] = "pending_or_unverified"
                phase_row["reason"] = reason
            else:
                run_dir = run_dir_for(results, label, phase)
                errors = [] if run_dir is not None else ["cannot locate exactly one arm directory"]
                if run_dir is not None:
                    errors += validate_run(run_dir, cell, phase)
                    compiler, compiler_errors = compiler_choice_summary(run_dir)
                    errors += compiler_errors
                    if compiler is not None:
                        phase_row["compiler_choices"] = compiler
                    if phase != "record":
                        summary = run_dir.parent / f"summary_replay_{phase}.json"
                        summary_data, forcing_errors = forcing_check(summary)
                        errors += forcing_errors
                        if summary_data is not None:
                            phase_row["verdict_P2"] = summary_data.get("verdict_P2")
                            phase_row["rows_compared"] = summary_data.get("rows_compared")
                            phase_row["forcing"] = summary_data.get("forcing_log")
                phase_row["state"] = "verified_valid" if not errors else "verified_invalid"
                phase_row["validation_errors"] = errors
                if errors:
                    all_errors.extend(f"{handoff_label}: {error}" for error in errors)
                phase_row["manifest_sha256"] = manifest["sha256"]
            row["phases"][phase] = phase_row
        row["complete"] = row["planned"] and all(
            phase.get("state") == "verified_valid" for phase in row["phases"].values()
        )
        table.append(row)
    failures = []
    for failure_file in sorted((results / "evidence").glob("*.failure.json")):
        try:
            failures.append({"path": str(failure_file.relative_to(results)), "detail": read_json(failure_file)})
        except AuditError as exc:
            failures.append({"path": str(failure_file.relative_to(results)), "detail": {"error": str(exc)}})
    return {
        "schema": 1,
        "frozen_source": FROZEN_SOURCE,
        "status": "complete" if table and all(row["complete"] for row in table) else "partial",
        "planned_cells": len(planned),
        "plan_errors": plan_errors,
        "verified_complete_cells": sum(row["complete"] for row in table),
        "cells": table,
        "failures": failures,
        "validation_errors": all_errors,
        "conditional_benchmark_interpretation": (
            "Treat benchmark_combo_kernel as a conditional factor: its operational contrast is the c=1 strata. "
            "Rows with combo_kernels=false retain the requested setting but do not by themselves establish an active "
            "benchmark-kernel contrast."
        ),
    }


def public_member(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(part == "cache" for part in path.parts):
        return False
    return path.suffix.lower() in {".py", ".h", ".hpp", ".c", ".cc", ".cpp", ".cu", ".ll", ".ttir", ".ttgir", ".json"}


def deterministic_gzip(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as handle:
        handle.write(data)
    return output.getvalue()


def gunzip(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def source_archive(full_archive: Path) -> tuple[bytes, int]:
    """Export text source/IR only; the full cache archive remains private."""
    selected: list[tuple[str, bytes]] = []
    with tarfile.open(full_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not public_member(member.name):
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                selected.append((member.name, handle.read()))
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as output:
            for name, data in sorted(selected):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                output.addfile(info, io.BytesIO(data))
    return raw.getvalue(), len(selected)


def verify_source_archive(path: Path) -> int:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if any(not member.isfile() or not public_member(member.name) for member in members):
            raise AuditError(f"non-source member in public source archive: {path}")
        if any((member.mtime, member.uid, member.gid, member.mode) != (0, 0, 0, 0o644) for member in members):
            raise AuditError(f"non-reproducible source archive metadata: {path}")
        names = [member.name for member in members]
        if names != sorted(names) or len(names) != len(set(names)):
            raise AuditError(f"source archive entries are not sorted/unique: {path}")
    return len(members)


def export_frozen_comparator(staging: Path, source_root: Path) -> Path:
    """Export only the frozen Python modules needed by ``run_e6 --compare``."""
    listing = subprocess.check_output(
        ["git", "-C", str(source_root), "ls-tree", "-r", "--name-only", FROZEN_SOURCE, "probes/shape"],
        text=True,
    ).splitlines()
    allowed_parents = {Path("probes/shape"), Path("probes/shape/shape_hook_pkg/shape_hook")}
    selected = [Path(name) for name in listing if Path(name).suffix == ".py" and Path(name).parent in allowed_parents]
    if Path("probes/shape/run_e6.py") not in selected:
        raise AuditError("frozen source lacks run_e6.py")
    for relative in selected:
        data = subprocess.check_output(["git", "-C", str(source_root), "show", f"{FROZEN_SOURCE}:{relative.as_posix()}"])
        target = staging / "frozen_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return staging / "frozen_source" / "probes" / "shape" / "run_e6.py"


def staged_bytes(staging: Path, relative: PurePosixPath) -> bytes | None:
    path = staging / "raw" / Path(*relative.parts)
    compressed = path.with_name(path.name + ".gz")
    return gunzip(compressed) if compressed.is_file() else None


def stage_public(results: Path, ack_dir: Path | None, report: dict[str, Any], staging: Path, source_root: Path) -> None:
    """Create a portable, compressed public bundle from verified handoffs."""
    if staging.exists():
        raise AuditError(f"refuse existing staging directory: {staging}")
    staging.mkdir(parents=True)
    copied, completed, private_caches = [], [], []
    for cell in report["cells"]:
        for phase, phase_row in cell["phases"].items():
            if not phase_row.get("ack_verified"):
                continue
            label = f"{cell['label']}_{phase}"
            manifest = read_json(manifest_path(results, label))
            source_ack = ack_path(results, ack_dir, label)
            if source_ack is None:
                raise AuditError(f"verified ACK disappeared during staging: {label}")
            ack = read_json(source_ack)
            (staging / "handoff" / label).mkdir(parents=True, exist_ok=True)
            (staging / "handoff" / label / "manifest.json").write_bytes(canonical_json(manifest))
            (staging / "handoff" / label / "verified-ack.json").write_bytes(canonical_json(ack))
            item = {"label": label, "manifest_sha256": manifest["sha256"], "kind": "record" if phase == "record" else "replay"}
            for entry in manifest["files"]:
                relative = PurePosixPath(entry["path"])
                source = results.joinpath(*relative.parts)
                # Cache archives stay in the private W root. JSON cache snapshots
                # are logs, not cache contents, and remain useful provenance.
                if relative.parts and relative.parts[0] == "archives":
                    if source.suffixes[-2:] == [".tar", ".gz"]:
                        public = staging / "sources_ir" / f"{label}.sources.tar.gz"
                        data, count = source_archive(source)
                        public.parent.mkdir(parents=True, exist_ok=True)
                        public.write_bytes(data)
                        private_caches.append({"label": label, "path": relative.as_posix(), "size": entry["size"], "sha256": entry["sha256"], "public_source_archive": str(public.relative_to(staging)), "public_source_files": count, "public_source_sha256": sha256_file(public)})
                    continue
                if relative.parts and relative.parts[0] in {"e6", "logs", "evidence"} or relative.name == "plan.json":
                    target = staging / "raw" / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    compressed = target.with_name(target.name + ".gz")
                    compressed.write_bytes(deterministic_gzip(source.read_bytes()))
                    copied.append(compressed.relative_to(staging).as_posix())
            if phase != "record":
                arm = run_dir_for(results, cell["label"], phase)
                if arm is None:
                    raise AuditError(f"cannot locate replay arm for {label}")
                item.update({"arm": arm.parent.name, "replay": arm.name})
            completed.append(item)
    comparator = export_frozen_comparator(staging, source_root)
    report = {**report, "public_files": sorted(set(copied)), "frozen_comparator_sha256": sha256_file(comparator)}
    (staging / "audit.json").write_bytes(canonical_json(report))
    (staging / "completed.json").write_bytes(canonical_json({"schema": 1, "labels": completed}))
    (staging / "private_caches.json").write_bytes(canonical_json({"schema": 1, "note": "Full cache archives remain private; only hashes and source/IR derivatives are public.", "archives": private_caches}))
    (staging / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(staging)}\n" for path in sorted(staging.rglob("*")) if path.is_file())
    )


def audit_public(staging: Path) -> dict[str, Any]:
    """Verify compressed handoffs and reproduce every public replay on CPU."""
    bundle_files_verified = verify_public_sums(staging)
    staged_audit = read_json(staging / "audit.json")
    completed = read_json(staging / "completed.json").get("labels", [])
    private = read_json(staging / "private_caches.json").get("archives", [])
    verified_files, reproductions = 0, []
    for item in completed:
        label = item["label"]
        manifest = read_json(staging / "handoff" / label / "manifest.json")
        ack = read_json(staging / "handoff" / label / "verified-ack.json")
        if manifest.get("sha256") != hashlib.sha256(canonical_json({k: v for k, v in manifest.items() if k != "sha256"})).hexdigest() or ack.get("handoff_manifest_sha256") != manifest.get("sha256") or ack.get("verified_manifest_sha256") != manifest.get("sha256"):
            raise AuditError(f"public ACK/manifest validation failed: {label}")
        for row in manifest["files"]:
            relative = PurePosixPath(row["path"])
            data = staged_bytes(staging, relative)
            if data is None:
                omission = next((x for x in private if x["label"] == label and x["path"] == relative.as_posix()), None)
                if omission is None or omission["size"] != row["size"] or omission["sha256"] != row["sha256"]:
                    raise AuditError(f"public bundle lacks verified handoff file: {label}/{relative}")
                source = staging / omission["public_source_archive"]
                if sha256_file(source) != omission["public_source_sha256"] or verify_source_archive(source) != omission["public_source_files"]:
                    raise AuditError(f"public source archive verification failed: {label}")
            elif len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise AuditError(f"public handoff hash mismatch: {label}/{relative}")
            verified_files += 1
    comparator = staging / "frozen_source" / "probes" / "shape" / "run_e6.py"
    if not comparator.is_file():
        raise AuditError("public frozen comparator is absent")
    comparator_digest = staged_audit.get("frozen_comparator_sha256") if isinstance(staged_audit, dict) else None
    if not isinstance(comparator_digest, str) or sha256_file(comparator) != comparator_digest:
        raise AuditError("public frozen comparator differs from audit.json")
    with tempfile.TemporaryDirectory(prefix="factorial-public-audit-") as temporary:
        temporary = Path(temporary)
        for item in completed:
            if item["kind"] != "replay":
                continue
            arm, replay = item["arm"], item["replay"]
            target_arm = temporary / "e6" / arm
            for source in sorted((staging / "raw" / "e6" / arm).rglob("*.gz")):
                relative = source.relative_to(staging / "raw" / "e6" / arm)
                target = target_arm / relative.with_suffix("")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gunzip(source))
            command = [sys.executable, str(comparator), "--compare", str(target_arm / "record"), str(target_arm / replay)]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            subprocess.run(command, check=True, cwd=comparator.parent, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            summary_name = "summary.json" if replay == "replay" else f"summary_{replay}.json"
            generated = (target_arm / summary_name).read_bytes()
            original = staged_bytes(staging, PurePosixPath("e6") / arm / summary_name)
            if original is None or generated != original:
                raise AuditError(f"frozen comparator summary is not byte-exact: {item['label']}")
            summary = json.loads(original)
            reproductions.append({"label": item["label"], "verdict_P2": summary["verdict_P2"], "rows_compared": summary["rows_compared"], "forcing_calls": summary["forcing_log"]["calls"]})
    return {"schema": 1, "frozen_source": FROZEN_SOURCE, "bundle_files_verified": bundle_files_verified, "handoff_files_verified": verified_files, "replays_reproduced": len(reproductions), "reproductions": reproductions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results", type=Path, help="private durable raw-results root")
    mode.add_argument("--audit-public", type=Path, help="read-only portable public bundle audit")
    parser.add_argument("--ack-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new audit JSON; never inside raw results")
    parser.add_argument("--stage-public", type=Path, help="new public staging directory")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refuse existing audit output: {args.output}")
    if args.audit_public:
        if args.stage_public or args.ack_dir:
            parser.error("--audit-public does not accept --stage-public or --ack-dir")
        report = audit_public(args.audit_public.resolve())
    else:
        assert args.results is not None
        results = args.results.resolve()
        ack_dir = args.ack_dir.resolve() if args.ack_dir else None
        report = audit(results, ack_dir)
        if args.stage_public:
            stage_public(results, ack_dir, report, args.stage_public.resolve(), args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(json.dumps({"status": report.get("status", "verified"), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
