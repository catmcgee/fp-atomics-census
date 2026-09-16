#!/usr/bin/env python3
"""Fail-closed runner for the prepared C2 follow-up queues.

This program deliberately has no provider API calls.  It runs the frozen E6
driver, writes evidence to a durable results root, and waits for an operator
acknowledgement after every arm.  The acknowledgement is the boundary between
the pod and the durable evidence store: it must attest that the handoff
manifest was copied and verified before this process will launch another arm.

Use ``--initialize`` after staging the historical ``record/`` directories, then
use ``--run`` with one prepared jobs file.  Both modes are CPU-only except for
the explicitly supplied smoke command and the E6 arms themselves.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
SCHEMA = 1
ALLOWED_ENV = {
    "VLLM_BATCH_INVARIANT",
    "TORCHINDUCTOR_DETERMINISTIC",
    "NCCL_NVLS_ENABLE",
}
REQUIRED_RECORD_FILES = {"run.json", "outputs.json", "schedule.json", "artefacts.json"}
REVIEWED_JOB_SHA256 = {
    "jobs_tp1.txt": "ac6dced8d0754b2b0b0311163c3b4fb5580847caae9eac3213ea5db8799371f1",
    "jobs_tp2.txt": "0a0429d0ce241a01ad4f0790188bae77b397ce4219c77b3658abbf4a2cee8dd4",
}


class GateError(RuntimeError):
    """A condition that must prevent another engine launch."""


@dataclass(frozen=True)
class Job:
    kind: str
    label: str
    cache: str
    env: dict[str, str]
    rest: str


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def absolute(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise GateError(f"path must be absolute: {path}")
    return path.resolve()


def safe_name(value: str, field: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise GateError(f"unsafe {field}: {value!r}")
    return value


def tree_manifest(root: Path) -> dict[str, Any]:
    """Digest every regular file below root, without following symlinks."""
    if not root.is_dir():
        raise GateError(f"missing directory for manifest: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GateError(f"symlink is not allowed in evidence: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {"schema": SCHEMA, "files": rows}
    return {**payload, "sha256": sha256_bytes(canonical_json(payload))}


def parse_env(text: str) -> dict[str, str]:
    if text == "-":
        return {}
    result: dict[str, str] = {}
    for item in shlex.split(text):
        if "=" not in item:
            raise GateError(f"bad environment item {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if key not in ALLOWED_ENV:
            raise GateError(f"environment {key!r} is not allowed by this reviewed queue")
        if not value:
            raise GateError(f"environment {key!r} has an empty value")
        result[key] = value
    return result


def parse_jobs(path: Path) -> list[Job]:
    jobs = []
    labels = set()
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|", 4)
        if len(fields) != 5:
            raise GateError(f"{path}:{line_number}: expected five pipe-separated fields")
        kind, label, cache, env_text, rest = (field.strip() for field in fields)
        if kind not in {"record", "replay"}:
            raise GateError(f"{path}:{line_number}: unsupported kind {kind!r}")
        safe_name(label, "label")
        if label in labels:
            raise GateError(f"{path}:{line_number}: duplicate label {label!r}")
        labels.add(label)
        if cache not in {"cold", "warm"}:
            raise GateError(f"{path}:{line_number}: unsupported cache mode {cache!r}")
        if not rest:
            raise GateError(f"{path}:{line_number}: empty arm arguments")
        tokens = shlex.split(rest)
        forbidden = {"--out", "--record", "--replay", "--compare"}
        if forbidden.intersection(tokens):
            raise GateError(f"{path}:{line_number}: runner owns E6 mode and --out arguments")
        if kind == "replay":
            if len(tokens) != 2:
                raise GateError(f"{path}:{line_number}: replay REST must be '<arm> <replay-dir>'")
            safe_name(tokens[0], "record arm")
            safe_name(tokens[1], "replay directory")
        elif "--model" not in tokens or "--revision" not in tokens or "--tag" not in tokens:
            raise GateError(f"{path}:{line_number}: record arm must name model, revision, and tag")
        jobs.append(Job(kind, label, cache, parse_env(env_text), rest))
    if not jobs:
        raise GateError(f"no jobs in {path}")
    return jobs


def reviewed_job_file(path: Path) -> None:
    expected = REVIEWED_JOB_SHA256.get(path.name)
    actual = sha256_file(path)
    if expected is None:
        raise GateError(f"job file is not one of the reviewed C2 queues: {path.name}")
    if actual != expected:
        raise GateError(f"job file digest {actual} != reviewed digest {expected}: {path}")


def cache_environment(cache_base: Path) -> dict[str, str]:
    """All enumerated compilation roots are inside one per-arm durable path."""
    base = absolute(cache_base)
    return {
        "XDG_CACHE_HOME": str(base / "xdg"),
        "TMPDIR": str(base / "tmp"),
        "VLLM_CACHE_ROOT": str(base / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(base / "inductor"),
        "TRITON_CACHE_DIR": str(base / "triton"),
        "FLASHINFER_WORKSPACE_BASE": str(base / "flashinfer-workspace"),
        "TORCH_EXTENSIONS_DIR": str(base / "torch_extensions"),
        "CUDA_CACHE_PATH": str(base / "cuda_jit"),
        "DG_JIT_CACHE_DIR": str(base / "deep_gemm"),
    }


def cache_snapshot(cache_base: Path) -> dict[str, Any]:
    roots = cache_environment(cache_base)
    rows = []
    for name, value in sorted(roots.items()):
        path = Path(value)
        files = []
        if path.exists():
            for child in sorted(path.rglob("*")):
                if child.is_symlink():
                    raise GateError(f"cache contains a symlink: {child}")
                if child.is_file():
                    files.append(
                        {
                            "path": child.relative_to(path).as_posix(),
                            "size": child.stat().st_size,
                            "sha256": sha256_file(child),
                        }
                    )
        rows.append({"name": name, "path": str(path), "files": files, "file_count": len(files)})
    payload = {"schema": SCHEMA, "roots": rows}
    return {**payload, "sha256": sha256_bytes(canonical_json(payload))}


def cache_is_empty(snapshot: dict[str, Any]) -> bool:
    return all(row["file_count"] == 0 for row in snapshot["roots"])


def cache_directories_gate(cache_base: Path) -> None:
    """Prevent cache variables, especially TMPDIR, from falling back elsewhere."""
    for name, value in cache_environment(cache_base).items():
        path = Path(value)
        if path.is_symlink() or not path.is_dir():
            raise GateError(f"cache environment directory is missing or unsafe: {name}={path}")


def source_gate(source_root: Path) -> None:
    source_root = absolute(source_root)
    try:
        head = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise GateError(f"cannot inspect source worktree: {exc}") from exc
    if head != EXPECTED_SOURCE:
        raise GateError(f"source commit {head} != frozen C2 source {EXPECTED_SOURCE}")
    if dirty:
        raise GateError("frozen source worktree is dirty")
    if not (source_root / "probes/shape/run_e6.py").is_file():
        raise GateError("frozen source has no probes/shape/run_e6.py")


def disk_gate(path: Path, reserve_gb: float) -> None:
    free = shutil.disk_usage(path).free
    reserve = int(reserve_gb * 1024**3)
    if free < reserve:
        raise GateError(f"free disk {free} bytes is below reserve {reserve} bytes")


def job_tokens(job: Job) -> list[str]:
    return shlex.split(job.rest)


def record_arm_from_job(job: Job) -> str | None:
    if job.kind != "replay":
        return None
    return job_tokens(job)[0]


def ensure_record(record_dir: Path) -> dict[str, Any]:
    if not record_dir.is_dir():
        raise GateError(f"record directory is missing: {record_dir}")
    missing = sorted(name for name in REQUIRED_RECORD_FILES if not (record_dir / name).is_file())
    if missing:
        raise GateError(f"record directory lacks required files: {record_dir}: {missing}")
    return tree_manifest(record_dir)


def state_path(results_root: Path) -> Path:
    return results_root / "runner_state.json"


def stop_path(results_root: Path) -> Path:
    return results_root / "STOP"


def load_state(results_root: Path) -> dict[str, Any]:
    path = state_path(results_root)
    if not path.is_file():
        raise GateError(f"missing initialization state: {path}; run --initialize after staging records")
    state = load_json(path)
    if state.get("schema") != SCHEMA:
        raise GateError(f"unsupported runner state schema in {path}")
    return state


def verify_runner_copy(results_root: Path, state: dict[str, Any], running_path: Path) -> None:
    """Require both the retained and executing runner to match the initialized digest."""
    relative = state.get("runner_copy")
    expected = state.get("runner_sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected, str):
        raise GateError("runner state lacks the retained runner path or SHA-256")
    retained = (results_root / relative).resolve()
    try:
        retained.relative_to(results_root.resolve())
    except ValueError as exc:
        raise GateError(f"retained runner escapes results root: {relative}") from exc
    if not retained.is_file():
        raise GateError(f"retained runner is missing: {retained}")
    retained_digest = sha256_file(retained)
    if retained_digest != expected:
        raise GateError(f"retained runner digest {retained_digest} != initialized digest {expected}")
    running_digest = sha256_file(running_path.resolve())
    if running_digest != expected:
        raise GateError(f"executing runner digest {running_digest} != initialized digest {expected}")


def save_state(results_root: Path, state: dict[str, Any]) -> None:
    state["updated_at_epoch"] = time.time()
    write_json(state_path(results_root), state)


def initialize(results_root: Path, runner_path: Path) -> None:
    results_root = absolute(results_root)
    e6 = results_root / "e6"
    if not e6.is_dir():
        raise GateError(f"staged historical records must be below {e6}")
    if state_path(results_root).exists():
        raise GateError(f"initialization already exists: {state_path(results_root)}")
    records = {}
    for record_dir in sorted(e6.glob("*/record")):
        records[record_dir.parent.name] = ensure_record(record_dir)
    if not records:
        raise GateError("no staged record directories found")
    bootstrap = results_root / "runner" / "run_followup.py"
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runner_path, bootstrap)
    runner_digest = sha256_file(bootstrap)
    state = {
        "schema": SCHEMA,
        "created_at_epoch": time.time(),
        "frozen_source": EXPECTED_SOURCE,
        "runner_copy": str(bootstrap.relative_to(results_root)),
        "runner_sha256": runner_digest,
        "records": records,
        "completed_labels": [],
    }
    write_json(results_root / "immutable_records.json", {"schema": SCHEMA, "records": records})
    save_state(results_root, state)
    print(json.dumps({"initialized": str(results_root), "records": sorted(records), "runner_sha256": runner_digest}, indent=2))


def verify_immutable_record(results_root: Path, state: dict[str, Any], arm: str) -> Path:
    record_dir = results_root / "e6" / safe_name(arm, "record arm") / "record"
    actual = ensure_record(record_dir)
    expected = state.get("records", {}).get(arm)
    if expected is None:
        raise GateError(f"record is not in immutable manifest: {arm}")
    expected_manifest = {key: value for key, value in expected.items() if key != "cache_owner"}
    if actual != expected_manifest:
        raise GateError(f"immutable record changed: {record_dir}")
    return record_dir


def _terminate_process_group(process: subprocess.Popen[bytes], grace: float = 20) -> None:
    """Terminate the complete session created for one engine command."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.1, max(0, end - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def run_process(command: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout: int) -> int:
    """Stream command output to both the durable log and the operator console."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise GateError(f"cannot start {command[0]!r}: {exc}") from exc
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        timed_out = False
        stream_open = True
        prior_handlers: dict[int, Any] = {}

        def interrupted(signum: int, _frame: Any) -> None:
            raise GateError(f"received signal {signal.Signals(signum).name} while running child process")

        if hasattr(signal, "SIGTERM"):
            for signum in (signal.SIGTERM, signal.SIGINT):
                prior_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupted)

        def emit(data: bytes, *, final: bool = False) -> None:
            text = decoder.decode(data, final=final)
            if text:
                handle.write(text)
                handle.flush()
                print(text, end="", flush=True)

        try:
            while stream_open or process.poll() is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0 and (stream_open or process.poll() is None):
                    timed_out = True
                    _terminate_process_group(process)
                events = selector.select(timeout=max(0, min(1, remaining)) if process.poll() is None else 0)
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    if chunk:
                        emit(chunk)
                    else:
                        selector.unregister(key.fileobj)
                        stream_open = False
                if process.poll() is not None and not events and stream_open:
                    # A final readiness poll drains bytes written immediately before exit.
                    events = selector.select(timeout=0.1)
                    if not events:
                        stream_open = False
                        selector.unregister(process.stdout)
                if timed_out and process.poll() is not None and not stream_open:
                    break
            emit(b"", final=True)
            process.wait()
        finally:
            for signum, handler in prior_handlers.items():
                signal.signal(signum, handler)
            selector.close()
            # A command leader can exit after leaving engine workers alive.
            # The helper is also safe and fast when its process group is gone.
            _terminate_process_group(process)
            process.stdout.close()
        if timed_out:
            handle.write(f"RUNNER_TIMEOUT seconds={timeout}\n")
            return 124
        handle.write(f"RUNNER_EXIT rc={process.returncode}\n")
        return int(process.returncode or 0)


def stop(results_root: Path, reason: str) -> None:
    payload = {"schema": SCHEMA, "reason": reason, "at_epoch": time.time()}
    write_json(stop_path(results_root), payload)
    print("STOP " + reason, file=sys.stderr)


def arm_cache_base(cache_root: Path, job: Job, state: dict[str, Any]) -> Path:
    if job.cache == "cold":
        return cache_root / "cold" / job.label
    arm = record_arm_from_job(job)
    assert arm is not None
    owner = state.get("records", {}).get(arm, {}).get("cache_owner")
    if not owner:
        raise GateError(f"warm replay {job.label} has no prior locally-recorded cache owner for {arm}")
    return cache_root / "cold" / safe_name(owner, "cache owner")


def prepare_cold_cache(cache_base: Path) -> dict[str, Any]:
    if cache_base.exists():
        raise GateError(f"cold cache path already exists; refuse to reuse it: {cache_base}")
    cache_base.mkdir(parents=True)
    for path in cache_environment(cache_base).values():
        Path(path).mkdir(parents=True, exist_ok=False)
    cache_directories_gate(cache_base)
    snapshot = cache_snapshot(cache_base)
    if not cache_is_empty(snapshot):
        raise GateError(f"new cold cache root is unexpectedly non-empty: {cache_base}")
    return snapshot


def record_result_dir(results_root: Path, before: set[Path]) -> Path:
    e6 = results_root / "e6"
    records = {path for path in e6.glob("*/record") if path.is_dir()}
    new = records - before
    if len(new) != 1:
        raise GateError(f"record arm did not create exactly one record directory: found {sorted(map(str, new))}")
    return next(iter(new))


def validate_replay_summary(results_root: Path, arm: str, replay_name: str) -> Path:
    filename = "summary.json" if replay_name == "replay" else f"summary_{replay_name}.json"
    summary = results_root / "e6" / arm / filename
    if not summary.is_file():
        raise GateError(f"frozen comparator did not write {summary}")
    report = load_json(summary)
    if report.get("verdict_P2") == "INVALID" or not report.get("requirements_met"):
        raise GateError(f"comparison is invalid for {arm}/{replay_name}: {report.get('validation_errors')}")
    return summary


def rank1_audit(source_root: Path, record_dir: Path, replay_dir: Path, rank0_summary: Path, output: Path) -> None:
    """Use the frozen comparator for rank 1 and report cross-rank observations."""
    source_shape = source_root / "probes/shape"
    sys.path.insert(0, str(source_shape))
    try:
        from run_e6 import compare_replay  # type: ignore
        from shape_common import read_hook, real_steps  # type: ignore
        rec1, rep1 = real_steps(read_hook(record_dir / "hook", 1)), real_steps(read_hook(replay_dir / "hook", 1))
        report = compare_replay(rec1, rep1)
        rank0 = load_json(rank0_summary)
        if report["validation_errors"] or report["first_mismatch"] is not None:
            raise GateError(
                f"rank-1 comparison is not valid/comparable: errors={report['validation_errors']}, "
                f"first_mismatch={report['first_mismatch']}"
            )
        if report["verdict_P2"] != rank0.get("verdict_P2") or report["rows_differing"] != rank0.get("rows_differing"):
            raise GateError("rank-1 verdict or differing-row count disagrees with rank 0")
        cross = {"h": 0, "logits_h": 0, "argmax": 0}
        for side, left, right in (("record", real_steps(read_hook(record_dir / "hook", 0)), rec1), ("replay", real_steps(read_hook(replay_dir / "hook", 0)), rep1)):
            if len(left) != len(right):
                raise GateError(f"{side} rank traces have different pass counts")
            for a, b in zip(left, right):
                if [row["req"] for row in a["requests"]] != [row["req"] for row in b["requests"]]:
                    raise GateError(f"{side} rank traces have different request order")
                for x, y in zip(a["requests"], b["requests"]):
                    if x.get("new_token_ids") != y.get("new_token_ids"):
                        raise GateError(f"{side} rank traces have different input token rows")
                    for field in cross:
                        cross[field] += int(x.get(field) != y.get(field))
        if cross["logits_h"] or cross["argmax"]:
            raise GateError(f"rank traces disagree on replicated logits/argmax outputs: {cross}")
        write_json(output, {"schema": SCHEMA, "rank0_summary": str(rank0_summary), "rank1": report, "cross_rank_differing_rows": cross})
    finally:
        sys.path.remove(str(source_shape))


def routing_audit(validator: Path, record_dir: Path, output: Path, log: Path, env: dict[str, str], timeout: int) -> None:
    command = [
        sys.executable,
        str(validator),
        str(record_dir),
        "--model", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "--revision", "ec052fda178e241c7c443468d2fa1db6618996be",
        "--output", str(output),
    ]
    if run_process(command, cwd=validator.parent, env=env, log=log, timeout=timeout) != 0:
        raise GateError(f"MoE routing audit failed for {record_dir}")


def handoff_manifest(results_root: Path, label: str, arm_dir: Path, logs: Path, archives: Path, extras: list[Path]) -> Path:
    """Make a path-relative manifest for the exact evidence that must leave the pod."""
    files: list[dict[str, Any]] = []
    targets = [arm_dir, *(logs.glob(label + ".*")), archives / label, *extras]
    for target in targets:
        if not target.exists():
            continue
        candidates = [target] if target.is_file() else sorted(path for path in target.rglob("*") if path.is_file())
        for path in candidates:
            if path.is_symlink():
                raise GateError(f"handoff evidence contains symlink: {path}")
            files.append({"path": str(path.relative_to(results_root)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    files.sort(key=lambda row: row["path"])
    payload = {"schema": SCHEMA, "label": label, "files": files}
    manifest = {**payload, "sha256": sha256_bytes(canonical_json(payload))}
    output = results_root / "handoff" / label / "manifest.json"
    write_json(output, manifest)
    instructions = output.with_name("OPERATOR_ACK.md")
    instructions.write_text(
        "Copy every path in manifest.json from RESULTS_ROOT to durable storage, verify each SHA-256, then write "
        f"ACK_DIR/{label}.json with schema=1, label={label!r}, and both handoff_manifest_sha256 and "
        "verified_manifest_sha256 equal to manifest.json's sha256. The runner will not launch another arm first.\n"
    )
    return output


def verify_ack(ack: Path, label: str, manifest: Path) -> None:
    if not ack.is_file():
        raise GateError(f"missing handoff acknowledgement for completed label {label}: {ack}")
    try:
        expected = load_json(manifest)["sha256"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise GateError(f"cannot read handoff manifest for {label}: {manifest}") from exc
    try:
        value = load_json(ack)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read handoff acknowledgement for {label}: {ack}") from exc
    if not isinstance(expected, str) or not isinstance(value, dict):
        raise GateError(f"invalid handoff acknowledgement: {ack}")
    if value.get("schema") != SCHEMA or value.get("label") != label:
        raise GateError(f"invalid handoff acknowledgement: {ack}")
    if value.get("handoff_manifest_sha256") != expected or value.get("verified_manifest_sha256") != expected:
        raise GateError(f"acknowledgement digest does not verify handoff: {ack}")


def snapshot_state_for_handoff(results_root: Path, label: str, state: dict[str, Any]) -> Path:
    """Freeze the exact runner state covered by one handoff manifest."""
    output = results_root / "handoff" / safe_name(label, "label") / "state_before_ack.json"
    if output.exists():
        raise GateError(f"handoff state snapshot already exists: {output}")
    write_json(output, state)
    return output


def wait_for_ack(ack_dir: Path, label: str, manifest: Path, timeout: int, deadline: int) -> None:
    ack = absolute(ack_dir) / f"{label}.json"
    if ack.exists():
        raise GateError(f"refuse stale acknowledgement path: {ack}")
    expected = load_json(manifest)["sha256"]
    end = min(time.time() + timeout, float(deadline))
    print(f"HANDOFF_READY label={label} manifest={manifest} sha256={expected}", flush=True)
    while time.time() < end:
        if ack.is_file():
            verify_ack(ack, label, manifest)
            return
        time.sleep(2)
    raise GateError(f"timed out waiting for durable-sync acknowledgement for {label}")


def archive_cache(cache_base: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(output), "gztar", root_dir=cache_base.parent, base_dir=cache_base.name)


def run(args: argparse.Namespace) -> int:
    results_root = absolute(args.results_root)
    source_root = absolute(args.source_root)
    cache_root = absolute(args.cache_root)
    ack_dir = absolute(args.ack_dir)
    reviewed_job_file(args.jobs)
    jobs = parse_jobs(args.jobs)
    state = load_state(results_root)
    verify_runner_copy(results_root, state, Path(__file__))
    completed = state.get("completed_labels")
    if not isinstance(completed, list) or any(not isinstance(label, str) for label in completed) or len(completed) != len(set(completed)):
        raise GateError("runner state completed_labels must be a unique list of strings")
    logs, archives = results_root / "logs", results_root / "archives"
    logs.mkdir(exist_ok=True)
    archives.mkdir(exist_ok=True)
    if not ack_dir.is_dir():
        raise GateError(f"acknowledgement directory is missing: {ack_dir}")
    if stop_path(results_root).exists():
        raise GateError(f"STOP already exists: {stop_path(results_root)}")
    source_gate(source_root)
    preflight = load_json(args.preflight_report)
    if preflight.get("compatible") is not True:
        raise GateError("preflight report is not compatible")
    if preflight.get("expected_source") != EXPECTED_SOURCE or preflight.get("actual_source") != EXPECTED_SOURCE:
        raise GateError("preflight report does not attest the frozen C2 source")
    (results_root / "evidence").mkdir(exist_ok=True)
    shutil.copy2(args.preflight_report, results_root / "evidence" / "preflight_report.json")
    base_env = os.environ.copy()
    base_env.update({"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_DISABLE_COMPILE_CACHE": "1"})
    smoke_log = logs / "smoke.log"
    if run_process(shlex.split(args.smoke_command), cwd=source_root / "probes/shape", env=base_env, log=smoke_log, timeout=args.smoke_timeout) != 0:
        raise GateError("CUDA/NCCL smoke command failed")
    for job in jobs:
        if job.label in completed:
            manifest = results_root / "handoff" / job.label / "manifest.json"
            if not manifest.is_file():
                raise GateError(f"completed label lacks handoff manifest: {job.label}")
            verify_ack(ack_dir / f"{job.label}.json", job.label, manifest)
            print(f"ARM_ALREADY_ACKED {job.label}", flush=True)
            continue
        if stop_path(results_root).exists():
            raise GateError("STOP exists before arm launch")
        now = time.time()
        if now + args.arm_timeout + args.ack_timeout + args.deadline_buffer >= args.deadline_epoch:
            raise GateError(f"deadline is too close to launch {job.label}")
        disk_gate(results_root, args.min_free_gb)
        cache_base = arm_cache_base(cache_root, job, state)
        if job.cache == "cold":
            before_cache = prepare_cold_cache(cache_base)
        else:
            if not cache_base.is_dir():
                raise GateError(f"warm cache root is unavailable: {cache_base}")
            cache_directories_gate(cache_base)
            before_cache = cache_snapshot(cache_base)
        write_json(logs / f"{job.label}.cache_before.json", before_cache)
        env = base_env.copy()
        env.update(cache_environment(cache_base))
        env.update(job.env)
        e6 = results_root / "e6"
        before_records = {path for path in e6.glob("*/record") if path.is_dir()}
        arm_dir: Path
        extras: list[Path] = [results_root / "evidence" / "preflight_report.json"]
        if job.kind == "record":
            command = [sys.executable, "run_e6.py", "--record", "--out", str(e6), "--prefix-caching", "0", *job_tokens(job)]
            rc = run_process(command, cwd=source_root / "probes/shape", env=env, log=logs / f"{job.label}.run.log", timeout=args.arm_timeout)
            if rc:
                raise GateError(f"arm {job.label} exited {rc}")
            record_dir = record_result_dir(results_root, before_records)
            ensure_record(record_dir)
            arm = record_dir.parent.name
            state["records"][arm] = {**tree_manifest(record_dir), "cache_owner": job.label}
            write_json(results_root / "immutable_records.json", {"schema": SCHEMA, "records": state["records"]})
            save_state(results_root, state)
            arm_dir = record_dir.parent
            if "routing_diag" in job.rest:
                routing_output = results_root / "evidence" / "moe_routing_record.json"
                routing_audit(args.routing_validator, record_dir, routing_output, logs / f"{job.label}.routing.log", env, args.smoke_timeout)
                extras.append(routing_output)
        else:
            arm, replay_name = job_tokens(job)
            record_dir = verify_immutable_record(results_root, state, arm)
            replay_dir = e6 / arm / replay_name
            if replay_dir.exists():
                raise GateError(f"replay output already exists: {replay_dir}")
            command = [sys.executable, "run_e6.py", "--replay", str(record_dir), "--out", str(replay_dir)]
            rc = run_process(command, cwd=source_root / "probes/shape", env=env, log=logs / f"{job.label}.run.log", timeout=args.arm_timeout)
            if rc:
                raise GateError(f"arm {job.label} exited {rc}")
            compare = [sys.executable, "run_e6.py", "--compare", str(record_dir), str(replay_dir)]
            if run_process(compare, cwd=source_root / "probes/shape", env=env, log=logs / f"{job.label}.compare.log", timeout=args.smoke_timeout):
                raise GateError(f"frozen comparator failed for {job.label}")
            summary = validate_replay_summary(results_root, arm, replay_name)
            extras.append(summary)
            if "_tp2_" in arm:
                rank_output = results_root / "evidence" / f"rank1_{job.label}.json"
                rank1_audit(source_root, record_dir, replay_dir, summary, rank_output)
                extras.append(rank_output)
            if "routing_diag" in arm:
                routing_output = results_root / "evidence" / "moe_routing_replay_warm.json"
                routing_audit(args.routing_validator, replay_dir, routing_output, logs / f"{job.label}.routing.log", env, args.smoke_timeout)
                extras.append(routing_output)
            arm_dir = e6 / arm
        after_cache = cache_snapshot(cache_base)
        write_json(logs / f"{job.label}.cache_after.json", after_cache)
        disk_gate(results_root, args.min_free_gb)
        if args.archive_full_cache:
            archive_base = archives / job.label / "cache"
            archive_cache(cache_base, archive_base)
        extras.append(snapshot_state_for_handoff(results_root, job.label, state))
        manifest = handoff_manifest(results_root, job.label, arm_dir, logs, archives, extras)
        wait_for_ack(ack_dir, job.label, manifest, args.ack_timeout, args.deadline_epoch)
        completed.append(job.label)
        save_state(results_root, state)
        print(f"ARM_ACKED {job.label}", flush=True)
    write_json(results_root / "COMPLETE.json", {"schema": SCHEMA, "completed_labels": state["completed_labels"], "at_epoch": time.time()})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--smoke-command")
    parser.add_argument("--routing-validator", type=Path, default=Path(__file__).with_name("validate_moe_routing.py"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--ack-dir", type=Path)
    parser.add_argument("--deadline-epoch", type=int)
    parser.add_argument("--arm-timeout", type=int, default=1500)
    parser.add_argument("--ack-timeout", type=int, default=1800)
    parser.add_argument("--smoke-timeout", type=int, default=300)
    parser.add_argument("--deadline-buffer", type=int, default=600)
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--archive-full-cache", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.initialize:
            initialize(args.results_root, Path(__file__).resolve())
            return 0
        required = {"source_root": args.source_root, "jobs": args.jobs, "preflight_report": args.preflight_report, "smoke_command": args.smoke_command, "cache_root": args.cache_root, "ack_dir": args.ack_dir, "deadline_epoch": args.deadline_epoch, "min_free_gb": args.min_free_gb}
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise GateError("--run requires " + ", ".join("--" + key.replace("_", "-") for key in missing))
        if args.arm_timeout <= 0 or args.ack_timeout <= 0 or args.smoke_timeout <= 0 or args.deadline_buffer < 0 or args.min_free_gb <= 0:
            raise GateError("timeouts and --min-free-gb must be positive; deadline buffer cannot be negative")
        return run(args)
    except GateError as exc:
        if args.run:
            try:
                stop(absolute(args.results_root), str(exc))
            except Exception as stop_error:  # preserve the original reason in the exit path
                print(f"also failed to write STOP: {stop_error}", file=sys.stderr)
        print(f"C2 FOLLOW-UP REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
