#!/usr/bin/env python3
"""Fail-closed runner for a fixed FlashInfer TP=2 collective experiment.

The five fresh processes are: a new record, a warm replay using its compiler
cache, and three cold replays.  Every process explicitly selects one
FlashInfer workspace backend and must produce valid per-rank dispatch traces
before its evidence can be acknowledged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SCHEMA = 1
EXPECTED_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
EXPECTED_VLLM = "98dff2a81d747d1dba01a47f939f48c3526d4206"
JOBS_SHA256 = "7f43744c69b8f42e6010b34f4abf5122f0a026f247dfe184188bff931ac767e4"
EXPECTED_VLLM_FILES = {
    "distributed/device_communicators/cuda_communicator.py": "03ece681b0bef349cb78dabc74585efea2cc160ae28bd711c81dfd807ca8dff5",
    "distributed/device_communicators/flashinfer_all_reduce.py": "607a9090f141afab59bbac73d2f6d06f557b8b13e150c2af9bc06d66d2ef3825",
}
CACHE_VARS = {
    "CUDA_CACHE_PATH": "cuda_jit",
    "DG_JIT_CACHE_DIR": "deep_gemm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer-workspace",
    "TMPDIR": "tmp",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton",
    "VLLM_CACHE_ROOT": "vllm",
    "XDG_CACHE_HOME": "xdg",
}


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path: Path) -> Path:
    if not path.expanduser().is_absolute():
        raise GateError(f"path must be absolute: {path}")
    return path.expanduser().resolve()


def tree_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise GateError(f"missing manifest root: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GateError(f"symlink is forbidden in evidence: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {"schema": SCHEMA, "files": files}
    return {**payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()}


def source_gate(source: Path) -> None:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if (head, status) != (EXPECTED_SOURCE, ""):
        raise GateError(f"source gate failed: head={head} dirty={bool(status)}")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise GateError("installed vllm package is unavailable")
    package = Path(spec.origin).resolve().parent
    actual = {
        relative: sha256(package / relative)
        for relative in EXPECTED_VLLM_FILES
    }
    if actual != EXPECTED_VLLM_FILES:
        raise GateError(f"installed vllm collective source bytes differ: {actual}")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise GateError("installed vllm distribution metadata is unavailable") from exc
    if version != "0.29.0":
        raise GateError(f"installed vllm version {version} != 0.29.0")
    hook = importlib.util.find_spec("shape_hook")
    expected_hook = source / "probes/shape/shape_hook_pkg/shape_hook/__init__.py"
    if hook is None or hook.origin is None or Path(hook.origin).resolve() != expected_hook.resolve():
        raise GateError("installed shape_hook does not resolve into the frozen source tree")


def cache_env(base: Path) -> dict[str, str]:
    return {name: str(base / child) for name, child in CACHE_VARS.items()}


def prepare_cold_cache(base: Path) -> None:
    if base.exists():
        raise GateError(f"cold cache already exists: {base}")
    base.mkdir(parents=True)
    for path in cache_env(base).values():
        Path(path).mkdir()


def cache_snapshot(base: Path) -> dict[str, Any]:
    roots = []
    for name, value in sorted(cache_env(base).items()):
        root = Path(value)
        if root.is_symlink() or not root.is_dir():
            raise GateError(f"cache root is absent or unsafe: {name}={root}")
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise GateError(f"cache symlink is forbidden: {path}")
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        roots.append({"name": name, "path": value, "file_count": len(files), "files": files})
    payload = {"schema": SCHEMA, "roots": roots}
    return {**payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()}


def terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_process(command: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0 and process.poll() is None:
                    timed_out = True
                    terminate_group(process)
                events = selector.select(max(0, min(1, remaining)) if process.poll() is None else 0.1)
                if not events and process.poll() is not None:
                    break
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        text = chunk.decode(errors="replace")
                        handle.write(text)
                        handle.flush()
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    else:
                        selector.unregister(key.fileobj)
            process.wait()
        finally:
            selector.close()
            terminate_group(process)
            process.stdout.close()
        handle.write(f"RUNNER_EXIT rc={124 if timed_out else process.returncode}\n")
        return 124 if timed_out else int(process.returncode or 0)


def arm_name(backend: str) -> str:
    return (
        "Qwen_Qwen2.5-7B-Instruct_tp2_none_compile_v2_graphs1_prefix0_"
        f"fixed_collective_{backend}"
    )


def label(backend: str, name: str) -> str:
    return f"TP2_{backend}_{name}"


def runner_files(here: Path) -> dict[str, Path]:
    return {
        "tp2_runner.py": here / "tp2_runner.py",
        "tp2_jobs.json": here / "tp2_jobs.json",
        "tp2_validate_trace.py": here / "tp2_validate_trace.py",
        "tp2_smoke.py": here / "tp2_smoke.py",
        "tp2_collective_hook": here / "tp2_collective_hook",
    }


def initialise(args: argparse.Namespace) -> int:
    source, results = absolute(args.source_root), absolute(args.results_root)
    source_gate(source)
    if sha256(args.jobs) != JOBS_SHA256:
        raise GateError("job plan digest is not reviewed")
    if results.exists():
        raise GateError(f"results root already exists: {results}")
    results.mkdir(parents=True)
    retained = results / "runner"
    retained.mkdir()
    here = Path(__file__).resolve().parent
    for name, path in runner_files(here).items():
        target = retained / name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    state = {
        "schema": SCHEMA,
        "source": EXPECTED_SOURCE,
        "vllm_source": EXPECTED_VLLM,
        "backend": args.backend,
        "completed": [],
        "record_manifest": None,
        "runner_manifest": tree_manifest(retained),
    }
    write_json(results / "state.json", state)
    print(f"INITIALISED retained_runner={retained / 'tp2_runner.py'}")
    return 0


def verify_runner(results: Path, state: dict[str, Any]) -> Path:
    retained = results / "runner"
    if tree_manifest(retained) != state["runner_manifest"]:
        raise GateError("retained runner bundle changed")
    running = Path(__file__).resolve()
    expected = retained / "tp2_runner.py"
    if running != expected or sha256(running) != sha256(expected):
        raise GateError(f"execute the retained runner copy: {expected}")
    return retained


def verify_plugin(retained: Path) -> None:
    plugin_root = retained / "tp2_collective_hook"
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    matches = [
        item
        for item in importlib.metadata.entry_points(group="vllm.general_plugins")
        if item.name == "tp2_collective_trace"
    ]
    if len(matches) != 1:
        raise GateError(f"expected one installed tp2_collective_trace plugin, found {len(matches)}")
    function = matches[0].load()
    module = sys.modules[function.__module__]
    expected = retained / "tp2_collective_hook/tp2_collective_hook/__init__.py"
    if Path(module.__file__).resolve() != expected.resolve():
        raise GateError(f"collective plugin is not installed from retained bundle: {module.__file__}")


def verify_record(results: Path, state: dict[str, Any]) -> Path:
    record = results / "e6" / arm_name(state["backend"]) / "record"
    actual = tree_manifest(record)
    if actual != state.get("record_manifest"):
        raise GateError("new causal record changed after its acknowledgement")
    return record


def manifest_handoff(results: Path, arm_label: str, targets: list[Path]) -> Path:
    files = []
    seen = set()
    for target in targets:
        candidates = [target] if target.is_file() else sorted(path for path in target.rglob("*") if path.is_file())
        for path in candidates:
            relative = path.relative_to(results).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    files.sort(key=lambda row: row["path"])
    payload = {"schema": SCHEMA, "label": arm_label, "files": files}
    output = results / "handoff" / arm_label / "manifest.json"
    write_json(output, {**payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()})
    return output


def wait_ack(ack_dir: Path, arm_label: str, manifest: Path, timeout: int, deadline: int) -> None:
    path = ack_dir / f"{arm_label}.json"
    if path.exists():
        raise GateError(f"stale ACK exists: {path}")
    expected = load_json(manifest)["sha256"]
    print(f"HANDOFF_READY label={arm_label} manifest={manifest} sha256={expected}", flush=True)
    end = min(time.time() + timeout, deadline)
    while time.time() < end:
        if path.is_file():
            value = load_json(path)
            if value != {
                "schema": SCHEMA,
                "label": arm_label,
                "handoff_manifest_sha256": expected,
                "verified_manifest_sha256": expected,
            }:
                raise GateError(f"invalid ACK: {path}")
            return
        time.sleep(2)
    raise GateError(f"ACK timeout: {arm_label}")


def verify_ack(ack_dir: Path, arm_label: str, manifest: Path) -> None:
    expected = load_json(manifest)["sha256"]
    value = load_json(ack_dir / f"{arm_label}.json")
    if value != {
        "schema": SCHEMA,
        "label": arm_label,
        "handoff_manifest_sha256": expected,
        "verified_manifest_sha256": expected,
    }:
        raise GateError(f"completed arm has invalid ACK: {arm_label}")


def run(args: argparse.Namespace) -> int:
    source, results = absolute(args.source_root), absolute(args.results_root)
    cache_root, ack_dir = absolute(args.cache_root), absolute(args.ack_dir)
    source_gate(source)
    state = load_json(results / "state.json")
    if state.get("backend") != args.backend:
        raise GateError("runner backend differs from initialised backend")
    retained = verify_runner(results, state)
    verify_plugin(retained)
    jobs = load_json(retained / "tp2_jobs.json")
    if sha256(retained / "tp2_jobs.json") != JOBS_SHA256:
        raise GateError("retained job plan changed")
    if not ack_dir.is_dir():
        raise GateError(f"ACK directory is missing: {ack_dir}")
    if (results / "STOP").exists():
        raise GateError("STOP exists")
    logs, evidence = results / "logs", results / "evidence"
    logs.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)
    base_env = os.environ.copy()
    plugin_root = str(retained / "tp2_collective_hook")
    prior_pythonpath = base_env.get("PYTHONPATH")
    base_env["PYTHONPATH"] = (
        plugin_root if not prior_pythonpath else plugin_root + os.pathsep + prior_pythonpath
    )
    base_env.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND": args.backend,
            "TP2_COLLECTIVE_EXPECT_BACKEND": args.backend,
            "VLLM_PLUGINS": "shape_hook,tp2_collective_trace",
            "NCCL_NVLS_ENABLE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    smoke_command = (
        shlex.split(args.smoke_command)
        if args.smoke_command
        else [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(retained / "tp2_smoke.py"),
        ]
    )
    if run_process(smoke_command, source / "probes/shape", base_env, logs / "smoke.log", args.smoke_timeout):
        raise GateError("CUDA/NCCL smoke failed")
    for job in jobs["sequence"]:
        arm_label = label(args.backend, job["name"])
        if arm_label in state["completed"]:
            completed_manifest = results / "handoff" / arm_label / "manifest.json"
            if not completed_manifest.is_file():
                raise GateError(f"completed arm lacks manifest: {arm_label}")
            verify_ack(ack_dir, arm_label, completed_manifest)
            print(f"ARM_ALREADY_ACKED {arm_label}", flush=True)
            continue
        if time.time() + args.arm_timeout + args.ack_timeout + args.deadline_buffer >= args.deadline_epoch:
            raise GateError(f"deadline is too close to launch {arm_label}")
        cache_base = cache_root / (
            label(args.backend, "record") if job["cache"] == "warm" else arm_label
        )
        if job["cache"] == "cold":
            prepare_cold_cache(cache_base)
        elif not cache_base.is_dir():
            raise GateError("warm replay has no record cache")
        before = cache_snapshot(cache_base)
        if job["cache"] == "cold" and any(row["file_count"] for row in before["roots"]):
            raise GateError(f"cold cache is not empty: {arm_label}")
        write_json(logs / f"{arm_label}.cache_before.json", before)
        run_id = f"{arm_label}-{uuid.uuid4()}"
        trace = evidence / "collective_traces" / arm_label
        env = {**base_env, **cache_env(cache_base)}
        env.update({"TP2_COLLECTIVE_TRACE_DIR": str(trace), "TP2_COLLECTIVE_RUN_ID": run_id})
        record = results / "e6" / arm_name(args.backend) / "record"
        if job["kind"] == "record":
            command = [
                sys.executable,
                "run_e6.py",
                "--record",
                "--out",
                str(results / "e6"),
                "--model",
                jobs["model"],
                "--revision",
                jobs["revision"],
                "--tp",
                "2",
                "--cudagraph",
                "1",
                "--prefix-caching",
                "0",
                "--tag",
                f"fixed_collective_{args.backend}",
            ]
            output_dir = record
        else:
            verify_record(results, state)
            replay_name = "replay_" + job["name"]
            output_dir = record.parent / replay_name
            if output_dir.exists():
                raise GateError(f"replay output already exists: {output_dir}")
            command = [sys.executable, "run_e6.py", "--replay", str(record), "--out", str(output_dir)]
        if run_process(command, source / "probes/shape", env, logs / f"{arm_label}.run.log", args.arm_timeout):
            raise GateError(f"engine arm failed: {arm_label}")
        trace_report = evidence / f"{arm_label}.collective.json"
        validator = [
            sys.executable,
            str(retained / "tp2_validate_trace.py"),
            str(trace),
            "--expected-backend",
            args.backend,
            "--expected-run-id",
            run_id,
            "--hook-dir",
            str(output_dir / "hook"),
            "--output",
            str(trace_report),
        ]
        if run_process(validator, retained, os.environ.copy(), logs / f"{arm_label}.trace_validate.log", args.smoke_timeout):
            raise GateError(f"collective trace failed: {arm_label}")
        targets = [output_dir, trace, trace_report, logs / f"{arm_label}.run.log", logs / f"{arm_label}.trace_validate.log"]
        if job["kind"] == "record":
            state["record_manifest"] = tree_manifest(record)
            write_json(results / "immutable_record.json", state["record_manifest"])
            targets.append(results / "immutable_record.json")
        else:
            summary_name = "summary_" + output_dir.name + ".json"
            compare_log = logs / f"{arm_label}.compare.log"
            if run_process([sys.executable, "run_e6.py", "--compare", str(record), str(output_dir)], source / "probes/shape", env, compare_log, args.smoke_timeout):
                raise GateError(f"comparison failed: {arm_label}")
            summary = record.parent / summary_name
            report = load_json(summary)
            if not report.get("requirements_met") or report.get("verdict_P2") not in {"IDENTICAL", "DIFFERS"}:
                raise GateError(f"invalid comparison: {arm_label}")
            targets.extend([summary, compare_log])
        after = cache_snapshot(cache_base)
        write_json(logs / f"{arm_label}.cache_after.json", after)
        targets.extend([logs / f"{arm_label}.cache_before.json", logs / f"{arm_label}.cache_after.json"])
        snapshot = results / "handoff" / arm_label / "state_before_ack.json"
        write_json(snapshot, state)
        targets.append(snapshot)
        handoff = manifest_handoff(results, arm_label, targets)
        wait_ack(ack_dir, arm_label, handoff, args.ack_timeout, args.deadline_epoch)
        state["completed"].append(arm_label)
        write_json(results / "state.json", state)
        print(f"ARM_ACKED {arm_label}", flush=True)
    write_json(results / "COMPLETE.json", {"schema": SCHEMA, "backend": args.backend, "completed": state["completed"]})
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--results-root", type=Path, required=True)
    value.add_argument("--backend", choices=("trtllm", "mnnvl"), required=True)
    value.add_argument("--jobs", type=Path, default=Path(__file__).resolve().with_name("tp2_jobs.json"))
    value.add_argument("--cache-root", type=Path)
    value.add_argument("--ack-dir", type=Path)
    value.add_argument("--deadline-epoch", type=int)
    value.add_argument("--arm-timeout", type=int, default=1800)
    value.add_argument("--smoke-timeout", type=int, default=300)
    value.add_argument("--ack-timeout", type=int, default=900)
    value.add_argument("--deadline-buffer", type=int, default=300)
    value.add_argument("--smoke-command", help="override the default two-rank torchrun NCCL smoke")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.initialize:
            return initialise(args)
        for name in ("cache_root", "ack_dir", "deadline_epoch"):
            if getattr(args, name) is None:
                raise GateError(f"--{name.replace('_', '-')} is required with --run")
        return run(args)
    except GateError as exc:
        if args.results_root.is_absolute() and args.results_root.exists():
            write_json(args.results_root / "STOP", {"schema": SCHEMA, "reason": str(exc), "at_epoch": time.time()})
        print(f"STOP {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
