#!/usr/bin/env python3
"""Diagnostic reproducer for vLLM issue #56900.

The controller starts one new Python process for every requested compilation,
CUDA-graph, and model-runner setting.  A worker never shares vLLM, Inductor or
Triton cache directories with another worker.  It deliberately keeps the model
and tokenizer cache outside those directories.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


ISSUE_URL = "https://github.com/vllm-project/vllm/issues/56900"
MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
MAX_TOKENS = 32
REPEATS = 2

# This is the posted batch exactly: eight distinct raw prompts followed by the
# same eight prompts in the same order.  Do not apply the model chat template.
SHORT = [
    "The verifier re-runs a sampled computation and compares hashes.",
    "Floating-point addition is not associative, so",
    "def fibonacci(n):",
    "In 1854 the",
]
LONG = [
    " ".join(["The verifier records the batch shape at every step and replays it later."] * k)
    for k in (3, 6, 9, 12)
]
PROMPTS = [prompt for pair in zip(SHORT, LONG) for prompt in pair] * 2


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def cache_snapshot(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "root": str(root),
        "file_count": len(files),
        "files": [str(path.relative_to(root)) for path in files],
    }


def module_provenance() -> dict[str, Any]:
    """Collect imports independently so a startup failure remains diagnosable."""
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    modules: dict[str, Any] = {}
    for distribution in ("torch", "triton", "vllm"):
        try:
            package = importlib.metadata.distribution(distribution)
            installed = package.version
            record = package.locate_file(
                next((file for file in package.files or () if str(file).endswith("RECORD")), "")
            )
            record_sha256 = sha256_bytes(record.read_bytes()) if record.is_file() else None
            direct_url = package.read_text("direct_url.json")
            location = str(package.locate_file(""))
        except importlib.metadata.PackageNotFoundError:
            installed, record_sha256, direct_url, location = None, None, None, None
        try:
            module = importlib.import_module(distribution)
            modules[distribution] = module
            import_error = None
        except BaseException as error:
            module = None
            import_error = {"type": type(error).__name__, "message": str(error)}
        result[distribution] = {
            "distribution_version": installed,
            "module_version": getattr(module, "__version__", None),
            "module_file": getattr(module, "__file__", None),
            "module_file_sha256": sha256_bytes(Path(module.__file__).read_bytes())
            if module is not None and getattr(module, "__file__", None) and Path(module.__file__).is_file()
            else None,
            "import_error": import_error,
            "distribution_location": location,
            "record_sha256": record_sha256,
            "direct_url": direct_url,
        }
    for distribution in ("tokenspeed-triton",):
        try:
            package = importlib.metadata.distribution(distribution)
            record = package.locate_file(
                next((file for file in package.files or () if str(file).endswith("RECORD")), "")
            )
            result[distribution] = {
                "distribution_version": package.version,
                "distribution_location": str(package.locate_file("")),
                "record_sha256": sha256_bytes(record.read_bytes()) if record.is_file() else None,
                "direct_url": package.read_text("direct_url.json"),
            }
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = {"distribution_version": None}
    result["triton_package_providers"] = importlib.metadata.packages_distributions().get("triton", [])
    torch = modules.get("torch")
    if torch is not None:
        torch_details: dict[str, Any] = {
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "torch_config": torch._C._show_config(),
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            torch_details["cuda_device"] = {
                "name": properties.name,
                "total_memory": properties.total_memory,
                "major": properties.major,
                "minor": properties.minor,
                "multi_processor_count": properties.multi_processor_count,
                "uuid": getattr(properties, "uuid", None),
            }
        result["torch"].update(torch_details)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        result["nvidia_smi"] = {"returncode": smi.returncode, "stdout": smi.stdout, "stderr": smi.stderr}
    except OSError as error:
        result["nvidia_smi"] = {"error": str(error)}
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        capture_output=True,
        check=False,
    )
    result["pip_freeze"] = {
        "returncode": pip_freeze.returncode,
        "stdout": pip_freeze.stdout,
        "stderr": pip_freeze.stderr,
    }
    return result


def backend_lines(*log_paths: Path) -> list[str]:
    """Keep only automatic runner/backend selection evidence from a worker log."""
    phrases = (
        "Using V2 Model Runner",
        "attention backend",
        "Using FlashAttention version",
        "MoE backend",
        "MoEPrepareAndFinalize",
        "TritonExperts",
    )
    raw_lines = [
        line
        for log_path in log_paths
        if log_path.exists()
        for line in log_path.read_text(errors="replace").splitlines()
        if any(word in line for word in phrases)
    ]
    return [line.rsplit("] ", 1)[-1] for line in raw_lines]


def loaded_cuda_libraries() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    names = ("libcuda", "libcudart", "libcudnn", "libnvrtc", "libnvidia", "libtorch", "libtriton")
    return sorted(
        {
            fields[-1]
            for line in maps.read_text(errors="replace").splitlines()
            if (fields := line.split()) and fields[-1].startswith("/") and any(name in fields[-1] for name in names)
        }
    )


def serialise_config(value: Any) -> Any:
    """Turn vLLM's nested dataclass configuration into reviewable JSON."""
    if dataclasses.is_dataclass(value):
        return {field.name: serialise_config(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): serialise_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialise_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    return repr(value)


def cycle_period(token_ids: list[int], tail: int = 16, max_period: int = 3) -> int | None:
    tail_ids = token_ids[-tail:]
    for period in range(1, max_period + 1):
        if len(tail_ids) > period and all(
            tail_ids[index] == tail_ids[index + period]
            for index in range(len(tail_ids) - period)
        ):
            return period
    return None


def tokens_identical(left: list[list[int]], right: list[list[int]]) -> bool:
    return left == right


def serialise_logprobs(logprobs: Any) -> list[dict[str, Any]] | None:
    if logprobs is None:
        return None
    serialised = []
    for candidates in logprobs:
        values = []
        if candidates is not None:
            for token_id, candidate in candidates.items():
                values.append(
                    {
                        "token_id": token_id,
                        "logprob": candidate.logprob,
                        "rank": candidate.rank,
                        "decoded_token": candidate.decoded_token,
                    }
                )
        values.sort(key=lambda item: item["logprob"], reverse=True)
        serialised.append(
            {
                "candidates": values,
                "top1_margin": values[0]["logprob"] - values[1]["logprob"]
                if len(values) > 1
                else None,
            }
        )
    return serialised


def output_summary(outputs: list[Any]) -> list[dict[str, Any]]:
    if len(outputs) != len(PROMPTS):
        raise RuntimeError(f"expected {len(PROMPTS)} outputs, got {len(outputs)}")
    result = []
    for index, output in enumerate(outputs):
        generated = output.outputs[0]
        token_ids = list(generated.token_ids)
        if len(token_ids) != MAX_TOKENS:
            raise RuntimeError(f"prompt {index}: expected {MAX_TOKENS} generated tokens, got {len(token_ids)}")
        result.append(
            {
                "index": index,
                "prompt_sha256": sha256_bytes(PROMPTS[index].encode()),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_bytes(json.dumps(token_ids).encode()),
                "text": generated.text,
                "cycle_period_last_16": cycle_period(token_ids),
                "logprobs": serialise_logprobs(generated.logprobs),
            }
        )
    return result


def result_metrics(repeats: list[list[dict[str, Any]]]) -> dict[str, Any]:
    first = [item["token_ids"] for item in repeats[0]]
    second = [item["token_ids"] for item in repeats[1]]
    duplicate_pairs = [first[index] == first[index + 8] for index in range(8)]
    return {
        "repeat_token_identical": tokens_identical(first, second),
        "duplicate_pairs_agree": sum(duplicate_pairs),
        "duplicate_pairs": duplicate_pairs,
        "short_cycles": sum(item["cycle_period_last_16"] is not None for item in repeats[0]),
    }


def requested_graph_mode(compile_mode: str, graph_setting: str) -> str:
    if graph_setting == "off":
        return "NONE"
    return "FULL_AND_PIECEWISE" if compile_mode == "on" else "FULL"


def vllm_version_matches(actual: str | None, expected: str) -> bool:
    """Allow a wheel build suffix only when the requested version omits one."""
    if actual is None:
        return False
    return actual == expected if "+" in expected else actual.split("+", 1)[0] == expected


def worker(arguments: argparse.Namespace) -> None:
    run_dir = arguments.run_dir.resolve()
    cache_root = run_dir / "cache"
    if run_dir.exists():
        raise SystemExit(f"worker output already exists: {run_dir}")
    cache_root.mkdir(parents=True)
    if cache_snapshot(cache_root)["file_count"]:
        raise SystemExit(f"cache root is not empty: {cache_root}")

    # These are process-local and set before importing vLLM.  HF_HOME is
    # intentionally not changed: weights and tokenizer must remain the exact
    # pinned revision already available to the chosen environment.
    inherited_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_MODULE_LOADING",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TORCHINDUCTOR_COMPILE_THREADS",
            "TORCHINDUCTOR_DETERMINISTIC",
            "CUBLAS_WORKSPACE_CONFIG",
            "NCCL_NVLS_ENABLE",
            "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            "VLLM_PLUGINS",
            "VLLM_DISABLE_COMPILE_CACHE",
            "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "PATH",
        )
    }
    environment = {
        "VLLM_PLUGINS": "",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1" if arguments.runner == "v2" else "0",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    if arguments.debug_dump:
        environment["VLLM_DEBUG_DUMP_PATH"] = str(run_dir / "vllm_debug_dump")
    else:
        os.environ.pop("VLLM_DEBUG_DUMP_PATH", None)
    os.environ.update(environment)

    try:
        write_json(
            run_dir / "environment.json",
            {"set_by_reproducer": environment, "inherited_relevant": inherited_environment},
        )
        write_json(run_dir / "cache_before.json", cache_snapshot(cache_root))
        provenance = module_provenance()
        write_json(run_dir / "provenance.json", provenance)
        torch_info = provenance.get("torch", {})
        if torch_info.get("import_error"):
            raise RuntimeError(f"torch import failed: {torch_info['import_error']}")
        if not torch_info["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; this diagnostic requires a CUDA GPU")

        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.collect_env import get_pretty_env_info
        from vllm import envs, plugins
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter

        (run_dir / "collect_env.txt").write_text(get_pretty_env_info())
        if not vllm_version_matches(provenance["vllm"]["module_version"], arguments.expected_vllm):
            raise RuntimeError(
                f"expected vLLM {arguments.expected_vllm}, got {provenance['vllm']['module_version']}"
            )
        provenance["fused_moe_router_select_experts_file"] = __import__(
            FusedMoERouter.__module__, fromlist=["__file__"]
        ).__file__
        write_json(run_dir / "provenance.json", provenance)
        graph_setting = arguments.graphs[0]
        if len(arguments.graphs) != 1:
            raise RuntimeError("a worker must receive exactly one graph setting")
        graph_mode = requested_graph_mode(arguments.compile, graph_setting)
        compilation = {"mode": 3 if arguments.compile == "on" else 0, "cudagraph_mode": graph_mode}
        llm = LLM(
            model=MODEL,
            tensor_parallel_size=1,
            seed=0,
            max_model_len=2048,
            enable_prefix_caching=False,
            enforce_eager=False,
            compilation_config=compilation,
            revision=REVISION,
            tokenizer_revision=REVISION,
        )
        config = llm.llm_engine.vllm_config
        write_json(
            run_dir / "resolved_config.json",
            {
                "requested": {
                    "compile": arguments.compile,
                    "graphs": graph_setting,
                    "runner": arguments.runner,
                    "compilation_config": compilation,
                },
                "use_v2_model_runner": config.use_v2_model_runner,
                "compilation_config": serialise_config(config.compilation_config),
                "model_config": serialise_config(config.model_config),
                "parallel_config": serialise_config(config.parallel_config),
                "attention_config": serialise_config(config.attention_config),
                "kernel_config": serialise_config(config.kernel_config),
            },
        )
        expected_mode = 3 if arguments.compile == "on" else 0
        resolved_mode = config.compilation_config.mode.value
        resolved_graph_mode = config.compilation_config.cudagraph_mode.name
        resolved = {
            "runner_matches_request": config.use_v2_model_runner == (arguments.runner == "v2"),
            "mode_matches_request": resolved_mode == expected_mode,
            "graph_mode_matches_request": resolved_graph_mode == graph_mode,
        }
        write_json(run_dir / "resolved_assertions.json", resolved)
        if not all(resolved.values()):
            raise RuntimeError(f"resolved configuration does not match request: {resolved}")

        tokenizer = llm.get_tokenizer()
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in PROMPTS]
        batch = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids]
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            logprobs=5,
            ignore_eos=True,
        )
        repeats = []
        for _ in range(REPEATS):
            repeats.append(output_summary(llm.generate(batch, sampling, use_tqdm=False)))
        write_json(
            run_dir / "result.json",
            {
                "issue": ISSUE_URL,
                "model": MODEL,
                "revision": REVISION,
                "expected_vllm": arguments.expected_vllm,
                "prompts": PROMPTS,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_ids_sha256": [
                    sha256_bytes(json.dumps(token_ids).encode()) for token_ids in prompt_token_ids
                ],
                "sampling": {
                    "temperature": 0.0,
                    "max_tokens": MAX_TOKENS,
                    "logprobs": 5,
                    "ignore_eos": True,
                },
                "repeats": repeats,
                "metrics": result_metrics(repeats),
            },
        )
        write_json(
            run_dir / "plugin_state.json",
            {
                "vllm_plugins": envs.VLLM_PLUGINS,
                "plugins_loaded": plugins.plugins_loaded,
                "entry_points": {
                    group: [
                        {"name": entry.name, "value": entry.value}
                        for entry in importlib.metadata.entry_points(group=group)
                    ]
                    for group in (
                        "vllm.general_plugins",
                        "vllm.io_processor_plugins",
                        "vllm.platform_plugins",
                        "vllm.stat_logger_plugins",
                        "vllm.endpoint_plugins",
                    )
                },
            },
        )
        write_json(run_dir / "loaded_cuda_libraries.json", {"paths": loaded_cuda_libraries()})
        write_json(run_dir / "cache_after.json", cache_snapshot(cache_root))
    except BaseException as error:
        write_json(
            run_dir / "failure.json",
            {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        write_json(run_dir / "cache_after.json", cache_snapshot(cache_root))
        raise


def compare_runs(output_dir: Path, completed: list[dict[str, Any]]) -> dict[str, Any]:
    successful = {}
    for item in completed:
        result_file = output_dir / item["name"] / "result.json"
        failure_file = output_dir / item["name"] / "failure.json"
        if item["returncode"] == 0 and result_file.exists() and not failure_file.exists():
            result = json.loads(result_file.read_text())
            successful[(item["compile"], item["graphs"], item["runner"])] = result

    comparisons = []
    for runner in ("v1", "v2"):
        for graphs in ("off", "on"):
            compiled = successful.get(("on", graphs, runner))
            eager = successful.get(("off", graphs, runner))
            if compiled is None or eager is None:
                continue
            compiled_status = next(
                item
                for item in completed
                if (item["compile"], item["graphs"], item["runner"])
                == ("on", graphs, runner)
            )
            eager_status = next(
                item
                for item in completed
                if (item["compile"], item["graphs"], item["runner"])
                == ("off", graphs, runner)
            )
            compiled_tokens = [item["token_ids"] for item in compiled["repeats"][0]]
            eager_tokens = [item["token_ids"] for item in eager["repeats"][0]]
            first_difference = []
            for index, (left, right) in enumerate(zip(compiled_tokens, eager_tokens)):
                difference = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
                if difference is None and len(left) != len(right):
                    difference = min(len(left), len(right))
                first_difference.append(difference)
            comparisons.append(
                {
                    "runner": runner,
                    "graphs": graphs,
                    "compiled_backend_lines": compiled_status["backend_lines"],
                    "eager_backend_lines": eager_status["backend_lines"],
                    "backends_held_fixed": bool(compiled_status["backend_lines"])
                    and compiled_status["backend_lines"] == eager_status["backend_lines"],
                    "confounded_by_backend_selection": not (
                        bool(compiled_status["backend_lines"])
                        and compiled_status["backend_lines"] == eager_status["backend_lines"]
                    ),
                    "compiled_vs_eager_identical_prompts": sum(
                        left == right for left, right in zip(compiled_tokens, eager_tokens)
                    ),
                    "first_difference_token_index": first_difference,
                }
            )
    return {"completed": completed, "compiled_vs_eager": comparisons}


def persist_status(output_dir: Path, completed: list[dict[str, Any]]) -> None:
    report = compare_runs(output_dir, completed)
    write_json(output_dir / "status.json", report)
    write_json(output_dir / "comparison.json", report)


def stop_worker_group(process: subprocess.Popen[str], stderr_file: Any) -> None:
    """Terminate and reap a worker group, including a child engine process."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        stderr_file.write("worker ignored SIGTERM; sending SIGKILL to process group\n")
        stderr_file.flush()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_worker(
    command: list[str], stdout_path: Path, stderr_path: Path, timeout_seconds: int
) -> tuple[int, bool]:
    """Run a worker in its own process group and retain logs while it runs."""
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            stderr_file.write(f"\nworker exceeded timeout of {timeout_seconds} seconds; terminating process group\n")
            stderr_file.flush()
            stop_worker_group(process, stderr_file)
            return 124, True
        except BaseException:
            stderr_file.write("\ncontroller interrupted; terminating worker process group\n")
            stderr_file.flush()
            try:
                stop_worker_group(process, stderr_file)
            finally:
                raise


def controller_sigterm_handler(signum: int, frame: Any) -> None:
    """Turn supervisor SIGTERM into a handled controller interruption."""
    del frame
    raise SystemExit(128 + signum)


def controller(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    requested = [
        (compile_mode, graph_mode, runner)
        for graph_mode in arguments.graphs
        for runner in arguments.runners
        for compile_mode in arguments.compiles
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "issue": ISSUE_URL,
            "model": MODEL,
            "revision": REVISION,
            "expected_vllm": arguments.expected_vllm,
            "debug_dump": arguments.debug_dump,
            "requested": [
                {"compile": compile_mode, "graphs": graph_mode, "runner": runner}
                for compile_mode, graph_mode, runner in requested
            ],
            "controller_python": sys.executable,
            "command": sys.argv,
        },
    )
    completed = []
    previous_sigterm_handler = signal.signal(signal.SIGTERM, controller_sigterm_handler)
    try:
        for compile_mode, graph_mode, runner in requested:
            name = f"compile-{compile_mode}_graphs-{graph_mode}_runner-{runner}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--compile",
                compile_mode,
                "--graphs",
                graph_mode,
                "--runner",
                runner,
                "--run-dir",
                str(output_dir / name),
                "--expected-vllm",
                arguments.expected_vllm,
            ]
            if arguments.debug_dump:
                command.append("--debug-dump")
            returncode, timed_out = run_worker(
                command,
                output_dir / f"{name}.stdout.log",
                output_dir / f"{name}.stderr.log",
                arguments.timeout_seconds,
            )
            stdout_path = output_dir / f"{name}.stdout.log"
            stderr_path = output_dir / f"{name}.stderr.log"
            completed.append(
                {
                    "name": name,
                    "compile": compile_mode,
                    "graphs": graph_mode,
                    "runner": runner,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "backend_lines": backend_lines(stdout_path, stderr_path),
                }
            )
            persist_status(output_dir, completed)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    failures = [item for item in completed if item["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} worker(s) failed; see failure.json and *.stderr.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help="worker-only output directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"moe_compile_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"),
        help="new directory for controller results",
    )
    parser.add_argument("--compile", choices=("on", "off"), default="on")
    parser.add_argument("--graphs", choices=("on", "off"), nargs="+", default=("off",))
    parser.add_argument("--runner", choices=("v1", "v2"), default="v2")
    parser.add_argument("--runners", choices=("v1", "v2"), nargs="+", default=("v2",))
    parser.add_argument("--compiles", choices=("on", "off"), nargs="+", default=("on", "off"))
    parser.add_argument(
        "--debug-dump", action="store_true",
        help="enable vLLM's depyf debug instrumentation (may be incompatible with torch); DEBUG logs and compiler caches are always retained",
    )
    parser.add_argument(
        "--expected-vllm",
        default="0.28.0",
        help="required imported vLLM version; prevents an accidental release change",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="maximum wall time for each isolated worker process",
    )
    arguments = parser.parse_args()
    if arguments.worker and arguments.run_dir is None:
        parser.error("--worker requires --run-dir")
    if arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        controller(arguments)


if __name__ == "__main__":
    main()
