#!/usr/bin/env python3
"""Audit one or more issue #56900 diagnostic output directories.

The inputs are controller output directories created by
``moe_compile_reproducer.py``.  Labels are deliberately supplied by the caller
because a directory name is not reliable evidence of the imported CUDA build.

Example::

    python moe_compile_audit.py \
      --input cu129=/evidence/moe_compile_cu129 \
      --input cu130=/evidence/moe_compile_cu130 \
      --json-out /evidence/moe_compile_audit.json \
      --markdown-out /evidence/moe_compile_audit.md

The audit never turns a failed, missing, or materially confounded cell into a
model-output verdict.  It also does not call the GPU or import either vLLM
environment, so it is safe to run after the paid GPU has been released.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path, PurePath
from typing import Any


GPU_UUID_RE = re.compile(r"(?:GPU|MIG)-[A-Za-z0-9_./-]+")
BARE_UUID_RE = re.compile(
    r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}(?![0-9A-Fa-f])"
)


def read_json(path: Path) -> tuple[Any | None, str | None]:
    compressed = path.with_name(path.name + ".gz")
    if path.is_file():
        reader = path.open
    elif compressed.is_file():
        reader = lambda: gzip.open(compressed, "rt")
    else:
        return None, f"missing {path.name}"
    try:
        with reader() as handle:
            return json.load(handle), None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"unreadable {path.name}: {error}"


def read_text_or_gzip(path: Path) -> str:
    if path.is_file():
        return path.read_text(errors="replace")
    compressed = path.with_name(path.name + ".gz")
    if compressed.is_file():
        with gzip.open(compressed, "rt", errors="replace") as handle:
            return handle.read()
    return ""


def sha256_json(value: Any) -> str:
    # This intentionally matches the reproducer's json.dumps(list).encode().
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cycle_period(token_ids: list[int], tail: int = 16, max_period: int = 3) -> int | None:
    tail_ids = token_ids[-tail:]
    for period in range(1, max_period + 1):
        if len(tail_ids) > period and all(
            tail_ids[index] == tail_ids[index + period]
            for index in range(len(tail_ids) - period)
        ):
            return period
    return None


def normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(text: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" in line:
            name = line.split("==", 1)[0]
        elif " @ " in line:
            name = line.split(" @ ", 1)[0]
        else:
            name = line
        result[normalise_distribution(name)] = line
    return result


def parse_smi(text: str | None) -> list[dict[str, str]]:
    rows = []
    for line in (text or "").splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) == 4:
            rows.append(dict(zip(("index", "uuid", "name", "driver"), fields)))
    return rows


def canonical_gpu_uuid(value: Any) -> str | None:
    if value is None:
        return None
    match = GPU_UUID_RE.search(str(value))
    if match:
        return match.group(0)
    # torch may expose uuid.UUID, whose repr omits nvidia-smi's GPU- prefix.
    match = BARE_UUID_RE.search(str(value))
    return f"GPU-{match.group(0)}" if match else None


def selected_gpu(provenance: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    torch_device = provenance.get("torch", {}).get("cuda_device", {}) or {}
    smi = provenance.get("nvidia_smi", {}) or {}
    smi_rows = parse_smi(smi.get("stdout")) if smi.get("returncode") == 0 else []
    torch_uuid = canonical_gpu_uuid(torch_device.get("uuid"))
    selected = next((row for row in smi_rows if row["uuid"] == torch_uuid), None)
    method = "torch_device_uuid" if torch_uuid else None

    visible = (environment.get("inherited_relevant", {}) or {}).get("CUDA_VISIBLE_DEVICES")
    first_visible = str(visible).split(",", 1)[0].strip() if visible not in (None, "") else None
    if selected is None and first_visible:
        visible_uuid = canonical_gpu_uuid(first_visible)
        selected = next(
            (
                row
                for row in smi_rows
                if row["uuid"] == visible_uuid or row["index"] == first_visible
            ),
            None,
        )
        if selected is not None:
            method = "cuda_visible_devices"
    if selected is None and len(smi_rows) == 1:
        selected = smi_rows[0]
        method = "single_nvidia_smi_row"

    uuid = torch_uuid or (selected or {}).get("uuid")
    return {
        "uuid": uuid,
        "name": torch_device.get("name") or (selected or {}).get("name"),
        "driver": (selected or {}).get("driver"),
        "capability": [torch_device.get("major"), torch_device.get("minor")],
        "multi_processor_count": torch_device.get("multi_processor_count"),
        "total_memory": torch_device.get("total_memory"),
        "selection_method": method,
        "selection_unambiguous": bool(uuid and (selected is not None or torch_uuid)),
        "nvidia_smi_rows": smi_rows,
    }


def backend_evidence(stdout: str, stderr: str, resolved: dict[str, Any] | None) -> dict[str, Any]:
    text = stdout + "\n" + stderr

    def values(pattern: str) -> list[str]:
        return sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))

    fields = {
        "attention_backend": values(r"Using\s+([A-Z0-9_]+)\s+attention backend"),
        "flash_attention_version": values(r"Using FlashAttention version\s+([0-9]+)"),
        "unquantized_moe_backend": values(r"Using\s+(.+?)\s+Unquantized MoE backend"),
        "moe_prepare_finalize": values(r"Using\s+(MoEPrepareAndFinalize\S+)"),
        "moe_experts": values(r"Using\s+(\S+Experts)\s+MoE backend"),
    }
    requested = (resolved or {}).get("requested", {})
    use_v2 = (resolved or {}).get("use_v2_model_runner")
    runner = "v2" if use_v2 is True else "v1" if use_v2 is False else None
    fields["resolved_runner"] = runner
    fields["requested_runner"] = requested.get("runner")
    required = (
        "attention_backend",
        "unquantized_moe_backend",
        "moe_prepare_finalize",
        "moe_experts",
    )
    attention_is_flash = fields["attention_backend"] == ["FLASH_ATTN"]
    flash_version_complete = (
        len(fields["flash_attention_version"]) == 1
        if attention_is_flash
        else len(fields["flash_attention_version"]) <= 1
    )
    complete = (
        all(len(fields[name]) == 1 for name in required)
        and flash_version_complete
        and runner is not None
    )
    signature = {
        name: fields[name][0] if len(fields[name]) == 1 else None
        for name in (*required, "flash_attention_version")
    }
    return {**fields, "complete": complete, "signature": signature}


def cache_evidence(run_dir: Path, environment: dict[str, Any]) -> dict[str, Any]:
    before, before_error = read_json(run_dir / "cache_before.json")
    after, after_error = read_json(run_dir / "cache_after.json")
    configured = (environment.get("set_by_reproducer", {}) or {})
    root = (before or {}).get("root")
    paths = {
        key: configured.get(key)
        for key in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    }
    beneath_root = False
    if root and all(paths.values()):
        root_parts = PurePath(root).parts
        beneath_root = all(PurePath(path).parts[: len(root_parts)] == root_parts for path in paths.values())
    verified = bool(
        before_error is None
        and after_error is None
        and before.get("file_count") == 0
        and before.get("files") == []
        and configured.get("VLLM_DISABLE_COMPILE_CACHE") == "1"
        and len(set(paths.values())) == 3
        and beneath_root
    )
    return {
        "verified_empty_named_roots_before_import": verified,
        "scope": "VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, and TRITON_CACHE_DIR only",
        "root": root,
        "configured_roots": paths,
        "before_file_count": (before or {}).get("file_count"),
        "after_file_count": (after or {}).get("file_count"),
        "errors": [error for error in (before_error, after_error) if error],
    }


def margin_at(output: dict[str, Any], position: int) -> dict[str, Any] | None:
    logprobs = output.get("logprobs")
    if not isinstance(logprobs, list) or position >= len(logprobs):
        return None
    entry = logprobs[position] or {}
    candidates = entry.get("candidates") or []
    if not candidates:
        return {"top1_token_id": None, "top1_margin": None, "candidates": []}
    ordered = sorted(candidates, key=lambda item: item.get("logprob", float("-inf")), reverse=True)
    margin = (
        ordered[0]["logprob"] - ordered[1]["logprob"]
        if len(ordered) > 1
        else None
    )
    return {
        "top1_token_id": ordered[0].get("token_id"),
        "top1_margin": margin,
        "recorded_top1_margin": entry.get("top1_margin"),
        "candidates": ordered,
    }


def validate_result(result: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    prompts = result.get("prompts")
    prompt_ids = result.get("prompt_token_ids")
    prompt_hashes = result.get("prompt_token_ids_sha256")
    repeats = result.get("repeats")
    if not isinstance(prompts, list) or len(prompts) != 16:
        errors.append("expected 16 prompts")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != 16:
        errors.append("expected 16 prompt-token-id lists")
    if not isinstance(prompt_hashes, list) or len(prompt_hashes) != 16:
        errors.append("expected 16 prompt-token-id hashes")
    if not isinstance(repeats, list) or len(repeats) != 2:
        errors.append("expected two in-process repeats")

    duplicate_input_pairs = []
    if isinstance(prompts, list) and isinstance(prompt_ids, list) and len(prompts) == len(prompt_ids) == 16:
        duplicate_input_pairs = [
            prompts[index] == prompts[index + 8] and prompt_ids[index] == prompt_ids[index + 8]
            for index in range(8)
        ]
        if not all(duplicate_input_pairs):
            errors.append("the second eight inputs do not exactly duplicate the first eight")
    if isinstance(prompt_ids, list) and isinstance(prompt_hashes, list) and len(prompt_ids) == len(prompt_hashes):
        if [sha256_json(ids) for ids in prompt_ids] != prompt_hashes:
            errors.append("prompt-token-id hashes do not match their payloads")

    repeat_summaries = []
    token_matrices: list[list[list[int]]] = []
    if isinstance(repeats, list):
        for repeat_index, repeat in enumerate(repeats):
            if not isinstance(repeat, list) or len(repeat) != 16:
                errors.append(f"repeat {repeat_index} does not contain 16 outputs")
                continue
            matrix = []
            cycle_rows = []
            first_margins = []
            for prompt_index, output in enumerate(repeat):
                token_ids = output.get("token_ids")
                if not isinstance(token_ids, list) or len(token_ids) != 32:
                    errors.append(f"repeat {repeat_index} prompt {prompt_index}: expected 32 token ids")
                    token_ids = token_ids if isinstance(token_ids, list) else []
                matrix.append(token_ids)
                if output.get("token_ids_sha256") != sha256_json(token_ids):
                    errors.append(f"repeat {repeat_index} prompt {prompt_index}: token hash mismatch")
                if (
                    isinstance(prompts, list)
                    and len(prompts) == 16
                    and output.get("prompt_sha256")
                    != hashlib.sha256(prompts[prompt_index].encode()).hexdigest()
                ):
                    errors.append(f"repeat {repeat_index} prompt {prompt_index}: prompt hash mismatch")
                observed_cycle = cycle_period(token_ids)
                if output.get("cycle_period_last_16") != observed_cycle:
                    errors.append(f"repeat {repeat_index} prompt {prompt_index}: cycle summary mismatch")
                if observed_cycle is not None:
                    cycle_rows.append({"prompt_index": prompt_index, "period": observed_cycle})
                margin = margin_at(output, 0)
                first_margins.append(
                    {
                        "prompt_index": prompt_index,
                        "generated_token_id": token_ids[0] if token_ids else None,
                        **(margin or {"top1_token_id": None, "top1_margin": None}),
                    }
                )
                if margin and margin["top1_margin"] != margin["recorded_top1_margin"]:
                    errors.append(f"repeat {repeat_index} prompt {prompt_index}: first-token margin mismatch")
            token_matrices.append(matrix)
            duplicate_pairs = [matrix[index] == matrix[index + 8] for index in range(8)]
            repeat_summaries.append(
                {
                    "repeat": repeat_index,
                    "duplicate_pairs": duplicate_pairs,
                    "duplicate_pairs_agree": sum(duplicate_pairs),
                    "short_cycles": cycle_rows,
                    "first_token_logprob_margins": first_margins,
                }
            )

    in_process_identical = len(token_matrices) == 2 and token_matrices[0] == token_matrices[1]
    metrics = result.get("metrics") or {}
    if len(token_matrices) == 2 and metrics.get("repeat_token_identical") != in_process_identical:
        errors.append("recorded repeat-token identity does not match outputs")
    if repeat_summaries and metrics.get("duplicate_pairs") != repeat_summaries[0]["duplicate_pairs"]:
        errors.append("recorded duplicate-pair summary does not match outputs")
    if repeat_summaries and metrics.get("short_cycles") != len(repeat_summaries[0]["short_cycles"]):
        errors.append("recorded cycle count does not match outputs")

    return {
        "in_process_repeat_tokens_identical": in_process_identical,
        "duplicate_input_pairs": duplicate_input_pairs,
        "repeats": repeat_summaries,
    }, errors


@dataclass(frozen=True)
class InputRoot:
    label: str
    path: Path


def cell_name(requested: dict[str, str]) -> str:
    return (
        f"compile-{requested['compile']}_graphs-{requested['graphs']}_runner-{requested['runner']}"
    )


def status_entries(root: Path) -> dict[str, dict[str, Any]]:
    for filename in ("status.json", "comparison.json"):
        value, error = read_json(root / filename)
        if error is None and isinstance(value, dict):
            return {item.get("name"): item for item in value.get("completed", []) if item.get("name")}
    return {}


def load_cell(
    input_root: InputRoot,
    requested: dict[str, str],
    status: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    name = cell_name(requested)
    run_dir = input_root.path / name
    result, result_error = read_json(run_dir / "result.json")
    failure, failure_error = read_json(run_dir / "failure.json")
    provenance, provenance_error = read_json(run_dir / "provenance.json")
    environment, environment_error = read_json(run_dir / "environment.json")
    resolved, resolved_error = read_json(run_dir / "resolved_config.json")
    assertions, assertions_error = read_json(run_dir / "resolved_assertions.json")
    entry = status.get(name)
    stdout_path = input_root.path / f"{name}.stdout.log"
    stderr_path = input_root.path / f"{name}.stderr.log"
    stdout = read_text_or_gzip(stdout_path)
    stderr = read_text_or_gzip(stderr_path)

    artefact_state = "missing"
    if result is not None and failure is None:
        artefact_state = "success"
    elif failure is not None and result is None:
        artefact_state = "failed"
    elif failure is not None and result is not None:
        artefact_state = "invalid_both_result_and_failure"
    if entry is not None and entry.get("returncode") not in (None, 0):
        artefact_state = "failed"

    validation: dict[str, Any] | None = None
    validation_errors: list[str] = []
    if isinstance(result, dict):
        validation, validation_errors = validate_result(result)
    if artefact_state == "success":
        if entry is None:
            validation_errors.append("cell is absent from controller status")
        elif entry.get("returncode") != 0:
            validation_errors.append(f"controller recorded return code {entry.get('returncode')}")
        if assertions_error is None and not all(assertions.values()):
            validation_errors.append("resolved assertions are not all true")
        elif assertions_error:
            validation_errors.append(assertions_error)

    env = environment if isinstance(environment, dict) else {}
    prov = provenance if isinstance(provenance, dict) else {}
    set_by_reproducer = env.get("set_by_reproducer", {}) or {}
    inherited = env.get("inherited_relevant", {}) or {}
    debug_dump_enabled = "VLLM_DEBUG_DUMP_PATH" in set_by_reproducer
    debug_dump_requested = manifest.get("debug_dump")
    debug_dump_consistent = (
        debug_dump_requested is None or bool(debug_dump_requested) == debug_dump_enabled
    )
    execution_controls = {
        "debug_dump_requested": debug_dump_requested,
        "debug_dump_enabled": debug_dump_enabled,
        "debug_dump_manifest_consistent": debug_dump_consistent,
        "vllm_logging_level": set_by_reproducer.get("VLLM_LOGGING_LEVEL"),
        "torchinductor_compile_threads": set_by_reproducer.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "vllm_plugins": set_by_reproducer.get("VLLM_PLUGINS"),
        "vllm_disable_compile_cache": set_by_reproducer.get("VLLM_DISABLE_COMPILE_CACHE"),
        "inherited_numerical_environment": {
            name: inherited.get(name)
            for name in (
                "CUDA_MODULE_LOADING",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCHINDUCTOR_DETERMINISTIC",
                "CUBLAS_WORKSPACE_CONFIG",
                "NCCL_NVLS_ENABLE",
            )
        },
    }
    backend = backend_evidence(stdout, stderr, resolved if isinstance(resolved, dict) else None)
    cache = cache_evidence(run_dir, env) if run_dir.is_dir() else {
        "verified_empty_named_roots_before_import": False,
        "scope": "VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, and TRITON_CACHE_DIR only",
        "errors": ["missing run directory"],
    }
    freeze = parse_freeze((prov.get("pip_freeze", {}) or {}).get("stdout"))
    distributions = {
        "pip_freeze_sha256": hashlib.sha256(
            ((prov.get("pip_freeze", {}) or {}).get("stdout") or "").encode()
        ).hexdigest(),
        "distribution_count": len(freeze),
        "key_builds": {
            package: {
                key: (prov.get(package, {}) or {}).get(key)
                for key in (
                    "distribution_version",
                    "module_version",
                    "module_file",
                    "module_file_sha256",
                    "record_sha256",
                    "direct_url",
                )
            }
            for package in ("torch", "triton", "tokenspeed-triton", "vllm")
        },
        "tokenspeed_mla": freeze.get("tokenspeed-mla"),
        "torch_cuda_build": (prov.get("torch", {}) or {}).get("cuda_build"),
    }
    provenance_complete = bool(
        freeze
        and distributions["torch_cuda_build"]
        and distributions["tokenspeed_mla"]
        and all(
            distributions["key_builds"][package].get("distribution_version")
            and distributions["key_builds"][package].get("record_sha256")
            for package in ("torch", "triton", "tokenspeed-triton", "vllm")
        )
        and all(
            distributions["key_builds"][package].get("module_file")
            and distributions["key_builds"][package].get("module_file_sha256")
            for package in ("torch", "triton", "vllm")
        )
    )

    evidence_errors = [
        error
        for error in (provenance_error, environment_error, resolved_error)
        if error and artefact_state == "success"
    ]
    validation_errors.extend(evidence_errors)
    if artefact_state == "success" and not debug_dump_consistent:
        validation_errors.append("manifest debug-dump request disagrees with worker environment")
    usable = bool(
        artefact_state == "success"
        and not validation_errors
        and cache.get("verified_empty_named_roots_before_import")
        and backend.get("complete")
        and selected_gpu(prov, env).get("selection_unambiguous")
        and provenance_complete
    )
    return {
        "id": f"{input_root.label}/{name}",
        "label": input_root.label,
        "name": name,
        "requested": requested,
        "state": artefact_state,
        "usable_for_controlled_comparison": usable,
        "controller_status": entry,
        "failure": failure if failure_error is None else None,
        "errors": validation_errors,
        "gpu": selected_gpu(prov, env),
        "distributions": distributions,
        "distribution_provenance_complete": provenance_complete,
        "backend": backend,
        "cache": cache,
        "execution_controls": execution_controls,
        "resolved_assertions": assertions if assertions_error is None else None,
        "output": validation,
        "model": (result or {}).get("model"),
        "revision": (result or {}).get("revision"),
        "sampling": (result or {}).get("sampling"),
        "prompt_token_ids": (result or {}).get("prompt_token_ids"),
        "prompt_token_ids_batch_sha256": sha256_json((result or {}).get("prompt_token_ids"))
        if (result or {}).get("prompt_token_ids") is not None
        else None,
        "prompt_text_batch_sha256": sha256_json((result or {}).get("prompts"))
        if (result or {}).get("prompts") is not None
        else None,
        "_result": result,
        "_pip_freeze": freeze,
        "_run_dir": run_dir,
    }


def distribution_diff(left: dict[str, str], right: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {"distribution": name, "left": left.get(name), "right": right.get(name)}
        for name in sorted(set(left) | set(right))
        if left.get(name) != right.get(name)
    ]


def equal_non_null(left: Any, right: Any) -> bool:
    return left is not None and left == right


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def compare_cells(left: dict[str, Any], right: dict[str, Any], kind: str) -> dict[str, Any]:
    confounds = []
    for side, cell in (("left", left), ("right", right)):
        if cell["state"] != "success":
            confounds.append(f"{side} cell is {cell['state']}")
        if cell["errors"]:
            confounds.append(f"{side} cell has validation errors")
        if not cell["cache"].get("verified_empty_named_roots_before_import"):
            confounds.append(f"{side} named cache roots were not verified empty")
        if not cell["backend"].get("complete"):
            confounds.append(f"{side} selected-backend evidence is incomplete")
        if not cell["gpu"].get("selection_unambiguous"):
            confounds.append(f"{side} selected GPU UUID is ambiguous")
        if not cell.get("distribution_provenance_complete"):
            confounds.append(f"{side} distribution provenance is incomplete")

    for field, description in (("uuid", "GPU UUID"), ("driver", "driver"), ("name", "GPU model")):
        if not left["gpu"].get(field) or left["gpu"].get(field) != right["gpu"].get(field):
            confounds.append(f"{description} missing or different")
    if left.get("prompt_token_ids") != right.get("prompt_token_ids"):
        confounds.append("prompt token ids differ or are missing")
    for field in ("model", "revision", "sampling"):
        if not left.get(field) or left.get(field) != right.get(field):
            confounds.append(f"{field} differs or is missing")
    backend_same = left["backend"].get("signature") == right["backend"].get("signature")
    if not backend_same:
        confounds.append("selected attention or MoE backend differs")
    controls_same = left.get("execution_controls") == right.get("execution_controls")
    if not controls_same:
        confounds.append("debug-dump or other recorded execution controls differ")

    distributions = distribution_diff(
        left["_pip_freeze"], right["_pip_freeze"]
    )
    distribution_names = set(left["_pip_freeze"]) | set(right["_pip_freeze"])
    identical_distributions = sum(
        left["_pip_freeze"].get(name) == right["_pip_freeze"].get(name)
        for name in distribution_names
    )
    if left["label"] == right["label"] and distributions:
        confounds.append("installed distributions changed within one labelled environment")

    left_result = left.get("_result") or {}
    right_result = right.get("_result") or {}
    left_outputs = (left_result.get("repeats") or [[]])[0] if left_result else []
    right_outputs = (right_result.get("repeats") or [[]])[0] if right_result else []
    prompt_rows = []
    if len(left_outputs) == len(right_outputs) == 16:
        for index, (left_output, right_output) in enumerate(zip(left_outputs, right_outputs)):
            left_tokens = left_output.get("token_ids") or []
            right_tokens = right_output.get("token_ids") or []
            position = first_difference(left_tokens, right_tokens)
            divergence = None
            if position is not None:
                divergence = {
                    "token_index": position,
                    "left_generated_token_id": left_tokens[position] if position < len(left_tokens) else None,
                    "right_generated_token_id": right_tokens[position] if position < len(right_tokens) else None,
                    "left_logprobs": margin_at(left_output, position),
                    "right_logprobs": margin_at(right_output, position),
                }
            prompt_rows.append(
                {
                    "prompt_index": index,
                    "tokens_identical": left_tokens == right_tokens,
                    "text_identical": left_output.get("text") == right_output.get("text"),
                    "logprobs_identical": left_output.get("logprobs") == right_output.get("logprobs"),
                    "first_difference": divergence,
                    "first_token_logprobs": {
                        "left": margin_at(left_output, 0),
                        "right": margin_at(right_output, 0),
                    },
                }
            )

    return {
        "kind": kind,
        "left": left["id"],
        "right": right["id"],
        "controlled": not confounds,
        "confounds": sorted(set(confounds)),
        "gpu": {
            "same_uuid": equal_non_null(left["gpu"].get("uuid"), right["gpu"].get("uuid")),
            "same_driver": equal_non_null(
                left["gpu"].get("driver"), right["gpu"].get("driver")
            ),
            "same_model": equal_non_null(left["gpu"].get("name"), right["gpu"].get("name")),
            "left": left["gpu"],
            "right": right["gpu"],
        },
        "cuda_builds": {
            "left": left["distributions"].get("torch_cuda_build"),
            "right": right["distributions"].get("torch_cuda_build"),
        },
        "distribution_differences": distributions,
        "distribution_summary": {
            "left_count": len(left["_pip_freeze"]),
            "right_count": len(right["_pip_freeze"]),
            "identical_entries": identical_distributions,
            "different_entries": len(distributions),
        },
        "selected_backends_held_fixed": backend_same,
        "selected_backends": {
            "left": left["backend"].get("signature"),
            "right": right["backend"].get("signature"),
        },
        "execution_controls_held_fixed": controls_same,
        "execution_controls": {
            "left": left.get("execution_controls"),
            "right": right.get("execution_controls"),
        },
        "prompt_token_ids_identical": equal_non_null(
            left.get("prompt_token_ids"), right.get("prompt_token_ids")
        ),
        "prompt_token_ids_batch_sha256": {
            "left": left.get("prompt_token_ids_batch_sha256"),
            "right": right.get("prompt_token_ids_batch_sha256"),
        },
        "prompt_text_batch_sha256": {
            "left": left.get("prompt_text_batch_sha256"),
            "right": right.get("prompt_text_batch_sha256"),
        },
        "tokens_identical_prompts": sum(row["tokens_identical"] for row in prompt_rows),
        "texts_identical_prompts": sum(row["text_identical"] for row in prompt_rows),
        "logprobs_identical_prompts": sum(row["logprobs_identical"] for row in prompt_rows),
        "prompt_count": len(prompt_rows),
        "prompts": prompt_rows,
    }


def comparison_kind(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    installed_stack_differs = (
        left["distributions"].get("pip_freeze_sha256")
        != right["distributions"].get("pip_freeze_sha256")
        or left["distributions"].get("torch_cuda_build")
        != right["distributions"].get("torch_cuda_build")
    )
    dimensions = {
        "environment": installed_stack_differs,
        "compile": left["requested"]["compile"] != right["requested"]["compile"],
        "graphs": left["requested"]["graphs"] != right["requested"]["graphs"],
        "runner": left["requested"]["runner"] != right["requested"]["runner"],
    }
    changed = [name for name, value in dimensions.items() if value]
    if not changed and left["label"] != right["label"]:
        return "fresh_process_repeat"
    if len(changed) != 1:
        return None
    if changed[0] == "environment":
        left_build = left["distributions"].get("torch_cuda_build")
        right_build = right["distributions"].get("torch_cuda_build")
        if left_build and right_build and left_build != right_build:
            return "cuda_build"
        return "fresh_process_environment"
    return {
        "compile": "compiled_vs_eager",
        "graphs": "graphs_off_vs_on",
        "runner": "v1_vs_v2",
    }[changed[0]]


def public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cell.items() if not key.startswith("_")}


def relative_evidence_path(path: Path, inputs: list[InputRoot]) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        pass
    for input_root in inputs:
        try:
            relative = path.resolve().relative_to(input_root.path.parent.resolve())
            return str(relative)
        except ValueError:
            continue
    return str(path.resolve())


def source_digest(path: Path, inputs: list[InputRoot]) -> dict[str, Any]:
    if not path.is_file() and path.with_name(path.name + ".gz").is_file():
        path = path.with_name(path.name + ".gz")
    return {
        "path": relative_evidence_path(path, inputs),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def historical_e4_comparison(
    cells: list[dict[str, Any]], historical_root: Path, inputs: list[InputRoot]
) -> dict[str, Any]:
    arm_names = {
        "on": "Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_compile_v2_graphs0_prefix0",
        "off": "Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs0_prefix0",
    }
    rows = []
    errors = []
    for cell in cells:
        requested = cell["requested"]
        if requested["runner"] != "v2" or requested["graphs"] != "off":
            continue
        if cell["state"] != "success" or not cell.get("_result"):
            continue
        arm_root = historical_root / arm_names[requested["compile"]]
        old_run_path = arm_root / "run.json"
        old_outputs_path = arm_root / "runs.json"
        old_run, run_error = read_json(old_run_path)
        old_outputs, outputs_error = read_json(old_outputs_path)
        if run_error or outputs_error:
            errors.append(
                f"{cell['id']}: {run_error or ''} {outputs_error or ''}".strip()
            )
            continue
        candidates = [
            item
            for item in old_outputs
            if item.get("kind") == "original"
            and item.get("repeat") == 0
            and str(item.get("target_rid")) == "0"
        ]
        if len(candidates) != 1:
            errors.append(f"{cell['id']}: expected one historical original repeat-0 target-0 row")
            continue
        historical_output = candidates[0].get("target") or {}
        current_output = cell["_result"]["repeats"][0][0]
        historical_logprobs = [
            {int(token_id): float.fromhex(value) for token_id, value in position}
            for position in historical_output.get("logprobs", [])
        ]
        current_logprobs = [
            {
                int(candidate["token_id"]): candidate["logprob"]
                for candidate in position.get("candidates", [])
            }
            for position in current_output.get("logprobs", [])
        ]
        top5_equal = [left == right for left, right in zip(historical_logprobs, current_logprobs)]
        if len(historical_logprobs) != len(current_logprobs):
            top5_equal.extend(
                [False] * abs(len(historical_logprobs) - len(current_logprobs))
            )
        current_files = [
            cell["_run_dir"] / name
            for name in ("result.json", "resolved_config.json", "provenance.json")
        ]
        rows.append(
            {
                "current_cell": cell["id"],
                "historical_arm": arm_root.name,
                "scope": "prompt/target 0, original repeat 0",
                "historical_input_sha256": old_run.get("input_sha256"),
                "current_input_sha256": cell.get("prompt_text_batch_sha256"),
                "exact_prompt_batch": equal_non_null(
                    old_run.get("input_sha256"), cell.get("prompt_text_batch_sha256")
                ),
                "model_revision": {
                    "historical": old_run.get("model_revision"),
                    "current": cell.get("revision"),
                },
                "resolved_compile": {
                    "historical": old_run.get("resolved_compile"),
                    "current": 3 if requested["compile"] == "on" else 0,
                },
                "resolved_cudagraph": {
                    "historical": old_run.get("resolved_cudagraph"),
                    "current": "NONE",
                },
                "generated_tokens": {
                    "count": len(current_output.get("token_ids", [])),
                    "exact": historical_output.get("tokens") == current_output.get("token_ids"),
                    "historical": historical_output.get("tokens"),
                    "current": current_output.get("token_ids"),
                },
                "top5_logprobs": {
                    "positions": len(top5_equal),
                    "exact_positions": sum(top5_equal),
                    "all_exact": bool(top5_equal) and all(top5_equal),
                },
                "source_files": [
                    source_digest(path, inputs)
                    for path in (old_run_path, old_outputs_path, *current_files)
                ],
            }
        )
    return {
        "scope": "Retained historical E4 output is target 0 only; other 15 outputs are unavailable in runs.json.",
        "hook_scope": (
            "The retained E4 arms contain hook records. The repository prose says a later hook-free "
            "check reproduced the corresponding E4 tokens, but separate raw hook-free output files "
            "were not found, so direct raw-file comparison to that check is unavailable."
        ),
        "errors": errors,
        "comparisons": rows,
    }


def audit(inputs: list[InputRoot], historical_root: Path | None = None) -> dict[str, Any]:
    cells = []
    input_errors = []
    for input_root in inputs:
        manifest, manifest_error = read_json(input_root.path / "manifest.json")
        if manifest_error or not isinstance(manifest, dict):
            input_errors.append(f"{input_root.label}: {manifest_error or 'invalid manifest'}")
            continue
        requested = manifest.get("requested")
        if not isinstance(requested, list):
            input_errors.append(f"{input_root.label}: manifest requested list is missing")
            continue
        status = status_entries(input_root.path)
        for item in requested:
            if not isinstance(item, dict) or set(("compile", "graphs", "runner")) - set(item):
                input_errors.append(f"{input_root.label}: invalid requested cell {item!r}")
                continue
            cells.append(load_cell(input_root, item, status, manifest))

    comparisons_report = []
    for left, right in combinations(cells, 2):
        kind = comparison_kind(left, right)
        if kind:
            comparisons_report.append(compare_cells(left, right, kind))

    failed = [cell["id"] for cell in cells if cell["state"] == "failed"]
    missing = [cell["id"] for cell in cells if cell["state"] == "missing"]
    invalid = [cell["id"] for cell in cells if cell["state"].startswith("invalid") or cell["errors"]]
    confounded = [
        f"{comparison['left']} :: {comparison['right']}"
        for comparison in comparisons_report
        if not comparison["controlled"]
    ]
    unusable = [
        cell["id"]
        for cell in cells
        if cell["state"] == "success" and not cell["usable_for_controlled_comparison"]
    ]
    if input_errors or invalid:
        state = "INVALID"
    elif failed or missing:
        state = "INCOMPLETE"
    elif confounded or unusable:
        state = "CONFOUNDED"
    else:
        state = "CONTROLLED"

    cache_roots = [
        path
        for cell in cells
        for path in cell["cache"].get("configured_roots", {}).values()
        if path
    ]
    global_checks = {
        "all_cells_use_distinct_named_cache_roots": len(cache_roots) == len(set(cache_roots)),
        "selected_gpu_uuids": sorted(
            {cell["gpu"].get("uuid") for cell in cells if cell["gpu"].get("uuid")}
        ),
        "drivers": sorted({cell["gpu"].get("driver") for cell in cells if cell["gpu"].get("driver")}),
        "gpu_models": sorted({cell["gpu"].get("name") for cell in cells if cell["gpu"].get("name")}),
        "torch_cuda_builds": sorted(
            {
                cell["distributions"].get("torch_cuda_build")
                for cell in cells
                if cell["distributions"].get("torch_cuda_build")
            }
        ),
        "distribution_snapshots_by_environment": {
            label: sorted(
                {
                    cell["distributions"]["pip_freeze_sha256"]
                    for cell in cells
                    if cell["label"] == label and cell["state"] == "success"
                }
            )
            for label in sorted({cell["label"] for cell in cells})
        },
    }
    global_identity_held = all(
        len(global_checks[field]) == 1
        for field in ("selected_gpu_uuids", "drivers", "gpu_models")
    )
    distribution_sets_stable = all(
        len(hashes) == 1
        for hashes in global_checks["distribution_snapshots_by_environment"].values()
    )
    global_checks["one_gpu_uuid_driver_and_model"] = global_identity_held
    global_checks["distributions_stable_within_each_environment"] = distribution_sets_stable
    if not (
        global_checks["all_cells_use_distinct_named_cache_roots"]
        and global_identity_held
        and distribution_sets_stable
    ):
        state = "CONFOUNDED" if state == "CONTROLLED" else state

    report = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_state": state,
        "input_errors": input_errors,
        "failed_cells": failed,
        "missing_cells": missing,
        "invalid_cells": invalid,
        "unusable_successful_cells": unusable,
        "confounded_comparisons": confounded,
        "global_checks": global_checks,
        "method_bounds": [
            "Cold-root proof covers only VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, "
            "and TRITON_CACHE_DIR; it does not prove every driver or library cache was cold.",
            "Two generate calls establish in-process repeatability only. One cell does not "
            "establish fresh-process repeatability of that configuration.",
            "Matching selection logs identify vLLM's named attention and MoE implementations, "
            "not the exact generated GPU kernels.",
            "Debug-dump instrumentation is recorded as an execution control. A no-dump run "
            "retains DEBUG logs and named cache trees, but not depyf/FX debug dumps.",
            "A CUDA-build comparison is a comparison of the recorded installed stacks. "
            "Distribution differences are reported rather than silently attributed to one wheel.",
            "Recorded versions, module paths, RECORD hashes, and distribution differences bound "
            "the current run; they do not by themselves prove exact identity with the historical stack.",
            "Eager vLLM is an internal reference, not an external accuracy oracle.",
            "V1 versus V2 changes the runner stack, and is not a one-kernel comparison.",
        ],
        "cells": [public_cell(cell) for cell in cells],
        "comparisons": comparisons_report,
    }
    if historical_root is not None:
        report["historical_e4"] = historical_e4_comparison(
            cells, historical_root.resolve(), inputs
        )
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MoE compile diagnostic audit",
        "",
        f"Evidence state: **{report['evidence_state']}**.",
        "",
        "## Controls",
        "",
        f"- Selected GPU UUIDs: `{report['global_checks']['selected_gpu_uuids']}`",
        f"- Drivers: `{report['global_checks']['drivers']}`",
        f"- GPU models: `{report['global_checks']['gpu_models']}`",
        f"- Torch CUDA builds: `{report['global_checks']['torch_cuda_builds']}`",
        "- Distinct named cache roots: "
        f"`{report['global_checks']['all_cells_use_distinct_named_cache_roots']}`",
        "",
        "## Cells",
        "",
        "| cell | state | usable | repeat tokens | duplicate pairs r0/r1 | short cycles r0/r1 | first-token margin range r0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        output = cell.get("output") or {}
        repeats = output.get("repeats") or []
        duplicates = "/".join(str(item.get("duplicate_pairs_agree")) for item in repeats) or "n/a"
        cycles = "/".join(str(len(item.get("short_cycles") or [])) for item in repeats) or "n/a"
        first_margins = [
            item["top1_margin"]
            for item in (repeats[0].get("first_token_logprob_margins") if repeats else [])
            if item.get("top1_margin") is not None
        ]
        margin_range = (
            f"{min(first_margins):.9g}..{max(first_margins):.9g}"
            if first_margins
            else "n/a"
        )
        lines.append(
            f"| `{cell['id']}` | {cell['state']} | {cell['usable_for_controlled_comparison']} | "
            f"{output.get('in_process_repeat_tokens_identical', 'n/a')} | {duplicates} | {cycles} | "
            f"{margin_range} |"
        )
    lines.extend(
        [
            "",
            "## One-axis comparisons",
            "",
            "| kind | left | right | controlled | identical tokens | identical logprobs | dist diffs |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["comparisons"]:
        lines.append(
            f"| {item['kind']} | `{item['left']}` | `{item['right']}` | {item['controlled']} | "
            f"{item['tokens_identical_prompts']}/{item['prompt_count']} | "
            f"{item['logprobs_identical_prompts']}/{item['prompt_count']} | "
            f"{item['distribution_summary']['different_entries']} |"
        )
        if item["confounds"]:
            lines.append(f"|  | confounds | {', '.join(item['confounds'])} |  |  |  |  |")
    lines.extend(["", "## Method bounds", ""])
    lines.extend(f"- {bound}" for bound in report["method_bounds"])
    if report["failed_cells"] or report["missing_cells"] or report["invalid_cells"]:
        lines.extend(
            [
                "",
                "## Unavailable evidence",
                "",
                f"- Failed cells: `{report['failed_cells']}`",
                f"- Missing cells: `{report['missing_cells']}`",
                f"- Invalid cells: `{report['invalid_cells']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_input(value: str) -> InputRoot:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    return InputRoot(label=label, path=Path(raw_path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", type=parse_input, required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--historical-e4-root",
        type=Path,
        help="optional probes/shape/results/e4 root for the retained target-0 comparison",
    )
    parser.add_argument(
        "--historical-out",
        type=Path,
        help="write only the optional historical comparison as JSON",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    labels = [item.label for item in arguments.input]
    if len(labels) != len(set(labels)):
        raise SystemExit("input labels must be unique")
    if arguments.historical_out and not arguments.historical_e4_root:
        raise SystemExit("--historical-out requires --historical-e4-root")
    report = audit(arguments.input, arguments.historical_e4_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.json_out:
        arguments.json_out.write_text(encoded)
    if arguments.markdown_out:
        arguments.markdown_out.write_text(markdown(report))
    if arguments.historical_out:
        arguments.historical_out.write_text(
            json.dumps(report["historical_e4"], indent=2, sort_keys=True) + "\n"
        )
    if not arguments.json_out and not arguments.markdown_out:
        sys.stdout.write(encoded)
    elif arguments.markdown_out:
        sys.stdout.write(markdown(report))
    else:
        print(
            f"evidence_state={report['evidence_state']} cells={len(report['cells'])} "
            f"comparisons={len(report['comparisons'])}"
        )


if __name__ == "__main__":
    main()
