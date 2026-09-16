#!/usr/bin/env python3
"""Read-only CPU audit of a completed actual-routing observation bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from routing_telemetry import (
    RoutingTelemetryError,
    compare_hook_hashes,
    compare_matched_samples,
    digest_token_ids,
    hook_evidence,
    summarise_selected_experts,
    validate_dynamic_inputs,
    validate_graph_dispatch,
)

EXPECTED_SOURCE = "7dc6f469726d3eec0719c98e9bf6458945b961af"
EXPECTED_VLLM = "98dff2a81d747d1dba01a47f939f48c3526d4206"
EXPECTED_MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPECTED_REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
EXPECTED_SHAPE = {"num_experts": 60, "num_layers": 24, "experts_per_token": 4}
EXPECTED_HOOK_SHA256 = "5bc003293fef95e525658226cfb05b2ad28199cb7010ebb09bbd9c37954e357f"
EXPECTED_HARNESS_SHA256 = "c952ac33b31a72647a3f8107e7e5e7672411ccdd7b8315ca1302a8be0b4506f5"
EXPECTED_TELEMETRY_SHA256 = "ea7974c33b2670a045f62d2beb82254e6d694e989699a3ee91c2b5d7af4cc4ba"
EXPECTED_VLLM_FILES = {
    "config/compilation.py": "3a79ea08d48d44879ac8cbfee1c7d88f9bd72927d9bd12eee31743e8da8a4d7e",
    "config/scheduler.py": "f1f70b83527ee2ccc78bf186d418739d2616de1f96834ad0bbb65f42b6a41787",
    "config/vllm.py": "788430b9211a5b0ccda0be565ad2a0f27e9069370f35d2b1f6a333a6ca135186",
    "engine/arg_utils.py": "950cb3b650c081d6d36d43d73e3ecd669b44eabd4fe8395b92e399c7a89b8011",
    "entrypoints/llm.py": "52de4ac99489e004ef6c61d0bedc84aa96020dd58b8bd1ae500814b548b2b83e",
    "model_executor/layers/fused_moe/routed_experts_capturer.py": "cd006bc1a1703418cca3b589fafb2abc4268a086fb3db044bbb64348aeac8b13",
    "model_executor/layers/fused_moe/router/fused_moe_router.py": "f182f30712983f61815f8df947af597277d660cd2891066fcb13789d55f7353e",
    "sampling_params.py": "2aba9ebd1c3921d601dcf94ac6a1f726e5172fe991e313cd5a69d3eaa02013fa",
    "v1/core/sched/scheduler.py": "abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983",
    "v1/engine/output_processor.py": "80a01067f4b3b351239506a3a1754dafdfc83cd162352d95545ba8ebed44c447",
    "v1/worker/gpu_model_runner.py": "3970d7b764f80847842e10fc20f0e3f4f3da84cd97d042bb92db6e99147337e2",
    "outputs.py": "346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425",
}
CONDITIONS = ("graph-control", "graph-telemetry", "eager-telemetry")
SOURCE_FILES = {
    "source_snapshot/routing_harness.py",
    "source_snapshot/routing_telemetry.py",
    "source_snapshot/shape_hook_init.py",
}


class AuditError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON value is not an object: {path}")
    return value


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AuditError(f"invalid relative evidence path: {relative!r}")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise AuditError(f"non-canonical relative evidence path: {relative!r}")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError(f"evidence path escapes data root: {relative}") from exc
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise AuditError(f"evidence symlink is forbidden: {relative}")
    return candidate


def label_for(report_path: Path, condition: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{report_path.stem}_{condition}").strip("_")


def verify_handoff(root: Path, label: str) -> tuple[dict[str, Any], set[str]]:
    manifest_path = root / "handoff" / label / "manifest.json"
    manifest = load_object(manifest_path)
    if set(manifest) != {"schema", "label", "files", "sha256"}:
        raise AuditError(f"unexpected manifest fields: {manifest_path}")
    if manifest["schema"] != 1 or manifest["label"] != label:
        raise AuditError(f"manifest identity mismatch: {manifest_path}")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise AuditError(f"manifest has no files: {manifest_path}")
    payload = {"schema": 1, "label": label, "files": files}
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    if manifest["sha256"] != digest:
        raise AuditError(f"canonical manifest digest mismatch: {manifest_path}")
    paths: list[str] = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise AuditError(f"invalid file row in {manifest_path}")
        path = safe_path(root, row["path"])
        if not path.is_file():
            raise AuditError(f"manifest file is missing: {row['path']}")
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            raise AuditError(f"manifest file digest mismatch: {row['path']}")
        paths.append(row["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise AuditError(f"manifest paths are not sorted and unique: {manifest_path}")
    ack_path = root / "acks" / f"{label}.json"
    expected_ack = {
        "schema": 1,
        "label": label,
        "handoff_manifest_sha256": digest,
        "verified_manifest_sha256": digest,
    }
    if load_object(ack_path) != expected_ack:
        raise AuditError(f"ACK mismatch: {ack_path}")
    return manifest, set(paths)


def read_hook_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"raw rank-0 hook log is missing or unsafe: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"malformed hook row at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise AuditError(f"non-object hook row at {path}:{line_number}")
        records.append(value)
    if not records:
        raise AuditError(f"empty hook log: {path}")
    return records


def verify_sample(sample: dict[str, Any], shape: dict[str, int], telemetry: bool) -> None:
    prompt_count = sample.get("prompt_token_count")
    generated_count = sample.get("generated_token_count")
    output_ids = sample.get("output_token_ids")
    if (
        not isinstance(prompt_count, int)
        or isinstance(prompt_count, bool)
        or prompt_count <= 0
        or not isinstance(generated_count, int)
        or isinstance(generated_count, bool)
        or generated_count <= 0
        or not isinstance(output_ids, list)
        or len(output_ids) != generated_count
    ):
        raise AuditError("sample token counts are invalid")
    if digest_token_ids(output_ids) != sample.get("output_token_ids_sha256"):
        raise AuditError("retained output-token digest differs from raw IDs")
    prompt_digest = sample.get("prompt_token_ids_sha256")
    if not isinstance(prompt_digest, str) or re.fullmatch(r"[0-9a-f]{64}", prompt_digest) is None:
        raise AuditError("sample lacks a valid prompt-token digest")
    routing = sample.get("routing")
    if not telemetry:
        if routing is not None:
            raise AuditError("control condition unexpectedly contains routing data")
        return
    if not isinstance(routing, dict) or not isinstance(routing.get("selected_ids"), list):
        raise AuditError("telemetry sample lacks raw selected IDs")
    valid_tokens = prompt_count + generated_count - 1
    if len(routing["selected_ids"]) != valid_tokens:
        raise AuditError("selected-ID rows do not equal prompt plus decode forwards")
    recomputed = summarise_selected_experts(
        routing["selected_ids"],
        num_experts=shape["num_experts"],
        valid_tokens=valid_tokens,
    )
    exact_keys = {
        "schema",
        "source",
        "valid_tokens",
        "num_layers",
        "experts_per_token",
        "selected_id_count",
        "per_layer_count_sums",
        "per_layer_distinct_experts",
        "per_layer_expert_counts",
        "selected_ids",
        "selected_ids_sha256",
    }
    if any(routing.get(key) != recomputed[key] for key in exact_keys):
        raise AuditError("stored routing summary differs from raw selected IDs")
    reported = routing.get("reported_tokens")
    padding = routing.get("capture_padding_trimmed")
    if (
        not isinstance(reported, int)
        or not isinstance(padding, int)
        or reported < valid_tokens
        or padding != reported - valid_tokens
    ):
        raise AuditError("capture-padding accounting is invalid")
    if routing["num_layers"] != shape["num_layers"] or routing["experts_per_token"] != shape["experts_per_token"]:
        raise AuditError("routing shape differs from the pinned model")


def verify_source_attestation(root: Path, conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    attestations = [conditions[name].get("source_attestation") for name in CONDITIONS]
    if not isinstance(attestations[0], dict) or any(value != attestations[0] for value in attestations[1:]):
        raise AuditError("condition source attestations differ")
    value = attestations[0]
    if (
        value.get("source_commit") != EXPECTED_SOURCE
        or value.get("reviewed_vllm_commit") != EXPECTED_VLLM
        or value.get("vllm_version") != "0.29.0"
        or value.get("vllm_files") != EXPECTED_VLLM_FILES
        or value.get("shape_hook_sha256") != EXPECTED_HOOK_SHA256
    ):
        raise AuditError("source attestation differs from reviewed bytes")
    hook_snapshot = root / "source_snapshot" / "shape_hook_init.py"
    if sha256_file(hook_snapshot) != EXPECTED_HOOK_SHA256:
        raise AuditError("retained shape-hook source differs from its attestation")
    if sha256_file(root / "source_snapshot" / "routing_harness.py") != EXPECTED_HARNESS_SHA256:
        raise AuditError("retained routing harness differs from the reviewed launch bytes")
    if sha256_file(root / "source_snapshot" / "routing_telemetry.py") != EXPECTED_TELEMETRY_SHA256:
        raise AuditError("retained routing validator differs from the reviewed launch bytes")
    return value


def audit(data_root: Path) -> dict[str, Any]:
    root = data_root.resolve()
    complete = load_object(root / "COMPLETE.json")
    report_relative = complete.get("output")
    report_path = safe_path(root, report_relative)
    report = load_object(report_path)
    shape = report.get("expected_routing_shape")
    if shape != EXPECTED_SHAPE:
        raise AuditError("report differs from the pinned Qwen1.5-MoE routing shape")
    if (
        report.get("model") != EXPECTED_MODEL
        or report.get("revision") != EXPECTED_REVISION
        or report.get("source") != "vllm 0.29 enable_return_routed_experts"
        or report.get("requested_graph_mode") is not True
    ):
        raise AuditError("aggregate model, revision, or routing source is not pinned")

    conditions: dict[str, dict[str, Any]] = {}
    manifests: dict[str, str] = {}
    expected_labels = []
    for condition in CONDITIONS:
        label = label_for(report_path, condition)
        expected_labels.append(label)
        manifest, paths = verify_handoff(root, label)
        manifests[label] = manifest["sha256"]
        condition_relative = f"{report_path.stem}.conditions/{condition}.json"
        log_relative = f"{report_path.stem}.conditions/{condition}.log"
        hook_relative = f"hook/{condition}/rank0.jsonl"
        required = SOURCE_FILES | {condition_relative, log_relative, hook_relative}
        missing = sorted(required - paths)
        if missing:
            raise AuditError(f"handoff {label} omits required files: {missing}")
        condition_value = load_object(root / condition_relative)
        if condition_value.get("condition") != condition or condition_value.get("status") != "complete":
            raise AuditError(f"condition identity/status mismatch: {condition}")
        controls = condition_value.get("engine_controls")
        expected_mode = "NONE" if condition == "eager-telemetry" else "FULL"
        expected_telemetry = condition != "graph-control"
        expected_controls = {
            "seed": 0,
            "max_model_len": 2048,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "enforce_eager": False,
            "compilation_config": {"mode": 0, "cudagraph_mode": expected_mode},
            "enable_return_routed_experts": expected_telemetry,
        }
        if controls != expected_controls:
            raise AuditError(f"condition controls differ: {condition}")
        samples = condition_value.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            raise AuditError(f"condition has too few samples: {condition}")
        for sample in samples:
            if not isinstance(sample, dict):
                raise AuditError(f"condition has a non-object sample: {condition}")
            verify_sample(sample, shape, expected_telemetry)
        raw_evidence = hook_evidence(
            read_hook_records(root / hook_relative), require_hashes=True
        )
        if raw_evidence != condition_value.get("hook_evidence"):
            raise AuditError(f"derived hook evidence differs from raw JSONL: {condition}")
        conditions[condition] = condition_value

    if complete.get("acked_labels") != expected_labels:
        raise AuditError("COMPLETE.json does not retain all ACKed labels in execution order")
    report_conditions = report.get("conditions")
    expected_report_conditions = {
        "graph_control": conditions["graph-control"],
        "graph_telemetry": conditions["graph-telemetry"],
        "eager_telemetry": conditions["eager-telemetry"],
    }
    if report_conditions != expected_report_conditions:
        raise AuditError("aggregate conditions differ from immutable condition JSON")
    source = verify_source_attestation(root, conditions)
    if report.get("source_attestation") != source:
        raise AuditError("aggregate source attestation differs from conditions")

    graph_control = conditions["graph-control"]
    graph_telemetry = conditions["graph-telemetry"]
    eager_telemetry = conditions["eager-telemetry"]
    recomputed = {
        "instrumentation_preserves_graph_tokens": compare_matched_samples(
            graph_control["samples"], graph_telemetry["samples"], compare_routing=False
        ),
        "instrumentation_preserves_graph_hidden_logits": compare_hook_hashes(
            graph_control["hook_evidence"], graph_telemetry["hook_evidence"]
        ),
        "graph_telemetry_dispatch": validate_graph_dispatch(
            graph_telemetry["hook_evidence"], expect_graph=True, expect_compile="0"
        ),
        "eager_telemetry_dispatch": validate_graph_dispatch(
            eager_telemetry["hook_evidence"], expect_graph=False, expect_compile="0"
        ),
        "graph_matches_eager": compare_matched_samples(
            graph_telemetry["samples"], eager_telemetry["samples"], compare_routing=True
        ),
        "dynamic_graph_inputs": validate_dynamic_inputs(graph_telemetry["samples"]),
    }
    for key, value in recomputed.items():
        json_value = json.loads(json.dumps(value, sort_keys=True))
        if report.get(key) != json_value:
            raise AuditError(f"aggregate field differs from raw recomputation: {key}")
    scientific_valid = all(
        (
            recomputed["instrumentation_preserves_graph_tokens"]["all_equal"],
            recomputed["instrumentation_preserves_graph_hidden_logits"]["all_equal"],
            recomputed["graph_telemetry_dispatch"]["valid"],
            recomputed["eager_telemetry_dispatch"]["valid"],
            recomputed["graph_matches_eager"]["all_equal"],
            recomputed["dynamic_graph_inputs"]["valid"],
        )
    )
    if report.get("valid") is not scientific_valid:
        raise AuditError("stored validity differs from raw recomputation")
    expected_status = "valid" if scientific_valid else "invalid"
    if complete.get("schema") != 1 or complete.get("valid") is not scientific_valid or complete.get("status") != expected_status:
        raise AuditError("COMPLETE.json differs from raw recomputation")
    return {
        "schema": 1,
        "audit_valid": True,
        "scientific_valid": scientific_valid,
        "conditions": list(CONDITIONS),
        "verified_handoff_manifests": manifests,
        "source_commit": source["source_commit"],
        "reviewed_vllm_commit": source["reviewed_vllm_commit"],
        "dynamic_graph_inputs": recomputed["dynamic_graph_inputs"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.data_root)
    except (AuditError, RoutingTelemetryError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"schema": 1, "audit_valid": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["scientific_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
