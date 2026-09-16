#!/usr/bin/env python3
"""Stage and independently audit portable MoE boundary-localisation evidence.

Private cache binaries and saved tensors stay in the durable work root.  Their
immutable handoff rows remain public as hash-only omissions.  Compiler Python
and textual IR are exported in deterministic tarballs.  All other admitted
JSON, logs, and source snapshots are deterministically gzip-compressed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any


MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
FROZEN_COMPARATOR_SHA256 = "285f0d9c68928fb6afb8da25304f72bb68e39f9a3b068e033abad9e2f37c27a5"
ATTENTION_COMPARATOR_SHA256 = "cc267b5ce8d4b2bb662b20cbf92cf6c70a66ef9e40a4ea86ffbe5434a677caec"
POST_OPERATOR_COMPARATOR_SHA256 = "39018437fb3bdeee13b433f2208d43fb542d101c23a05648f04588d54b39d6fd"
PRECISION_CAST_CONTROL_SHA256 = "a2313baecce5aa69bb96604313413f580d4e16eea5b635204d8789fd0e0da901"
ROUNDED_RESIDUAL_CONTROL_SHA256 = "633c7d6b4bdd58633af4567922eea8a4a83c453101f0ec6793f81a9d99e9bf63"
INDUCTOR_REUSE_CONTROL_SHA256 = "e493c3c294bfc516608b21f930e5008a8d6d169a1e75142a1852a9573eaf9a96"
VLLM_C_RMS_CONTROL_SHA256 = "d34b8166328e7a7fa006e65ae988df625d6166bceec9546e316f811fa46f0677"
KNOWN_CONTROLLERS = {
    "moe_boundary_localise.py": FROZEN_COMPARATOR_SHA256,
    "attention_boundary_localise.py": ATTENTION_COMPARATOR_SHA256,
    "post_attention_operator_localise.py": POST_OPERATOR_COMPARATOR_SHA256,
    "precision_cast_control.py": PRECISION_CAST_CONTROL_SHA256,
    "rounded_residual_control.py": ROUNDED_RESIDUAL_CONTROL_SHA256,
    "inductor_reuse_control.py": INDUCTOR_REUSE_CONTROL_SHA256,
    "vllm_c_rms_control.py": VLLM_C_RMS_CONTROL_SHA256,
}
SINGLE_CONTROL_PROFILES = {
    "precision_cast_control.py": ("compiled-emulate-precision-casts", 1),
    "inductor_reuse_control.py": ("compiled-reuse-off", 32),
    "vllm_c_rms_control.py": ("compiled-vllm-c-rms", 32),
}
POST_OPERATOR_PUBLIC_TENSORS = {
    "compiled-operator/tensors/o_projection_output.pt",
    "compiled-operator/tensors/residual.pt",
    "compiled-operator/tensors/norm_weight.pt",
    "compiled-operator/tensors/norm_output.pt",
    "eager-boundary/tensors/o_projection_output.pt",
    "eager-boundary/tensors/residual.pt",
    "eager-boundary/tensors/norm_weight.pt",
    "eager-boundary/tensors/norm_output.pt",
    "eager-boundary/tensors/updated_residual.pt",
}
POST_OPERATOR_UNUSED_LOADER_TENSORS = {
    "compiled-operator/tensors/attention_output.pt",
    "compiled-operator/tensors/o_projection_matrix.pt",
    "compiled-operator/tensors/norm_projection_input.pt",
}
TEXT_CACHE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".ll",
    ".llir",
    ".ptx",
    ".py",
    ".source",
    ".ttgir",
    ".ttir",
}


class AuditError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise AuditError(f"manifest path is not a string: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise AuditError(f"unsafe manifest path: {value!r}")
    return relative


def is_cache(relative: PurePosixPath) -> bool:
    return "cache" in relative.parts


def is_tensor(relative: PurePosixPath) -> bool:
    return "tensors" in relative.parts or relative.suffix.lower() in {".pt", ".pth", ".safetensors"}


def deterministic_gzip(value: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as stream:
        stream.write(value)
    return output.getvalue()


def gunzip(path: Path) -> bytes:
    try:
        with gzip.open(path, "rb") as stream:
            return stream.read()
    except (OSError, EOFError) as error:
        raise AuditError(f"cannot decompress {path}: {error}") from error


def ack_path(ack_dir: Path, label: str) -> Path:
    candidates = [ack_dir / f"{label}.verified-ack.json", ack_dir / f"{label}.json"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise AuditError(f"expected exactly one durable ACK for {label} under {ack_dir}")
    return present[0]


def validate_manifest_value(value: Any, *, label: str, origin: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"schema", "label", "files", "sha256"}:
        raise AuditError(f"invalid manifest schema: {origin}")
    if value["schema"] != 1 or value["label"] != label or not isinstance(value["files"], list):
        raise AuditError(f"invalid manifest identity: {origin}")
    payload = {key: value[key] for key in ("schema", "label", "files")}
    if value["sha256"] != sha256_bytes(canonical_json(payload)):
        raise AuditError(f"invalid manifest digest: {origin}")
    paths: list[str] = []
    for row in value["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise AuditError(f"invalid manifest file row: {origin}")
        relative = safe_relative(row["path"])
        if not isinstance(row["size"], int) or row["size"] < 0:
            raise AuditError(f"invalid manifest size: {relative}")
        if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
            raise AuditError(f"invalid manifest file digest: {relative}")
        paths.append(relative.as_posix())
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise AuditError(f"manifest paths are not sorted and unique: {origin}")
    return value["files"]


def validate_ack(value: Any, label: str, digest: str, origin: str) -> None:
    expected = {
        "schema": 1,
        "label": label,
        "handoff_manifest_sha256": digest,
        "verified_manifest_sha256": digest,
    }
    if value != expected:
        raise AuditError(f"invalid durable ACK: {origin}")


def admitted_handoffs(results: Path, ack_dir: Path) -> list[dict[str, Any]]:
    manifests = sorted((results / "handoff").glob("*/manifest.json"))
    admitted = []
    for manifest_path in manifests:
        label = manifest_path.parent.name
        manifest = read_json(manifest_path)
        rows = validate_manifest_value(manifest, label=label, origin=str(manifest_path))
        source_ack = ack_path(ack_dir, label)
        ack = read_json(source_ack)
        validate_ack(ack, label, manifest["sha256"], str(source_ack))
        root = results.resolve()
        for row in rows:
            relative = safe_relative(row["path"])
            source = results.joinpath(*relative.parts)
            try:
                resolved = source.resolve(strict=True)
            except OSError as error:
                raise AuditError(f"missing admitted file {relative}: {error}") from error
            if resolved == root or root not in resolved.parents or source.is_symlink() or not source.is_file():
                raise AuditError(f"invalid admitted file: {relative}")
            if source.stat().st_size != row["size"] or sha256_file(source) != row["sha256"]:
                raise AuditError(f"admitted file does not match manifest: {relative}")
        admitted.append(
            {
                "label": label,
                "manifest": manifest,
                "manifest_path": manifest_path,
                "ack": ack,
                "ack_path": source_ack,
            }
        )
    if not admitted:
        raise AuditError(f"no ACKed handoffs under {results}")
    return admitted


def load_comparator(path: Path, dependencies: dict[str, Path] | None = None) -> Any:
    sys.dont_write_bytecode = True
    temporary: tempfile.TemporaryDirectory[str] | None = None
    old_modules: dict[str, Any] = {}
    try:
        if dependencies:
            temporary = tempfile.TemporaryDirectory(prefix="moe-frozen-controller-")
            import_root = Path(temporary.name)
            controller = import_root / path.name
            controller.write_bytes(path.read_bytes())
            for name, dependency in dependencies.items():
                (import_root / name).write_bytes(dependency.read_bytes())
                if name.endswith(".py"):
                    module_name = Path(name).stem
                    if module_name in sys.modules:
                        old_modules[module_name] = sys.modules.pop(module_name)
            path = controller
        name = f"moe_public_frozen_{sha256_file(path)[:16]}_{id(path)}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise AuditError(f"cannot import frozen comparator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if dependencies:
            for dependency_name in dependencies:
                sys.modules.pop(Path(dependency_name).stem, None)
            sys.modules.update(old_modules)
        if temporary is not None:
            temporary.cleanup()
    required = "analyze" if path.name == "post_attention_operator_localise.py" else "compare_cells"
    if not callable(getattr(module, required, None)):
        raise AuditError(f"frozen source lacks {required}: {path}")
    return module


def controller_profile(controller: Any, origin: str) -> str:
    if not isinstance(controller, dict):
        raise AuditError(f"controller snapshot is not an object: {origin}")
    expected = {"model": MODEL, "revision": REVISION}
    for key, value in expected.items():
        if controller.get(key) != value:
            raise AuditError(f"controller snapshot lacks {key}={value!r}: {origin}")
    command = controller.get("command")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise AuditError(f"controller snapshot lacks its executed command: {origin}")
    profile = Path(command[0]).name
    if profile not in KNOWN_CONTROLLERS:
        raise AuditError(f"unsupported frozen controller {profile!r}: {origin}")
    expected_max_tokens = (
        SINGLE_CONTROL_PROFILES.get(profile, (None, 1))[1]
        if profile != "rounded_residual_control.py"
        else 32
    )
    if controller.get("max_tokens") != expected_max_tokens:
        raise AuditError(
            f"controller snapshot lacks max_tokens={expected_max_tokens!r}: {origin}"
        )
    cells = controller.get("cells")
    expected_cells = {
        "attention_boundary_localise.py": ["compiled-attention", "eager-attention"],
        "post_attention_operator_localise.py": ["compiled-operator", "eager-boundary"],
    }.get(profile)
    if expected_cells is not None and cells != expected_cells:
        raise AuditError(f"unexpected cell plan for {profile}: {cells!r}")
    if profile == "moe_boundary_localise.py":
        valid = {
            ("compiled-baseline", "compiled-recorded", "eager-recorded"),
            ("compiled-recorded", "eager-recorded"),
        }
        if not isinstance(cells, list) or tuple(cells) not in valid:
            raise AuditError(f"unexpected cell plan for {profile}: {cells!r}")
    elif profile in SINGLE_CONTROL_PROFILES:
        expected_cell, _ = SINGLE_CONTROL_PROFILES[profile]
        if controller.get("cell") != expected_cell:
            raise AuditError(f"unexpected control cell for {profile}: {controller.get('cell')!r}")
        expected_control = {
            "precision_cast_control.py": {
                "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
                "torch._inductor.config.emulate_precision_casts": True,
            },
            "inductor_reuse_control.py": {
                "torch._inductor.config.allow_buffer_reuse": False,
                "torch._inductor.config.inplace_buffers": False,
            },
            "vllm_c_rms_control.py": {
                "expected_lowered_op": "torch.ops._C.fused_add_rms_norm",
                "vllm.ir.ops.fused_add_rms_norm.priority": ["vllm_c"],
            },
        }[profile]
        if controller.get("control") != expected_control:
            raise AuditError(f"unexpected control configuration for {profile}: {origin}")
    elif profile == "rounded_residual_control.py":
        cell = controller.get("cell")
        control = controller.get("control")
        expected_round = {"compiled-opaque-identity": False, "compiled-opaque-rounded": True}.get(cell)
        expected_control = {
            "barrier_output_dtype": "torch.float32",
            "round_residual": expected_round,
            "vllm.ir.ops.fused_add_rms_norm.native": "opaque residual barrier",
        }
        if expected_round is None or control != expected_control:
            raise AuditError(f"rounded-residual control identity mismatch: {origin}")
    return profile


def comparator_for(
    root: Path,
    handoff: Path,
    controller: dict[str, Any],
    comparison: dict[str, Any],
    dependency_root: Path | None = None,
) -> tuple[str, Path, dict[str, Path]]:
    """Resolve only explicitly reviewed frozen controller layouts."""
    profile = controller_profile(controller, str(handoff / "controller_manifest.json"))
    dependency = handoff / "moe_boundary_localise.py"
    if not dependency.is_file() or sha256_file(dependency) != FROZEN_COMPARATOR_SHA256:
        raise AuditError(f"frozen MoE comparator/dependency differs: {dependency}")
    if profile == "moe_boundary_localise.py":
        return profile, dependency, {}
    if profile in {*SINGLE_CONTROL_PROFILES, "rounded_residual_control.py"}:
        cell_name = controller["cell"]
        source = root / cell_name / profile
        if not source.is_file() or sha256_file(source) != KNOWN_CONTROLLERS[profile]:
            raise AuditError(f"frozen control source differs: {source}")
        return profile, source, {"moe_boundary_localise.py": dependency}
    cells = comparison.get("cells")
    if not isinstance(cells, list) or not cells or not isinstance(cells[-1], dict):
        raise AuditError(f"attention comparison lacks a completed cell: {handoff}")
    cell_name = cells[-1].get("name")
    if profile == "attention_boundary_localise.py":
        if cell_name not in {"compiled-attention", "eager-attention"}:
            raise AuditError(f"unexpected attention comparison cell: {cell_name!r}")
        source = root / cell_name / profile
        if not source.is_file() or sha256_file(source) != ATTENTION_COMPARATOR_SHA256:
            raise AuditError(f"frozen attention controller differs: {source}")
        return profile, source, {"moe_boundary_localise.py": dependency}
    if cell_name not in {"compiled-operator", "eager-boundary"}:
        raise AuditError(f"unexpected post-operator comparison cell: {cell_name!r}")
    source = root / cell_name / profile
    if not source.is_file() or sha256_file(source) != POST_OPERATOR_COMPARATOR_SHA256:
        raise AuditError(f"frozen post-operator controller differs: {source}")
    attention = (
        dependency_root / "attention_boundary_localise.py"
        if dependency_root is not None
        else Path(__file__).resolve().with_name("attention_boundary_localise.py")
    )
    if not attention.is_file() or sha256_file(attention) != ATTENTION_COMPARATOR_SHA256:
        raise AuditError(f"frozen attention dependency differs: {attention}")
    return profile, source, {
        "moe_boundary_localise.py": dependency,
        "attention_boundary_localise.py": attention,
    }


def invoke_comparison(module: Any, profile: str, root: Path, cells: list[dict[str, Any]]) -> dict[str, Any]:
    if profile == "post_attention_operator_localise.py":
        generated: dict[str, Any] = {"schema": 1, "cells": cells}
        if len(cells) == 2 and all(cell.get("returncode") == 0 for cell in cells):
            generated.update(module.analyze(root))
        return generated
    return module.compare_cells(root, cells)


def invoke_control_comparison(
    profile: str,
    root: Path,
    controller: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the single-cell controller snapshot from admitted raw result."""
    cell = controller["cell"]
    result_path = root / cell / "result.json"
    result = read_json(result_path) if result_path.is_file() else None
    generated = {
        "schema": 1,
        "cell": cell,
        # A successful immutable handoff can only be emitted after this process
        # exits cleanly.  Reconstruct these fields rather than copying them from
        # the snapshot that is being checked.
        "returncode": 0,
        "timed_out": False,
        "result": result,
    }
    if profile == "rounded_residual_control.py":
        generated["round_residual"] = controller["control"]["round_residual"]
    return generated


def validate_control_evidence(profile: str, root: Path, controller: dict[str, Any]) -> None:
    """Verify that the requested control reached runtime and generated code."""
    cell = controller["cell"]
    cell_root = root / cell
    resolved = read_json(cell_root / "resolved.json")
    source = read_json(cell_root / "generated_source_audit.json")
    result = read_json(cell_root / "result.json")
    token_rows = result.get("output", {}).get("token_ids") if isinstance(result, dict) else None
    expected_tokens = SINGLE_CONTROL_PROFILES.get(profile, (None, 32))[1]
    if (
        result.get("model") != MODEL
        or result.get("revision") != REVISION
        or not isinstance(token_rows, list)
        or len(token_rows) != 16
        or any(not isinstance(row, list) or len(row) != expected_tokens for row in token_rows)
    ):
        raise AuditError(f"control result lacks 16 complete {expected_tokens}-token rows: {cell}")
    if (
        resolved.get("mode") != 3
        or resolved.get("cudagraph_mode") != "NONE"
        or resolved.get("use_v2_model_runner") is not True
        or resolved.get("vllm") != "0.28.0"
        or not isinstance(source.get("moe_forward_shared_occurrences"), int)
        or source["moe_forward_shared_occurrences"] <= 0
    ):
        raise AuditError(f"control runtime/source evidence is incomplete: {cell}")
    if profile == "precision_cast_control.py":
        environment = read_json(cell_root / "environment.json")
        inductor = resolved.get("inductor", {})
        valid = (
            environment.get("TORCHINDUCTOR_EMULATE_PRECISION_CASTS") == "1"
            and inductor.get("emulate_precision_casts") is True
            and source.get("enable_fp_fusion_false_occurrences", 0) > 0
        )
    elif profile == "rounded_residual_control.py":
        override = resolved.get("fused_add_rms_norm_override", {})
        valid = (
            override.get("round_residual") is controller["control"]["round_residual"]
            and override.get("barrier_output_dtype") == "torch.float32"
            and override.get("provider") == "native"
            and override.get("priority_after") == ["native"]
            and override.get("priority_before") == []
            and source.get("residual_barrier_occurrences", 0) > 0
        )
    elif profile == "vllm_c_rms_control.py":
        configured = resolved.get("configured_fused_add_rms_norm_priority")
        runtime = resolved.get("runtime_fused_add_rms_norm_priority")
        providers = resolved.get("supported_fused_add_rms_norm_providers")
        valid = (
            isinstance(configured, list)
            and configured[:1] == ["vllm_c"]
            and isinstance(runtime, list)
            and runtime[:1] == ["vllm_c"]
            and isinstance(providers, list)
            and "vllm_c" in providers
            and source.get("cuda_fused_add_rms_norm_occurrences", 0) > 0
            and source.get("unlowered_ir_fused_add_rms_norm_occurrences") == 0
        )
    else:
        before, after = resolved.get("inductor_before", {}), resolved.get("inductor_after", {})
        valid = (
            before.get("allow_buffer_reuse") is True
            and before.get("inplace_buffers") is True
            and after == {"allow_buffer_reuse": False, "inplace_buffers": False}
            and source.get("reuse_markers") == 0
        )
    if not valid:
        raise AuditError(f"requested control was not observed in runtime/source evidence: {cell}")


def comparisons_match(profile: str, expected: dict[str, Any], generated: dict[str, Any]) -> bool:
    """Accept only documented CPU reduction drift for post-operator metrics."""
    if profile != "post_attention_operator_localise.py":
        return generated == expected
    expected_other = {key: value for key, value in expected.items() if key != "comparisons"}
    generated_other = {key: value for key, value in generated.items() if key != "comparisons"}
    if expected_other != generated_other:
        return False
    left, right = expected.get("comparisons"), generated.get("comparisons")
    if left is None and right is None:
        return True
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right) or len(left) != 6:
        return False
    formula = "compiled_norm_vs_fp32_formula"
    exact_fields = {"same_shape", "exact", "elements_differing", "rows_differing", "max_abs"}
    for name in left:
        if not isinstance(left[name], dict) or not isinstance(right[name], dict):
            return False
        if name == formula:
            for value in (left[name], right[name]):
                if not (
                    value.get("same_shape") is True
                    and value.get("exact") is False
                    and isinstance(value.get("elements_differing"), int)
                    and value["elements_differing"] > 0
                    and isinstance(value.get("rows_differing"), int)
                    and value["rows_differing"] > 0
                    and 0.0 < value.get("max_abs", float("inf")) <= 0.01
                    and 0.0 < value.get("mean_abs", float("inf")) <= 1e-7
                ):
                    return False
            continue
        if any(left[name].get(field) != right[name].get(field) for field in exact_fields):
            return False
        if not math.isclose(
            float(left[name].get("mean_abs", float("nan"))),
            float(right[name].get("mean_abs", float("nan"))),
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            return False
    return True


def recompute_snapshots(root: Path, admitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute every incremental snapshot with its byte-attested comparator."""
    checks = []
    ordered = []
    for item in admitted:
        handoff = root / "handoff" / item["label"]
        comparison_path = handoff / "comparison_before_ack.json"
        controller_path = handoff / "controller_manifest.json"
        for path in (comparison_path, controller_path):
            if not path.is_file():
                raise AuditError(f"handoff lacks required immutable snapshot: {path}")
        comparison = read_json(comparison_path)
        controller = read_json(controller_path)
        profile = controller_profile(controller, str(controller_path))
        if profile in {*SINGLE_CONTROL_PROFILES, "rounded_residual_control.py"}:
            order = 1
        else:
            cells = comparison.get("cells") if isinstance(comparison, dict) else None
            if not isinstance(cells, list):
                raise AuditError(f"comparison snapshot lacks cells: {comparison_path}")
            order = len(cells)
        ordered.append((order, item["label"], item, comparison, controller_path, controller))
    for _, label, item, expected, controller_path, controller in sorted(ordered):
        profile, source, dependencies = comparator_for(
            root, root / "handoff" / label, controller, expected
        )
        source_digest = sha256_file(source)
        if profile in {*SINGLE_CONTROL_PROFILES, "rounded_residual_control.py"}:
            validate_control_evidence(profile, root, controller)
            generated = invoke_control_comparison(profile, root, controller, expected)
            cell_count = 1
        else:
            module = load_comparator(source, dependencies)
            generated = invoke_comparison(module, profile, root, expected["cells"])
            cell_count = len(expected["cells"])
        if not comparisons_match(profile, expected, generated):
            raise AuditError(f"frozen compare_cells result differs from immutable snapshot: {label}")
        checks.append(
            {
                "label": label,
                "cells": cell_count,
                "recorded_comparison_sha256": sha256_bytes(canonical_json(expected)),
                "comparison_sha256": sha256_bytes(canonical_json(generated)),
                "controller": profile,
                "comparator_sha256": source_digest,
                "dependency_sha256": {
                    name: sha256_file(path) for name, path in dependencies.items()
                },
                "trace_comparison_reproduced": "trace_comparison" in generated,
                "raw_tensor_comparison_reproduced": "raw_tensor_comparison" in generated,
                "recomputed_post_operator_metrics": (
                    generated.get("comparisons")
                    if profile == "post_attention_operator_localise.py"
                    else None
                ),
            }
        )
    return checks


def source_archive(results: Path, label: str, rows: list[dict[str, Any]]) -> tuple[bytes, list[dict[str, Any]]]:
    selected = []
    for row in rows:
        relative = safe_relative(row["path"])
        if is_cache(relative) and relative.suffix.lower() in TEXT_CACHE_SUFFIXES:
            selected.append((relative.as_posix(), results.joinpath(*relative.parts).read_bytes(), row))
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data, _ in sorted(selected):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    index = [
        {"path": name, "size": row["size"], "sha256": row["sha256"]}
        for name, _, row in sorted(selected)
    ]
    return raw.getvalue(), index


def write_sums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths)
    )


def stage_public(results: Path, ack_dir: Path, staging: Path) -> dict[str, Any]:
    if staging.exists():
        raise AuditError(f"refuse existing staging directory: {staging}")
    admitted = admitted_handoffs(results, ack_dir)
    private_checks = recompute_snapshots(results, admitted)
    staging.mkdir(parents=True)
    profiles = {row["label"]: row["controller"] for row in private_checks}
    dependency_records = []
    if "post_attention_operator_localise.py" in profiles.values():
        dependency = Path(__file__).resolve().with_name("attention_boundary_localise.py")
        if sha256_file(dependency) != ATTENTION_COMPARATOR_SHA256:
            raise AuditError(f"frozen attention dependency differs: {dependency}")
        target = staging / "frozen_dependencies" / dependency.name
        target.parent.mkdir(parents=True)
        target.write_bytes(dependency.read_bytes())
        dependency_records.append(
            {
                "path": target.relative_to(staging).as_posix(),
                "size": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    label_reports = []
    mappings = []
    for item in admitted:
        label = item["label"]
        target_handoff = staging / "handoff" / label
        target_handoff.mkdir(parents=True)
        shutil.copyfile(item["manifest_path"], target_handoff / "manifest.json")
        shutil.copyfile(item["ack_path"], target_handoff / "verified-ack.json")
        rows = item["manifest"]["files"]
        archive_bytes, archive_index = source_archive(results, label, rows)
        archive_relative = PurePosixPath("compiler_sources") / f"{label}.sources-ir.tar.gz"
        if archive_index:
            archive_path = staging.joinpath(*archive_relative.parts)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(archive_bytes)
        archive_by_path = {row["path"]: row for row in archive_index}
        counts = {
            "raw_gzip": 0,
            "raw_tensor_gzip": 0,
            "compiler_source": 0,
            "hash_only": 0,
        }
        for row in rows:
            relative = safe_relative(row["path"])
            base = {"label": label, **row}
            if relative.as_posix() in archive_by_path:
                mapping = {
                    **base,
                    "storage": "compiler_source",
                    "archive": archive_relative.as_posix(),
                    "member": relative.as_posix(),
                }
            elif (
                profiles[label] == "post_attention_operator_localise.py"
                and relative.as_posix() in POST_OPERATOR_PUBLIC_TENSORS
            ):
                source = results.joinpath(*relative.parts)
                public_relative = PurePosixPath("raw") / PurePosixPath(relative.as_posix() + ".gz")
                target = staging.joinpath(*public_relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(deterministic_gzip(source.read_bytes()))
                mapping = {
                    **base,
                    "storage": "raw_tensor_gzip",
                    "public_path": public_relative.as_posix(),
                    "reason": "required_for_six_metric_post_operator_recomputation",
                }
            elif is_cache(relative) or is_tensor(relative):
                reason = "private_cache_binary_or_metadata"
                if is_tensor(relative):
                    reason = (
                        "unused_by_six_metric_post_operator_comparison"
                        if profiles[label] == "post_attention_operator_localise.py"
                        and relative.as_posix() in POST_OPERATOR_UNUSED_LOADER_TENSORS
                        else "private_saved_tensor"
                    )
                mapping = {
                    **base,
                    "storage": "hash_only",
                    "reason": reason,
                }
            else:
                source = results.joinpath(*relative.parts)
                public_relative = PurePosixPath("raw") / PurePosixPath(relative.as_posix() + ".gz")
                target = staging.joinpath(*public_relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(deterministic_gzip(source.read_bytes()))
                mapping = {**base, "storage": "raw_gzip", "public_path": public_relative.as_posix()}
            counts[mapping["storage"]] += 1
            mappings.append(mapping)
        label_reports.append(
            {
                "label": label,
                "manifest_sha256": item["manifest"]["sha256"],
                "manifest_files": len(rows),
                "storage_counts": counts,
                "compiler_source_archive": archive_relative.as_posix() if archive_index else None,
                "compiler_source_archive_sha256": sha256_bytes(archive_bytes) if archive_index else None,
                "compiler_source_files": len(archive_index),
            }
        )
    bundle = {
        "schema": 1,
        "model": MODEL,
        "revision": REVISION,
        "frozen_comparator_sha256": FROZEN_COMPARATOR_SHA256,
        "known_controller_sha256": {
            name: KNOWN_CONTROLLERS[name]
            for name in sorted(
                {"moe_boundary_localise.py", *profiles.values()}
                | ({"attention_boundary_localise.py"} if "post_attention_operator_localise.py" in profiles.values() else set())
            )
        },
        "frozen_dependencies": dependency_records,
        "labels": label_reports,
        "files": mappings,
        "private_recomputation": private_checks,
        "omission_policy": (
            "Cache binaries/metadata and tensors unused by the reviewed comparison remain in the private durable root. "
            "Tensors required for all six post-operator metrics are included byte-exact. Every omission retains its "
            "immutable manifest size and SHA-256 row."
        ),
    }
    (staging / "bundle.json").write_bytes(canonical_json(bundle))
    write_sums(staging)
    return bundle


def verify_sums(root: Path) -> int:
    sums = root / "SHA256SUMS.txt"
    if not sums.is_file():
        raise AuditError("public bundle lacks SHA256SUMS.txt")
    expected: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        try:
            digest, relative_text = line.split("  ", 1)
        except ValueError as error:
            raise AuditError("malformed SHA256SUMS.txt") from error
        relative = safe_relative(relative_text)
        if relative.as_posix() in expected or len(digest) != 64:
            raise AuditError("duplicate or malformed SHA256SUMS row")
        expected[relative.as_posix()] = digest
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path != sums
    }
    if expected != actual:
        raise AuditError("SHA256SUMS does not exactly cover the public bundle")
    return len(actual)


def extract_sources(path: Path) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != sorted(names) or len(names) != len(set(names)):
                raise AuditError(f"source archive members are not sorted and unique: {path}")
            for member in members:
                relative = safe_relative(member.name)
                if (
                    not member.isfile()
                    or relative.suffix.lower() not in TEXT_CACHE_SUFFIXES
                    or (member.mtime, member.uid, member.gid, member.mode) != (0, 0, 0, 0o644)
                ):
                    raise AuditError(f"invalid public compiler-source member: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise AuditError(f"cannot read public compiler-source member: {member.name}")
                values[relative.as_posix()] = stream.read()
    except (OSError, tarfile.TarError) as error:
        raise AuditError(f"cannot read source archive {path}: {error}") from error
    return values


def reconstruct_raw(staging: Path, mappings: list[dict[str, Any]], target: Path) -> None:
    for row in mappings:
        if row.get("storage") not in {"raw_gzip", "raw_tensor_gzip"}:
            continue
        relative = safe_relative(row["path"])
        public = safe_relative(row["public_path"])
        data = gunzip(staging.joinpath(*public.parts))
        if len(data) != row["size"] or sha256_bytes(data) != row["sha256"]:
            raise AuditError(f"public raw file differs from immutable row: {relative}")
        output = target.joinpath(*relative.parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)


def public_recompute(root: Path, admitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    ordered = []
    for item in admitted:
        handoff = root / "handoff" / item["label"]
        comparison = read_json(handoff / "comparison_before_ack.json")
        controller_value = read_json(handoff / "controller_manifest.json")
        profile = controller_profile(controller_value, str(handoff / "controller_manifest.json"))
        if profile in {*SINGLE_CONTROL_PROFILES, "rounded_residual_control.py"}:
            order = 1
        else:
            cells = comparison.get("cells") if isinstance(comparison, dict) else None
            if not isinstance(cells, list):
                raise AuditError(f"comparison snapshot lacks cells: {item['label']}")
            order = len(cells)
        ordered.append((order, item["label"], comparison, handoff, controller_value))
    for _, label, expected, handoff, controller_value in sorted(ordered):
        controller = handoff / "controller_manifest.json"
        profile, source, dependencies = comparator_for(
            root,
            handoff,
            controller_value,
            expected,
            dependency_root=root / "frozen_dependencies",
        )
        # Tensor payloads intentionally remain private.  Preserve raw trace
        # evidence, but remove only tensor_file pointers from a temporary copy
        # so the frozen comparator can reproduce every hash-based field.
        stripped_tensor_pointers = 0
        original_traces: dict[Path, bytes] = {}
        cells = expected.get("cells", [])
        for cell in cells:
            trace_path = root / cell["name"] / "trace.json"
            if not trace_path.is_file():
                continue
            original_traces[trace_path] = trace_path.read_bytes()
            trace = read_json(trace_path)
            changed = False
            for event in trace.get("events", []):
                if "tensor_file" in event:
                    event.pop("tensor_file")
                    stripped_tensor_pointers += 1
                    changed = True
            if changed:
                trace_path.write_bytes(canonical_json(trace))
        placeholder_paths: list[Path] = []
        if profile == "post_attention_operator_localise.py" and len(expected["cells"]) == 2:
            donor = root / "compiled-operator/tensors/norm_weight.pt"
            if not donor.is_file():
                raise AuditError("public post-operator bundle lacks required tensor donor")
            for relative_text in sorted(POST_OPERATOR_UNUSED_LOADER_TENSORS):
                placeholder = root.joinpath(*PurePosixPath(relative_text).parts)
                if not placeholder.exists():
                    placeholder.parent.mkdir(parents=True, exist_ok=True)
                    placeholder.write_bytes(donor.read_bytes())
                    placeholder_paths.append(placeholder)
        try:
            if profile in {*SINGLE_CONTROL_PROFILES, "rounded_residual_control.py"}:
                validate_control_evidence(profile, root, controller_value)
                generated = invoke_control_comparison(profile, root, controller_value, expected)
                cell_count = 1
            else:
                module = load_comparator(source, dependencies)
                generated = invoke_comparison(module, profile, root, cells)
                cell_count = len(cells)
        finally:
            for trace_path, data in original_traces.items():
                trace_path.write_bytes(data)
            for placeholder in placeholder_paths:
                placeholder.unlink()
        comparable_expected = dict(expected)
        tensor_metrics = comparable_expected.pop("raw_tensor_comparison", None)
        if not comparisons_match(profile, comparable_expected, generated):
            raise AuditError(f"public frozen compare_cells result differs from immutable snapshot: {label}")
        checks.append(
            {
                "label": label,
                "cells": cell_count,
                "recorded_comparison_sha256": sha256_bytes(canonical_json(expected)),
                "comparison_sha256": sha256_bytes(canonical_json(generated)),
                "controller": profile,
                "comparator_sha256": sha256_file(source),
                "trace_comparison_reproduced": "trace_comparison" in generated,
                "six_post_operator_metrics_reproduced": (
                    len(generated.get("comparisons", {})) == 6
                    if profile == "post_attention_operator_localise.py"
                    else False
                ),
                "unused_loader_tensor_placeholders": len(placeholder_paths),
                "recomputed_post_operator_metrics": (
                    generated.get("comparisons")
                    if profile == "post_attention_operator_localise.py"
                    else None
                ),
                "post_operator_float_validation": (
                    "five direct comparisons require exact counts/maxima and tightly matched means; "
                    "the fp32 formula requires independently recomputed bounded nonidentity because CPU "
                    "torch/architecture rsqrt rounding differs"
                    if profile == "post_attention_operator_localise.py" and len(cells) == 2
                    else None
                ),
                "private_raw_tensor_metrics_attested_only": tensor_metrics is not None,
                "private_tensor_pointers_removed_for_recomputation": stripped_tensor_pointers,
            }
        )
    return checks


def audit_public(staging: Path) -> dict[str, Any]:
    files_verified = verify_sums(staging)
    bundle = read_json(staging / "bundle.json")
    if not isinstance(bundle, dict) or bundle.get("schema") != 1:
        raise AuditError("invalid public bundle manifest")
    if (bundle.get("model"), bundle.get("revision"), bundle.get("frozen_comparator_sha256")) != (
        MODEL,
        REVISION,
        FROZEN_COMPARATOR_SHA256,
    ):
        raise AuditError("public bundle scientific identity mismatch")
    retained_allowlist = bundle.get("known_controller_sha256")
    if retained_allowlist is not None:
        if (
            not isinstance(retained_allowlist, dict)
            or not retained_allowlist
            or any(KNOWN_CONTROLLERS.get(name) != digest for name, digest in retained_allowlist.items())
        ):
            raise AuditError("public bundle controller allowlist mismatch")
    # Early schema-1 scan/attention bundles predate this inventory.  Profiles
    # needing an external dependency still fail closed in comparator_for.
    dependency_rows = bundle.get("frozen_dependencies", [])
    if not isinstance(dependency_rows, list):
        raise AuditError("public bundle lacks frozen dependency inventory")
    dependencies: dict[str, bytes] = {}
    for row in dependency_rows:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise AuditError("invalid frozen dependency row")
        relative = safe_relative(row["path"])
        path = staging.joinpath(*relative.parts)
        data = path.read_bytes() if path.is_file() else b""
        if len(data) != row["size"] or sha256_bytes(data) != row["sha256"]:
            raise AuditError(f"frozen dependency mismatch: {relative}")
        if relative.name != "attention_boundary_localise.py" or row["sha256"] != ATTENTION_COMPARATOR_SHA256:
            raise AuditError(f"unsupported frozen dependency: {relative}")
        dependencies[relative.name] = data
    mappings = bundle.get("files")
    labels = bundle.get("labels")
    if not isinstance(mappings, list) or not isinstance(labels, list):
        raise AuditError("public bundle lacks labels/files")
    mapping_by_label: dict[str, list[dict[str, Any]]] = {}
    for row in mappings:
        if not isinstance(row, dict) or row.get("storage") not in {
            "raw_gzip", "raw_tensor_gzip", "compiler_source", "hash_only"
        }:
            raise AuditError("invalid public storage mapping")
        mapping_by_label.setdefault(row.get("label"), []).append(row)
    admitted = []
    source_archives: dict[str, dict[str, bytes]] = {}
    handoff_files_verified = 0
    hash_only = 0
    for summary in labels:
        label = summary.get("label") if isinstance(summary, dict) else None
        if not isinstance(label, str):
            raise AuditError("invalid public label summary")
        handoff = staging / "handoff" / label
        manifest = read_json(handoff / "manifest.json")
        rows = validate_manifest_value(manifest, label=label, origin=str(handoff / "manifest.json"))
        ack = read_json(handoff / "verified-ack.json")
        validate_ack(ack, label, manifest["sha256"], str(handoff / "verified-ack.json"))
        if manifest["sha256"] != summary.get("manifest_sha256") or len(rows) != summary.get("manifest_files"):
            raise AuditError(f"public label summary differs from manifest: {label}")
        mapped = mapping_by_label.get(label, [])
        if [(x["path"], x["size"], x["sha256"]) for x in mapped] != [
            (x["path"], x["size"], x["sha256"]) for x in rows
        ]:
            raise AuditError(f"public storage mapping does not exactly cover manifest: {label}")
        for row in mapped:
            relative = safe_relative(row["path"])
            storage = row["storage"]
            if storage in {"raw_gzip", "raw_tensor_gzip"}:
                public = safe_relative(row["public_path"])
                data = gunzip(staging.joinpath(*public.parts))
                if len(data) != row["size"] or sha256_bytes(data) != row["sha256"]:
                    raise AuditError(f"public raw evidence mismatch: {relative}")
            elif storage == "compiler_source":
                archive_relative = safe_relative(row["archive"])
                archive_key = archive_relative.as_posix()
                if archive_key not in source_archives:
                    source_archives[archive_key] = extract_sources(staging.joinpath(*archive_relative.parts))
                member = safe_relative(row["member"]).as_posix()
                data = source_archives[archive_key].get(member)
                if data is None or len(data) != row["size"] or sha256_bytes(data) != row["sha256"]:
                    raise AuditError(f"public compiler source mismatch: {relative}")
            else:
                hash_only += 1
            handoff_files_verified += 1
        admitted.append({"label": label, "manifest": manifest})
    if set(mapping_by_label) != {item["label"] for item in admitted}:
        raise AuditError("public mappings contain unknown labels")
    with tempfile.TemporaryDirectory(prefix="moe-public-audit-") as temporary:
        reconstructed = Path(temporary)
        reconstruct_raw(staging, mappings, reconstructed)
        for name, data in dependencies.items():
            target = reconstructed / "frozen_dependencies" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        recomputations = public_recompute(reconstructed, admitted)
    used_controllers = {row["controller"] for row in recomputations}
    if retained_allowlist is None:
        # The first schema-1 MoE scan predates the explicit allowlist field.
        # Its sole controller remains bound by frozen_comparator_sha256 and by
        # comparator_for's byte check above.  Later profiles must carry their
        # explicit retained allowlist.
        if used_controllers != {"moe_boundary_localise.py"}:
            raise AuditError("public bundle lacks controller allowlist for a non-legacy profile")
    elif not used_controllers.issubset(retained_allowlist):
        raise AuditError("public bundle controller allowlist omits a used controller")
    return {
        "schema": 1,
        "status": "verified",
        "labels_verified": len(admitted),
        "bundle_files_verified": files_verified,
        "handoff_files_verified": handoff_files_verified,
        "hash_only_private_files": hash_only,
        "compiler_source_archives_verified": len(source_archives),
        "frozen_dependencies_verified": len(dependencies),
        "comparisons_recomputed": len(recomputations),
        "recomputations": recomputations,
    }


def cycle_period(token_ids: list[int], tail: int = 16, max_period: int = 3) -> int | None:
    """Use the existing compiled-MoE audit's short-tail cycle classifier."""
    tail_ids = token_ids[-tail:]
    for period in range(1, max_period + 1):
        if len(tail_ids) > period and all(
            tail_ids[index] == tail_ids[index + period]
            for index in range(len(tail_ids) - period)
        ):
            return period
    return None


def public_control(staging: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = read_json(staging / "bundle.json")
    mappings = bundle.get("files") if isinstance(bundle, dict) else None
    if not isinstance(mappings, list):
        raise AuditError(f"control bundle lacks file mappings: {staging}")
    controllers, results = [], []
    for row in mappings:
        if row.get("storage") != "raw_gzip":
            continue
        relative = safe_relative(row["path"])
        if relative.name not in {"controller_manifest.json", "result.json"}:
            continue
        public = safe_relative(row["public_path"])
        value = json.loads(gunzip(staging.joinpath(*public.parts)))
        (controllers if relative.name == "controller_manifest.json" else results).append(value)
    if len(controllers) != 1 or len(results) != 1:
        raise AuditError(f"expected one control manifest/result in {staging}")
    return controllers[0], results[0]


def control_token_comparison(identity: dict[str, Any], rounded: dict[str, Any]) -> dict[str, Any]:
    for name, value in (("identity", identity), ("rounded", rounded)):
        if value.get("model") != MODEL or value.get("revision") != REVISION:
            raise AuditError(f"{name} control model/revision mismatch")
        sampling = value.get("sampling")
        outputs = value.get("output")
        if not isinstance(sampling, dict) or sampling.get("max_tokens") != 32:
            raise AuditError(f"{name} control does not attest 32-token decoding")
        if not isinstance(outputs, dict) or not isinstance(outputs.get("token_ids"), list):
            raise AuditError(f"{name} control lacks raw token IDs")
    if identity.get("prompts") != rounded.get("prompts") or identity.get("prompt_token_ids") != rounded.get("prompt_token_ids"):
        raise AuditError("opaque controls did not use identical prompts")
    left, right = identity["output"]["token_ids"], rounded["output"]["token_ids"]
    if len(left) != 16 or len(right) != 16 or any(len(row) != 32 for row in left + right):
        raise AuditError("opaque controls do not contain 16 matched 32-token rows")
    rows = []
    for index, (a, b) in enumerate(zip(left, right)):
        first = next((offset for offset, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
        rows.append(
            {
                "prompt_index": index,
                "identical": a == b,
                "first_difference": first,
                "identity_cycle_period_last_16": cycle_period(a),
                "rounded_cycle_period_last_16": cycle_period(b),
            }
        )
    return {
        "schema": 1,
        "classifier": "last 16 tokens are periodic with the smallest period <= 3",
        "matched_prompts": len(rows),
        "identical_token_rows": sum(row["identical"] for row in rows),
        "identity_short_cycles": sum(row["identity_cycle_period_last_16"] is not None for row in rows),
        "rounded_short_cycles": sum(row["rounded_cycle_period_last_16"] is not None for row in rows),
        "rows": rows,
    }


def audit_control_pair(identity_root: Path, rounded_root: Path) -> dict[str, Any]:
    identity_audit, rounded_audit = audit_public(identity_root), audit_public(rounded_root)
    identity_controller, identity = public_control(identity_root)
    rounded_controller, rounded = public_control(rounded_root)
    if (
        controller_profile(identity_controller, str(identity_root)) != "rounded_residual_control.py"
        or controller_profile(rounded_controller, str(rounded_root)) != "rounded_residual_control.py"
        or identity_controller["control"].get("round_residual") is not False
        or rounded_controller["control"].get("round_residual") is not True
    ):
        raise AuditError("opaque identity/rounded bundle roles are invalid")
    return {
        "schema": 1,
        "status": "verified",
        "identity_bundle": str(identity_root),
        "rounded_bundle": str(rounded_root),
        "identity_bundle_manifest_sha256": sha256_file(identity_root / "bundle.json"),
        "rounded_bundle_manifest_sha256": sha256_file(rounded_root / "bundle.json"),
        "identity_audit": identity_audit,
        "rounded_audit": rounded_audit,
        "token_comparison": control_token_comparison(identity, rounded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results", type=Path, help="private MoE run output root")
    mode.add_argument("--audit-public", type=Path, help="portable public staging root")
    mode.add_argument(
        "--audit-control-pair",
        type=Path,
        nargs=2,
        metavar=("IDENTITY", "ROUNDED"),
        help="audit matched opaque identity/rounded public bundles",
    )
    parser.add_argument("--ack-dir", type=Path, help="durable verified-ACK directory")
    parser.add_argument("--stage-public", type=Path, help="new portable staging directory")
    parser.add_argument("--output", type=Path, required=True, help="new audit report JSON")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refuse existing output: {arguments.output}")
    output_resolved = arguments.output.resolve()
    protected_roots = []
    if arguments.audit_public:
        protected_roots.append(arguments.audit_public.resolve())
    if arguments.audit_control_pair:
        protected_roots.extend(path.resolve() for path in arguments.audit_control_pair)
    if arguments.stage_public:
        protected_roots.append(arguments.stage_public.resolve())
    if any(
        output_resolved == root or root in output_resolved.parents
        for root in protected_roots
    ):
        parser.error("--output must be outside the staged public bundle so SHA256SUMS stays exact")
    try:
        if arguments.audit_control_pair:
            if arguments.ack_dir or arguments.stage_public:
                parser.error("--audit-control-pair does not accept --ack-dir or --stage-public")
            report = audit_control_pair(*[path.resolve() for path in arguments.audit_control_pair])
        elif arguments.audit_public:
            if arguments.ack_dir or arguments.stage_public:
                parser.error("--audit-public does not accept --ack-dir or --stage-public")
            report = audit_public(arguments.audit_public.resolve())
        else:
            if not arguments.ack_dir or not arguments.stage_public:
                parser.error("--results requires --ack-dir and --stage-public")
            report = stage_public(
                arguments.results.resolve(),
                arguments.ack_dir.resolve(),
                arguments.stage_public.resolve(),
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(canonical_json(report))
    except AuditError as error:
        raise SystemExit(f"audit failed: {error}") from error
    print(json.dumps({"status": report.get("status", "staged"), "output": str(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
