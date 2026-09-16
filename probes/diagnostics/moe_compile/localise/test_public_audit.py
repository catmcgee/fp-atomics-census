from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("moe_public_audit_test_module", HERE / "public_audit.py")
localise = load("moe_boundary_localise_test_module", HERE / "moe_boundary_localise.py")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.canonical_json(value))


def result(token: int, compile_mode: str, record: bool) -> dict:
    return {
        "compile": compile_mode,
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "record": record,
        "prompts": ["p"],
        "prompt_token_ids": [[1]],
        "selected_runners": ["model.layers.0.mlp.experts"],
        "output": {
            "token_ids": [[token]],
            "texts": [str(token)],
            "duplicate_pairs": [False],
        },
    }


def trace(digest: str, tensor_file: str) -> dict:
    return {
        "events": [
            {
                "call": 0,
                "layer": "model.layers.0.mlp.experts",
                "layer_index": 0,
                "stage": "routed_output",
                "tensor": {"sha256": digest, "row_sha256": [digest]},
                "tensor_file": tensor_file,
            }
        ]
    }


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.acks = root.parent / "verified"
        self.acks.mkdir()
        self.cells: list[dict] = []
        self._add("compiled-baseline", "on", False, 7, None, None)
        self._add("compiled-recorded", "on", True, 7, "compiled", torch.tensor([1.0, 2.0]))
        self._add("eager-recorded", "off", True, 8, "eager", torch.tensor([1.0, 3.0]))

    def _add(
        self,
        name: str,
        compile_mode: str,
        record: bool,
        token: int,
        trace_digest: str | None,
        tensor: torch.Tensor | None,
    ) -> None:
        cell = self.root / name
        cell.mkdir(parents=True)
        write_json(cell / "result.json", result(token, compile_mode, record))
        if tensor is not None:
            tensors = cell / "tensors"
            tensors.mkdir()
            torch.save(tensor, tensors / "boundary.pt")
            write_json(cell / "trace.json", trace(trace_digest or "x", "tensors/boundary.pt"))
        cache = cell / "cache" / "triton" / name
        cache.mkdir(parents=True)
        (cache / "kernel.ttir").write_text(f"// {name}\n")
        (cache / "kernel.cubin").write_bytes(b"\x7fCUBIN" + name.encode())
        (self.root / f"{name}.stdout.log").write_text(f"completed {name}\n")
        (self.root / f"{name}.stderr.log").write_text("")
        self.cells.append(
            {
                "name": name,
                "compile": compile_mode,
                "record": record,
                "returncode": 0,
                "timed_out": False,
            }
        )
        label = name.replace("-", "_")
        handoff = self.root / "handoff" / label
        handoff.mkdir(parents=True)
        (handoff / "moe_boundary_localise.py").write_bytes(
            (HERE / "moe_boundary_localise.py").read_bytes()
        )
        write_json(
            handoff / "controller_manifest.json",
            {
                "schema": 1,
                "model": audit.MODEL,
                "revision": audit.REVISION,
                "max_tokens": 1,
                "command": ["/frozen/moe_boundary_localise.py"],
                "cells": ["compiled-baseline", "compiled-recorded", "eager-recorded"],
            },
        )
        write_json(
            handoff / "comparison_before_ack.json",
            localise.compare_cells(self.root, self.cells),
        )
        self.refresh(label)

    def refresh(self, label: str) -> None:
        name = label.replace("_", "-")
        targets = [
            self.root / name,
            self.root / f"{name}.stdout.log",
            self.root / f"{name}.stderr.log",
            self.root / "handoff" / label / "moe_boundary_localise.py",
            self.root / "handoff" / label / "controller_manifest.json",
            self.root / "handoff" / label / "comparison_before_ack.json",
        ]
        rows = []
        for target in targets:
            paths = [target] if target.is_file() else sorted(path for path in target.rglob("*") if path.is_file())
            for path in paths:
                data = path.read_bytes()
                rows.append(
                    {
                        "path": path.relative_to(self.root).as_posix(),
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        rows.sort(key=lambda row: row["path"])
        payload = {"schema": 1, "label": label, "files": rows}
        digest = hashlib.sha256(audit.canonical_json(payload)).hexdigest()
        write_json(self.root / "handoff" / label / "manifest.json", {**payload, "sha256": digest})
        write_json(
            self.acks / f"{label}.verified-ack.json",
            {
                "schema": 1,
                "label": label,
                "handoff_manifest_sha256": digest,
                "verified_manifest_sha256": digest,
            },
        )


def test_stages_and_recomputes_scan_with_hash_only_tensors(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path / "private")
    public = tmp_path / "public"
    staged = audit.stage_public(fixture.root, fixture.acks, public)
    report = audit.audit_public(public)
    assert [row["label"] for row in staged["labels"]] == [
        "compiled_baseline",
        "compiled_recorded",
        "eager_recorded",
    ]
    assert report["status"] == "verified"
    assert report["labels_verified"] == 3
    assert report["comparisons_recomputed"] == 3
    assert report["hash_only_private_files"] >= 5  # cubins and two tensors
    final = report["recomputations"][-1]
    assert final["trace_comparison_reproduced"]
    assert final["private_raw_tensor_metrics_attested_only"]
    assert final["private_tensor_pointers_removed_for_recomputation"] == 2
    source_archive = public / "compiler_sources/eager_recorded.sources-ir.tar.gz"
    assert set(audit.extract_sources(source_archive)) == {
        "eager-recorded/cache/triton/eager-recorded/kernel.ttir"
    }


def test_rejects_exact_ack_mismatch(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path / "private")
    ack = fixture.acks / "compiled_baseline.verified-ack.json"
    value = json.loads(ack.read_text())
    value["extra"] = True
    write_json(ack, value)
    with pytest.raises(audit.AuditError, match="invalid durable ACK"):
        audit.stage_public(fixture.root, fixture.acks, tmp_path / "public")


def test_rejects_wrong_comparison_even_when_manifest_and_ack_match(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path / "private")
    comparison = fixture.root / "handoff/eager_recorded/comparison_before_ack.json"
    value = json.loads(comparison.read_text())
    value["trace_comparison"]["compiled_events"] = 999
    write_json(comparison, value)
    fixture.refresh("eager_recorded")
    with pytest.raises(audit.AuditError, match="compare_cells result differs"):
        audit.stage_public(fixture.root, fixture.acks, tmp_path / "public")


def test_public_raw_tamper_fails_after_fresh_outer_sums(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path / "private")
    public = tmp_path / "public"
    audit.stage_public(fixture.root, fixture.acks, public)
    bundle = json.loads((public / "bundle.json").read_text())
    row = next(item for item in bundle["files"] if item["path"].endswith("result.json"))
    raw = public / row["public_path"]
    raw.write_bytes(audit.deterministic_gzip(b"{}\n"))
    audit.write_sums(public)
    with pytest.raises(audit.AuditError, match="public raw evidence mismatch"):
        audit.audit_public(public)


def test_accepts_retained_controller_subset_but_rejects_unknown_hash(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path / "private")
    public = tmp_path / "public"
    audit.stage_public(fixture.root, fixture.acks, public)
    bundle_path = public / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["known_controller_sha256"] = {
        "moe_boundary_localise.py": audit.FROZEN_COMPARATOR_SHA256,
    }
    write_json(bundle_path, bundle)
    audit.write_sums(public)
    assert audit.audit_public(public)["status"] == "verified"

    bundle["known_controller_sha256"]["moe_boundary_localise.py"] = "0" * 64
    write_json(bundle_path, bundle)
    audit.write_sums(public)
    with pytest.raises(audit.AuditError, match="controller allowlist mismatch"):
        audit.audit_public(public)


def test_accepts_legacy_base_bundle_without_controller_allowlist(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path / "private")
    public = tmp_path / "public"
    audit.stage_public(fixture.root, fixture.acks, public)
    bundle_path = public / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle.pop("known_controller_sha256")
    write_json(bundle_path, bundle)
    audit.write_sums(public)
    assert audit.audit_public(public)["status"] == "verified"


def test_explicit_attention_controller_loads_its_frozen_dependency(tmp_path: Path) -> None:
    root = tmp_path / "attention"
    handoff = root / "handoff/compiled_attention"
    handoff.mkdir(parents=True)
    (handoff / "moe_boundary_localise.py").write_bytes(
        (HERE / "moe_boundary_localise.py").read_bytes()
    )
    cell = root / "compiled-attention"
    cell.mkdir()
    (cell / "attention_boundary_localise.py").write_bytes(
        (HERE / "attention_boundary_localise.py").read_bytes()
    )
    baseline = [
        [279], [279], [99339], [576], [5205], [220], [13], [315],
        [151643], [99483], [151643], [279], [308], [1075], [1270], [11],
    ]
    write_json(cell / "result.json", {"output": {"token_ids": baseline}})
    comparison = {
        "schema": 1,
        "cells": [
            {"name": "compiled-attention", "compile": "on", "returncode": 0, "timed_out": False}
        ],
        "instrumentation_preserves_compiled_tokens": True,
    }
    controller = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "max_tokens": 1,
        "command": ["/frozen/attention_boundary_localise.py"],
        "cells": ["compiled-attention", "eager-attention"],
    }
    profile, source, dependencies = audit.comparator_for(
        root, handoff, controller, comparison
    )
    assert profile == "attention_boundary_localise.py"
    module = audit.load_comparator(source, dependencies)
    assert module.compare_cells(root, comparison["cells"]) == comparison


def test_unknown_controller_profile_fails_closed() -> None:
    controller = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "max_tokens": 1,
        "command": ["/frozen/unreviewed.py"],
        "cells": [],
    }
    with pytest.raises(audit.AuditError, match="unsupported frozen controller"):
        audit.controller_profile(controller, "fixture")


def test_post_operator_profile_recomputes_all_six_metrics(tmp_path: Path) -> None:
    root = tmp_path / "post"
    handoff = root / "handoff/eager_boundary"
    handoff.mkdir(parents=True)
    (handoff / "moe_boundary_localise.py").write_bytes(
        (HERE / "moe_boundary_localise.py").read_bytes()
    )
    dependencies = root / "deps"
    dependencies.mkdir()
    (dependencies / "attention_boundary_localise.py").write_bytes(
        (HERE / "attention_boundary_localise.py").read_bytes()
    )
    for cell_name in ("compiled-operator", "eager-boundary"):
        cell = root / cell_name
        (cell / "tensors").mkdir(parents=True)
        (cell / "post_attention_operator_localise.py").write_bytes(
            (HERE / "post_attention_operator_localise.py").read_bytes()
        )
        token_ids = [
            [279], [279], [99339], [576], [5205], [220], [13], [315],
            [151643], [99483], [151643], [279], [308], [1075], [1270], [11],
        ]
        write_json(cell / "result.json", {"output": {"token_ids": token_ids}})
    projection = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.5, 0.25], [0.5, 0.25]], dtype=torch.bfloat16)
    weight = torch.tensor([1.0, 1.5], dtype=torch.bfloat16)
    summed = projection.float() + residual.float()
    norm = (summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + 1e-6) * weight.float()).to(torch.bfloat16)
    compiled_values = {
        "attention_output": torch.zeros(1),
        "o_projection_matrix": torch.zeros(1),
        "o_projection_output": projection,
        "norm_projection_input": torch.zeros(1),
        "residual": residual,
        "norm_weight": weight,
        "norm_output": norm,
    }
    eager_values = {
        "o_projection_output": projection,
        "residual": residual,
        "norm_weight": weight,
        "norm_output": norm.clone(),
        "updated_residual": summed.to(torch.bfloat16),
    }
    for name, value in compiled_values.items():
        torch.save(value, root / "compiled-operator/tensors" / f"{name}.pt")
    for name, value in eager_values.items():
        torch.save(value, root / "eager-boundary/tensors" / f"{name}.pt")
    cells = [
        {"name": "compiled-operator", "compile": "on", "returncode": 0, "timed_out": False},
        {"name": "eager-boundary", "compile": "off", "returncode": 0, "timed_out": False},
    ]
    controller = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "max_tokens": 1,
        "command": ["/frozen/post_attention_operator_localise.py"],
        "cells": ["compiled-operator", "eager-boundary"],
    }
    profile, source, frozen_dependencies = audit.comparator_for(
        root, handoff, controller, {"cells": cells}, dependency_root=dependencies
    )
    module = audit.load_comparator(source, frozen_dependencies)
    report = audit.invoke_comparison(module, profile, root, cells)
    assert profile == "post_attention_operator_localise.py"
    assert len(report["comparisons"]) == 6
    assert report["comparisons"]["compiled_norm_vs_eager_norm"]["exact"]
    assert len(audit.POST_OPERATOR_PUBLIC_TENSORS) == 9
    assert audit.POST_OPERATOR_PUBLIC_TENSORS.isdisjoint(
        audit.POST_OPERATOR_UNUSED_LOADER_TENSORS
    )


def test_single_control_profile_rebuilds_snapshot_from_raw_result(tmp_path: Path) -> None:
    root = tmp_path / "precision"
    cell = root / "compiled-emulate-precision-casts"
    handoff = root / "handoff/compiled_emulate_precision_casts"
    cell.mkdir(parents=True)
    handoff.mkdir(parents=True)
    (cell / "precision_cast_control.py").write_bytes(
        (HERE / "precision_cast_control.py").read_bytes()
    )
    (handoff / "moe_boundary_localise.py").write_bytes(
        (HERE / "moe_boundary_localise.py").read_bytes()
    )
    result_value = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "output": {"token_ids": [[1] for _ in range(16)]},
    }
    write_json(cell / "result.json", result_value)
    write_json(cell / "environment.json", {"TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1"})
    write_json(
        cell / "resolved.json",
        {
            "mode": 3,
            "cudagraph_mode": "NONE",
            "use_v2_model_runner": True,
            "vllm": "0.28.0",
            "inductor": {"emulate_precision_casts": True},
        },
    )
    write_json(
        cell / "generated_source_audit.json",
        {"moe_forward_shared_occurrences": 1, "enable_fp_fusion_false_occurrences": 1},
    )
    controller = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "max_tokens": 1,
        "command": ["/frozen/precision_cast_control.py"],
        "cell": "compiled-emulate-precision-casts",
        "control": {
            "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
            "torch._inductor.config.emulate_precision_casts": True,
        },
    }
    expected = {
        "schema": 1,
        "cell": "compiled-emulate-precision-casts",
        "returncode": 0,
        "timed_out": False,
        "result": result_value,
    }
    profile, source, dependencies = audit.comparator_for(root, handoff, controller, expected)
    assert profile == "precision_cast_control.py"
    assert audit.sha256_file(source) == audit.PRECISION_CAST_CONTROL_SHA256
    assert set(dependencies) == {"moe_boundary_localise.py"}
    audit.validate_control_evidence(profile, root, controller)
    assert audit.invoke_control_comparison(profile, root, controller, expected) == expected


def test_control_pair_recomputes_existing_short_cycle_classifier() -> None:
    base = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "prompts": [f"p{index}" for index in range(16)],
        "prompt_token_ids": [[index] for index in range(16)],
        "sampling": {"max_tokens": 32},
    }
    identity = {
        **base,
        "output": {"token_ids": [[index % 2] * 32 for index in range(16)]},
    }
    rounded = {
        **base,
        "output": {"token_ids": [list(range(32)) for _ in range(16)]},
    }
    report = audit.control_token_comparison(identity, rounded)
    assert report["matched_prompts"] == 16
    assert report["identical_token_rows"] == 0
    assert report["identity_short_cycles"] == 16
    assert report["rounded_short_cycles"] == 0


def test_no_reuse_profile_is_prepared_for_max_32() -> None:
    controller = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "max_tokens": 32,
        "command": ["/frozen/inductor_reuse_control.py"],
        "cell": "compiled-reuse-off",
        "control": {
            "torch._inductor.config.allow_buffer_reuse": False,
            "torch._inductor.config.inplace_buffers": False,
        },
    }
    assert audit.controller_profile(controller, "fixture") == "inductor_reuse_control.py"


def test_vllm_c_rms_profile_validates_provider_and_lowering(tmp_path: Path) -> None:
    root = tmp_path / "native-rms"
    cell = root / "compiled-vllm-c-rms"
    cell.mkdir(parents=True)
    write_json(
        cell / "result.json",
        {
            "model": audit.MODEL,
            "revision": audit.REVISION,
            "output": {"token_ids": [[1] * 32 for _ in range(16)]},
        },
    )
    write_json(
        cell / "resolved.json",
        {
            "mode": 3,
            "cudagraph_mode": "NONE",
            "use_v2_model_runner": True,
            "vllm": "0.28.0",
            "configured_fused_add_rms_norm_priority": ["vllm_c", "native"],
            "runtime_fused_add_rms_norm_priority": ["vllm_c", "native"],
            "supported_fused_add_rms_norm_providers": ["native", "vllm_c"],
        },
    )
    write_json(
        cell / "generated_source_audit.json",
        {
            "moe_forward_shared_occurrences": 1,
            "cuda_fused_add_rms_norm_occurrences": 2,
            "unlowered_ir_fused_add_rms_norm_occurrences": 0,
        },
    )
    controller = {
        "model": audit.MODEL,
        "revision": audit.REVISION,
        "max_tokens": 32,
        "command": ["/frozen/vllm_c_rms_control.py"],
        "cell": "compiled-vllm-c-rms",
        "control": {
            "expected_lowered_op": "torch.ops._C.fused_add_rms_norm",
            "vllm.ir.ops.fused_add_rms_norm.priority": ["vllm_c"],
        },
    }
    profile = audit.controller_profile(controller, "fixture")
    assert profile == "vllm_c_rms_control.py"
    audit.validate_control_evidence(profile, root, controller)
