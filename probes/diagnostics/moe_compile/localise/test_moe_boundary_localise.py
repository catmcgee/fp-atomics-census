from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


MODULE_PATH = Path(__file__).with_name("moe_boundary_localise.py")
SPEC = importlib.util.spec_from_file_location("moe_boundary_localise", MODULE_PATH)
assert SPEC and SPEC.loader
localise = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(localise)

AUDIT_PATH = Path(__file__).with_name("retained_graph_audit.py")
AUDIT_SPEC = importlib.util.spec_from_file_location("retained_graph_audit", AUDIT_PATH)
assert AUDIT_SPEC and AUDIT_SPEC.loader
graph_audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(graph_audit)

CONTROL_PATH = Path(__file__).with_name("inductor_reuse_control.py")
CONTROL_SPEC = importlib.util.spec_from_file_location("inductor_reuse_control", CONTROL_PATH)
assert CONTROL_SPEC and CONTROL_SPEC.loader
reuse_control = importlib.util.module_from_spec(CONTROL_SPEC)
CONTROL_SPEC.loader.exec_module(reuse_control)

ATTENTION_PATH = Path(__file__).with_name("attention_boundary_localise.py")
ATTENTION_SPEC = importlib.util.spec_from_file_location(
    "attention_boundary_localise", ATTENTION_PATH
)
assert ATTENTION_SPEC and ATTENTION_SPEC.loader
attention_localise = importlib.util.module_from_spec(ATTENTION_SPEC)
ATTENTION_SPEC.loader.exec_module(attention_localise)

POST_ATTENTION_PATH = Path(__file__).with_name("post_attention_operator_localise.py")
POST_ATTENTION_SPEC = importlib.util.spec_from_file_location(
    "post_attention_operator_localise", POST_ATTENTION_PATH
)
assert POST_ATTENTION_SPEC and POST_ATTENTION_SPEC.loader
post_attention_localise = importlib.util.module_from_spec(POST_ATTENTION_SPEC)
POST_ATTENTION_SPEC.loader.exec_module(post_attention_localise)

PRECISION_PATH = Path(__file__).with_name("precision_cast_control.py")
PRECISION_SPEC = importlib.util.spec_from_file_location(
    "precision_cast_control", PRECISION_PATH
)
assert PRECISION_SPEC and PRECISION_SPEC.loader
precision_control = importlib.util.module_from_spec(PRECISION_SPEC)
PRECISION_SPEC.loader.exec_module(precision_control)

ROUNDED_PATH = Path(__file__).with_name("rounded_residual_control.py")
ROUNDED_SPEC = importlib.util.spec_from_file_location(
    "rounded_residual_control", ROUNDED_PATH
)
assert ROUNDED_SPEC and ROUNDED_SPEC.loader
rounded_control = importlib.util.module_from_spec(ROUNDED_SPEC)
ROUNDED_SPEC.loader.exec_module(rounded_control)

VLLM_C_RMS_PATH = Path(__file__).with_name("vllm_c_rms_control.py")
VLLM_C_RMS_SPEC = importlib.util.spec_from_file_location(
    "vllm_c_rms_control", VLLM_C_RMS_PATH
)
assert VLLM_C_RMS_SPEC and VLLM_C_RMS_SPEC.loader
vllm_c_rms_control = importlib.util.module_from_spec(VLLM_C_RMS_SPEC)
VLLM_C_RMS_SPEC.loader.exec_module(vllm_c_rms_control)

TOKEN_AUDIT_PATH = Path(__file__).with_name("control_token_audit.py")
TOKEN_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "control_token_audit", TOKEN_AUDIT_PATH
)
assert TOKEN_AUDIT_SPEC and TOKEN_AUDIT_SPEC.loader
token_audit = importlib.util.module_from_spec(TOKEN_AUDIT_SPEC)
TOKEN_AUDIT_SPEC.loader.exec_module(token_audit)


def test_duplicate_block_report_accepts_exact_packed_pairs() -> None:
    lengths = [1, 2, 1, 2, 1, 2, 1, 2] * 2
    first = [[f"{index}:{row}" for row in range(length)] for index, length in enumerate(lengths[:8])]
    rows = [item for block in first + first for item in block]
    report = localise.duplicate_block_report(rows, lengths)
    assert report == {"valid": True, "pairs_equal": [True] * 8, "pairs_equal_count": 8}


def test_duplicate_block_report_rejects_wrong_row_count() -> None:
    report = localise.duplicate_block_report(["only-one"], [1] * 16)
    assert not report["valid"]
    assert report["rows"] == 1
    assert report["prompt_tokens"] == 16


def event(layer: int, stage: str, digest: str, rows: list[str]) -> dict:
    return {
        "call": 0,
        "layer": f"model.layers.{layer}.mlp.experts",
        "stage": stage,
        "tensor": {"sha256": digest, "row_sha256": rows},
    }


def test_compare_traces_reports_first_differing_boundary_and_rows() -> None:
    compiled = {
        "events": [
            event(0, "moe_input", "same", ["a", "b"]),
            event(0, "routed_output", "compiled", ["x", "y"]),
        ]
    }
    eager = {
        "events": [
            event(0, "moe_input", "same", ["a", "b"]),
            event(0, "routed_output", "eager", ["x", "z"]),
        ]
    }
    report = localise.compare_traces(compiled, eager)
    assert report["first_difference"]["stage"] == "routed_output"
    assert report["first_difference"]["rows_differing"] == 1
    assert report["first_differing_layer_call"]["layer_index"] == 0
    assert report["first_differing_layer_call"]["differing_stages"] == ["routed_output"]


def test_compare_traces_treats_missing_event_as_difference() -> None:
    compiled = {"events": [event(0, "moe_input", "x", ["x"])]}
    eager = {"events": []}
    report = localise.compare_traces(compiled, eager)
    assert report["first_difference"]["missing"] == "eager"


def test_compare_traces_rejects_duplicate_keys() -> None:
    duplicate = event(0, "moe_input", "x", ["x"])
    with pytest.raises(ValueError, match="duplicate event keys"):
        localise.compare_traces({"events": [duplicate, duplicate]}, {"events": [duplicate]})


def test_recorder_validation_rejects_missing_topk_stage(tmp_path: Path) -> None:
    recorder = localise.BoundaryRecorder(None, tmp_path, [1] * 16, False)
    layer = "model.layers.0.mlp.experts"
    stages = {
        "moe_input",
        "router_logits",
        "topk_ids",
        "shared_output",
        "routed_output",
        "moe_input_after",
    }
    recorder.events = [
        {"call": 0, "layer": layer, "stage": stage} for stage in sorted(stages)
    ]
    with pytest.raises(RuntimeError, match="topk_weights"):
        recorder.validate([layer])


def test_handoff_manifest_and_exact_ack(tmp_path: Path) -> None:
    output = tmp_path / "scan"
    cell = output / "compiled-baseline"
    cell.mkdir(parents=True)
    (cell / "result.json").write_text("{}\n")
    (output / "compiled-baseline.stdout.log").write_text("stdout\n")
    (output / "compiled-baseline.stderr.log").write_text("")
    (output / "manifest.json").write_text("{}\n")
    (output / "comparison.json").write_text("{}\n")
    manifest_path = localise.handoff_manifest(output, "compiled-baseline")
    manifest = json.loads(manifest_path.read_text())
    payload = {key: manifest[key] for key in ("schema", "label", "files")}
    assert manifest["sha256"] == hashlib.sha256(localise.canonical_json(payload)).hexdigest()
    paths = {item["path"] for item in manifest["files"]}
    assert "compiled-baseline/result.json" in paths
    assert "handoff/compiled_baseline/moe_boundary_localise.py" in paths
    assert "handoff/compiled_baseline/controller_manifest.json" in paths
    assert "handoff/compiled_baseline/comparison_before_ack.json" in paths
    assert "manifest.json" not in paths
    assert "comparison.json" not in paths
    ack = output / "acks" / "compiled_baseline.json"
    ack.parent.mkdir()
    ack.write_text(
        json.dumps(
            {
                "schema": 1,
                "label": "compiled_baseline",
                "handoff_manifest_sha256": manifest["sha256"],
                "verified_manifest_sha256": manifest["sha256"],
            }
        )
    )
    localise.verify_ack(ack, "compiled_baseline", manifest_path)
    (output / "manifest.json").write_text('{"later": true}\n')
    (output / "comparison.json").write_text('{"later": true}\n')
    audit = localise.audit_handoffs(output)
    assert audit["handoffs_verified"] == 1
    value = json.loads(ack.read_text())
    value["extra"] = True
    ack.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="invalid durable"):
        localise.verify_ack(ack, "compiled_baseline", manifest_path)


def test_generated_source_audit_counts_moe_calls_and_reuse(tmp_path: Path) -> None:
    source = tmp_path / "ab" / "generated.py"
    source.parent.mkdir()
    source.write_text(
        "out = torch.ops.vllm.moe_forward_shared.default(x, logits, x, None, layer, 0)\n"
        "buf8 = x; del x  # reuse\n"
    )
    report = reuse_control.generated_source_audit(tmp_path)
    assert report["python_sources"] == 1
    assert report["moe_forward_shared_occurrences"] == 1
    assert report["reuse_markers"] == 1


def test_discover_layer0_attention_is_narrow() -> None:
    class Implementation:
        def forward(self):
            return None

    class Layer:
        def __init__(self):
            self.impl = Implementation()

    class Compilation:
        static_forward_context = {
            "model.layers.0.self_attn.attn": Layer(),
            "model.layers.1.self_attn.attn": Layer(),
            "model.layers.0.mlp.experts": object(),
        }

    class Config:
        compilation_config = Compilation()

    found = attention_localise.discover_layer0_attention(Config())
    assert [name for name, _ in found] == ["model.layers.0.self_attn.attn"]


def test_attention_recorder_validation_rejects_missing_output() -> None:
    recorder = attention_localise.AttentionRecorder(None, [1] * 16)
    layer = "model.layers.0.self_attn.attn"
    recorder.events = [
        {"call": 0, "layer": layer, "stage": stage}
        for stage in ("query", "key", "value")
    ]
    with pytest.raises(RuntimeError, match="attention_output"):
        recorder.validate([layer])


def test_o_projection_shape_filter_is_strict() -> None:
    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    assert post_attention_localise.is_o_projection_call(
        Tensor((962, 2048)), Tensor((2048, 2048)), Tensor((962, 2048))
    )
    assert not post_attention_localise.is_o_projection_call(
        Tensor((962, 2048)), Tensor((2048, 64)), Tensor((962, 64))
    )
    assert not post_attention_localise.is_o_projection_call(
        Tensor((962, 2048)), Tensor((2048, 2048)), Tensor((961, 2048))
    )


def test_generated_module_selection_requires_cache_and_symbols(tmp_path: Path) -> None:
    source = tmp_path / "ab" / "generated.py"
    source.parent.mkdir()
    source.write_text("# generated\n")

    class Candidate:
        __file__ = str(source)
        Runner = object()
        extern_kernels = object()

    setattr(Candidate, post_attention_localise.KERNEL_NAME, object())
    duplicate = Candidate

    class Outside(Candidate):
        __file__ = "/tmp/outside.py"

    selected = post_attention_localise.candidate_generated_modules(
        [Candidate, duplicate, Outside], tmp_path
    )
    assert selected == [Candidate]


def test_find_layer0_norm_is_fail_closed() -> None:
    layer0, layer1 = object(), object()

    class Model:
        def named_modules(self):
            return [
                ("model.layers.0.post_attention_layernorm", layer0),
                ("model.layers.1.post_attention_layernorm", layer1),
            ]

    name, selected = post_attention_localise.find_layer0_post_attention_norm(Model())
    assert name == "model.layers.0.post_attention_layernorm"
    assert selected is layer0


def test_precision_control_source_audit_requires_generated_attestation(tmp_path: Path) -> None:
    source = tmp_path / "ab" / "generated.py"
    source.parent.mkdir()
    source.write_text(
        "torch.ops.vllm.moe_forward_shared.default(x)\n"
        "inductor_meta={'enable_fp_fusion': False}\n"
    )
    report = precision_control.generated_source_audit(tmp_path)
    assert report["moe_forward_shared_occurrences"] == 1
    assert report["enable_fp_fusion_false_occurrences"] == 1


def test_rounded_control_source_audit_records_matched_barrier(tmp_path: Path) -> None:
    source = tmp_path / "ab" / "generated.py"
    source.parent.mkdir()
    source.write_text(
        "torch.ops.vllm.moe_forward_shared.default(x)\n"
        f"rounded = {rounded_control.BARRIER_OP}(value)\n"
    )
    report = rounded_control.generated_source_audit(tmp_path)
    assert report["moe_forward_shared_occurrences"] == 1
    assert report["residual_barrier_occurrences"] == 1


def test_vllm_c_rms_source_audit_attests_lowered_cuda_op(tmp_path: Path) -> None:
    source = tmp_path / "ab" / "generated.py"
    source.parent.mkdir()
    source.write_text(
        "torch.ops.vllm.moe_forward_shared.default(x)\n"
        f"{vllm_c_rms_control.CUDA_OP}.default(x, residual, weight, 1e-6)\n"
    )
    report = vllm_c_rms_control.generated_source_audit(tmp_path)
    assert report["moe_forward_shared_occurrences"] == 1
    assert report["cuda_fused_add_rms_norm_occurrences"] == 1
    assert report["unlowered_ir_fused_add_rms_norm_occurrences"] == 0


def test_vllm_c_rms_source_audit_detects_unlowered_ir(tmp_path: Path) -> None:
    source = tmp_path / "generated.py"
    source.write_text("torch.ops.vllm_ir.fused_add_rms_norm.default(x)\n")
    report = vllm_c_rms_control.generated_source_audit(tmp_path)
    assert report["cuda_fused_add_rms_norm_occurrences"] == 0
    assert report["unlowered_ir_fused_add_rms_norm_occurrences"] == 1


def test_control_token_audit_compares_common_prefix_and_cycles(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    rows = [[index] * 40 for index in range(16)]
    left.write_text(json.dumps({"repeats": [[{"token_ids": row} for row in rows]]}))
    right.write_text(
        json.dumps({"result": {"output": {"token_ids": [row[:32] for row in rows]}}})
    )
    report = token_audit.audit(str(left), str(right))
    comparison = report["comparison"]
    assert comparison["exact_prefix_identity"] is True
    assert comparison["left_prefix"]["canonical_sha256"] == comparison["right_prefix"][
        "canonical_sha256"
    ]
    assert comparison["right_prefix"]["short_cycles"] == 16


def write_graph_archive(path: Path, build: str) -> None:
    body = (
        f"# /opt/issue56900/{build}/lib/source.py\n"
        "x = torch.ops.vllm.moe_forward_shared(hidden, logits, hidden, None, layer, 0)\n"
    ).encode()
    info = tarfile.TarInfo("./compile-on_x/cache/computation_graph.py")
    info.size = len(body)
    with tarfile.open(path, "w:gz") as bundle:
        bundle.addfile(info, io.BytesIO(body))


def test_retained_graph_audit_normalises_only_build_path(tmp_path: Path) -> None:
    left, right = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    write_graph_archive(left, "cu129")
    write_graph_archive(right, "cu130")
    report = graph_audit.audit([left, right])
    assert report["normalised_graphs_identical"]
    assert report["graphs"][0]["moe_forward_shared_calls"] == 1
    assert report["graphs"][0]["all_calls_alias_hidden_and_shared_input"]
