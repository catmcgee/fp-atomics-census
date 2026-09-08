import json
from pathlib import Path

from triage.attach_runtime import collapse_cublaslt, load_reports, related_rows, summarise
from triage.summary import summarise as inventory_summary
from triage.validate import candidate_matches, check_ref


def report(tag="a", status="bitwise-identical", first="abc", **kw):
    return {"probe": "name", "env": {"run_tag": tag}, "verdict": status, "first_hash": first, "runs": [{}, {}, {}], **kw}


def test_missing_hash_and_errors_are_not_identical_or_differences():
    assert summarise({"a": report(first=None), "b": report("b", first=None)})[:2] == ("INVALID", "n/a")
    for status in ("ERROR", "SKIPPED", "REJECTED", "INVALID"):
        assert summarise({"a": report(status=status)})[0] == "INVALID"


def test_different_configuration_does_not_count_as_fresh_process_test():
    assert summarise({"a": report(comparison_key="x"), "b": report("b", comparison_key="y")})[1] == "n/a"


def test_sweep_preserves_real_counts_and_missing_members():
    probes = {f"cublaslt_mm_m{m}_n4_k4_bf16": {"a": report(), "b": report("b")} for m in (1, 2, 3)}
    summary = collapse_cublaslt(probes)
    assert summarise(summary["cuBLASLt mm bf16"]) == ("identical", "identical", 24)
    del probes["cublaslt_mm_m3_n4_k4_bf16"]["b"]
    assert summarise(collapse_cublaslt(probes)["cuBLASLt mm bf16"]) == ("identical", "n/a", 20)


def test_loader_keeps_duplicate_tags(tmp_path):
    (tmp_path / "stack/newrun").mkdir(parents=True)
    for path in (tmp_path / "stack/x.a.json", tmp_path / "stack/newrun/x.a.json"):
        path.write_text(json.dumps(report()))
    assert len(load_reports(tmp_path)["stack"]["name"]) == 2


def test_wrong_kernel_mappings_are_withdrawn():
    assert "flashinfer-0011" not in {r for r, _, _ in related_rows("fi_b12x_moe_nvfp4_tokens16_topk2")}
    assert all(r.startswith("flashinfer-") for r, _, _ in related_rows("fi_radix_topk_order_detFalse"))
    assert "vllm-0181" not in {r for r, _, _ in related_rows("cuBLASLt mm bf16")}
    assert not related_rows("vllm_qwen3_8b_bf16_tp2_custom_allreduce_x6")
    assert all(relation != "exact_kernel" for _, relation, _ in related_rows("fi_b12x_moe_nvfp4_tokens16_topk2"))


def test_candidate_link_must_match_sha_file_and_cited_range():
    row = {"engine": "vllm", "sha": "pinned", "file": "kernel.cu", "line_start": 10, "line_end": 14}
    candidate = {"engine": "vllm", "sha": "pinned", "file": "kernel.cu", "line": 12}
    assert candidate_matches(row, candidate)
    for mutation in ({"sha": "other"}, {"file": "benchmarks/other.cu"}, {"line": 15}):
        assert not candidate_matches(row, {**candidate, **mutation})


def test_reference_rejects_zero_reversed_and_missing_ranges():
    class Sources:
        def lines(self, *args): return ["a"] * 10
    for ref in ("x:0", "x:7-2", "x:11"):
        assert check_ref(Sources(), "vllm", ref)
    assert check_ref(Sources(), "vllm", "x:1-10") is None


def test_definition_records_do_not_inflate_site_counts():
    rows = [json.loads(l) for p in (Path(__file__).resolve().parents[1] / "inventory").glob("*.jsonl") for l in p.read_text().splitlines()]
    result = inventory_summary(rows)
    assert result["rows"] == result["sites"] + result["definition_only"]
    assert sum(result["totals"].values()) == result["sites"]


def test_validator_uses_pinned_blobs_and_rejects_wrong_heads(tmp_path):
    import subprocess
    from triage.validate import Checkouts
    repo = tmp_path / "vllm"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "kernel.cu").write_text("line one\nline two\n")
    git("add", "kernel.cu")
    git("commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD")
    manifest = {"vllm": {"sha": sha}}
    (repo / "kernel.cu").write_text("changed worktree\n")
    checkout = Checkouts(tmp_path, manifest)
    assert checkout.available("vllm")
    assert checkout.lines("vllm", "kernel.cu")[:2] == ["line one", "line two"]
    git("add", "kernel.cu")
    git("commit", "-qm", "changed head")
    assert not Checkouts(tmp_path, manifest).available("vllm")
    assert not Checkouts(tmp_path / "missing", manifest).available("vllm")
