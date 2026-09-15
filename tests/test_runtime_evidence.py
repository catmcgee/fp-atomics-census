import json
from pathlib import Path

from triage.attach_runtime import campaign_of, campaign_rows, campaigns, collapse_cublaslt, load_reports, related_rows, summarise
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


def new_report(tag, run_id, digest="c71ba85c0851c29072a63c895cf98716707588fc35b3846f8a43b5c63d1e2be2", key="k", **kw):
    return report(tag, schema=2, comparison_key=key, env={"run_tag": tag, "run_id": run_id, "source": {"probe_source_sha256": digest}}, **kw)


def test_fresh_process_pairing_stays_within_one_campaign():
    # The H100 det_strict runs share a stack directory with three legacy reports of the same probe.
    legacy = {"x.a.json": report("a"), "x.b.json": report("b"), "x.json": report(None)}
    new = {"run1/x.a.json": new_report("a", "run1"), "run2/x.b.json": new_report("b", "run2")}
    merged = {**legacy, **new}
    assert summarise(merged) == ("identical", "n/a", 20)  # mixed campaigns cannot share a comparison key
    groups = campaigns(merged)
    assert list(groups) == ["legacy", "c71ba85c0851"]
    assert summarise(groups["legacy"]) == ("identical", "identical", 12)
    assert summarise(groups["c71ba85c0851"]) == ("identical", "identical", 8)
    assert [(label, several) for _, label, several, _ in campaign_rows({"name": merged})] == [("legacy", True), ("c71ba85c0851", True)]
    assert [(label, several) for _, label, several, _ in campaign_rows({"name": new})] == [("c71ba85c0851", False)]


def test_campaign_split_never_invents_a_verdict():
    # Two processes from different probe sources: neither group has a second identity, so both stay n/a.
    groups = campaigns({"run1/x.a.json": new_report("a", "run1", digest="a" * 64, key="k1"), "run2/x.b.json": new_report("b", "run2", digest="b" * 64, key="k2")})
    assert [summarise(g)[1] for g in groups.values()] == ["n/a", "n/a"]
    # Same source, different configuration: still one campaign and still n/a, as before.
    assert summarise(campaigns({"a": new_report("a", "run1", key="k1"), "b": new_report("b", "run2", key="k2")})["c71ba85c0851"])[1] == "n/a"
    # Aggregates carry their members' campaign.
    probes = {f"cublaslt_mm_m{m}_n4_k4_bf16": {"a": report(), "b": report("b")} for m in (1, 2)}
    assert campaign_of(collapse_cublaslt(probes)["cuBLASLt mm bf16"]["a"]) == "legacy"


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
    assert not related_rows("vllm_mixtral_gptq_moe_wna16_x6")
    assert [relation for _, relation, _ in related_rows("sglang_qwen2.5_7b_gptq_marlin_tp2_x6")] == ["not_reached"]


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


def test_one_source_against_two_library_versions_stays_apart():
    # Same probe source digest and stack directory, different installed FlashInfer: two campaigns, each paired on its own.
    def fi(tag, run_id, version, first):
        r = new_report(tag, run_id, first=first)
        r["env"]["packages"] = {"flashinfer-python": version, "torch": "2.14.0+cu130"}
        return r
    reports = {"r1/x.a.json": fi("a", "r1", "0.6.18.post1", "h1"), "r2/x.b.json": fi("b", "r2", "0.6.18.post1", "h2"),
               "r3/x.a.json": fi("a", "r3", "0.7.0", "h3"), "r4/x.b.json": fi("b", "r4", "0.7.0", "h3")}
    groups = campaigns(reports)
    assert list(groups) == ["c71ba85c0851, flashinfer-python 0.6.18.post1", "c71ba85c0851, flashinfer-python 0.7.0"]
    assert [summarise(g)[1] for g in groups.values()] == ["DIFFERS", "identical"]
    # A single library version keeps the plain digest label.
    assert list(campaigns({k: v for k, v in reports.items() if k.startswith(("r3", "r4"))})) == ["c71ba85c0851"]
