"""CPU-only tests for the issue #56900 offline audit."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "probes/diagnostics/moe_compile/moe_compile_audit.py"
SPEC = importlib.util.spec_from_file_location("moe_compile_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


UUID = "fcd67bc9-4348-93cc-aba6-22a2076f2fdc"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def make_result(token_offset: int = 0) -> dict:
    prompts = [f"prompt {index}" for index in range(8)] * 2
    prompt_ids = [[index, index + 100] for index in range(8)] * 2
    repeats = []
    for _ in range(2):
        outputs = []
        for index, prompt in enumerate(prompts):
            token_ids = [token_offset + index % 8 * 100 + position for position in range(32)]
            logprobs = []
            for token_id in token_ids:
                logprobs.append(
                    {
                        "candidates": [
                            {"token_id": token_id, "logprob": -0.1, "rank": 1, "decoded_token": "a"},
                            {"token_id": token_id + 1, "logprob": -1.1, "rank": 2, "decoded_token": "b"},
                        ],
                        "top1_margin": 1.0,
                    }
                )
            outputs.append(
                {
                    "index": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "token_ids": token_ids,
                    "token_ids_sha256": sha_json(token_ids),
                    "text": " ".join(map(str, token_ids)),
                    "cycle_period_last_16": None,
                    "logprobs": logprobs,
                }
            )
        repeats.append(outputs)
    return {
        "model": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "revision": "ec052fda178e241c7c443468d2fa1db6618996be",
        "expected_vllm": "0.28.0",
        "prompts": prompts,
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": [sha_json(ids) for ids in prompt_ids],
        "sampling": {"temperature": 0.0, "max_tokens": 32, "logprobs": 5, "ignore_eos": True},
        "repeats": repeats,
        "metrics": {
            "repeat_token_identical": True,
            "duplicate_pairs_agree": 8,
            "duplicate_pairs": [True] * 8,
            "short_cycles": 0,
        },
    }


def make_root(
    base: Path,
    label: str,
    cuda_build: str,
    *,
    token_offset: int = 0,
    omit_backend: bool = False,
    failed: bool = False,
    runner: str = "v2",
) -> MODULE.InputRoot:
    root = base / label
    root.mkdir()
    requested = {"compile": "on", "graphs": "off", "runner": runner}
    name = MODULE.cell_name(requested)
    run = root / name
    run.mkdir()
    write_json(root / "manifest.json", {"requested": [requested]})
    write_json(
        root / "status.json",
        {
            "completed": [
                {
                    "name": name,
                    **requested,
                    "returncode": 1 if failed else 0,
                    "timed_out": False,
                }
            ]
        },
    )
    if failed:
        write_json(run / "failure.json", {"type": "RuntimeError", "message": "synthetic failure"})
        return MODULE.InputRoot(label, root)

    cache = run / "cache"
    cache.mkdir()
    configured = {
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
    }
    write_json(
        run / "environment.json",
        {
            "set_by_reproducer": configured,
            "inherited_relevant": {"CUDA_VISIBLE_DEVICES": "0"},
        },
    )
    write_json(run / "cache_before.json", {"root": str(cache), "file_count": 0, "files": []})
    write_json(run / "cache_after.json", {"root": str(cache), "file_count": 4, "files": ["one"]})
    freeze = (
        f"torch==2.13.0+cu{cuda_build}\n"
        "triton==3.7.1\n"
        "tokenspeed-mla==0.1.8\n"
        "tokenspeed-triton==3.8.10.post20260906\n"
        f"vllm==0.28.0{'+cu129' if cuda_build == '129' else ''}\n"
    )
    package = lambda version: {  # noqa: E731 - compact synthetic provenance
        "distribution_version": version,
        "module_version": version,
        "module_file": "/venv/site-packages/module.py",
        "module_file_sha256": "a" * 64,
        "record_sha256": "b" * 64,
        "direct_url": None,
    }
    write_json(
        run / "provenance.json",
        {
            "torch": {
                **package(f"2.13.0+cu{cuda_build}"),
                "cuda_build": f"12.{cuda_build[-1]}" if cuda_build == "129" else "13.0",
                "cuda_device": {
                    "uuid": f"UUID('{UUID}')",
                    "name": "NVIDIA H100 80GB HBM3",
                    "major": 9,
                    "minor": 0,
                    "multi_processor_count": 120,
                    "total_memory": 80_000_000_000,
                },
            },
            "triton": package("3.7.1"),
            "tokenspeed-triton": package("3.8.10.post20260906"),
            "vllm": package(f"0.28.0{'+cu129' if cuda_build == '129' else ''}"),
            "nvidia_smi": {
                "returncode": 0,
                "stdout": f"0, GPU-{UUID}, NVIDIA H100 80GB HBM3, 580.126.09\n",
                "stderr": "",
            },
            "pip_freeze": {"returncode": 0, "stdout": freeze, "stderr": ""},
        },
    )
    write_json(
        run / "resolved_config.json",
        {"requested": requested, "use_v2_model_runner": runner == "v2"},
    )
    write_json(
        run / "resolved_assertions.json",
        {"runner_matches_request": True, "mode_matches_request": True, "graph_mode_matches_request": True},
    )
    write_json(run / "result.json", make_result(token_offset))
    lines = [
        "INFO 01:02:03 [cuda.py:486] Using FLASH_ATTN attention backend out of potential backends: []\n",
        "INFO 01:02:03 [flash_attn.py:866] Using FlashAttention version 3\n",
        "INFO 01:02:03 [unquantized.py:319] Using TRITON Unquantized MoE backend out of potential backends: []\n",
        "INFO 01:02:04 [unquantized.py:400] Using MoEPrepareAndFinalizeNoDPEPModular\n",
        "INFO 01:02:04 [unquantized.py:401] Using TritonExperts MoE backend\n",
    ]
    if omit_backend:
        lines.pop()
    (root / f"{name}.stdout.log").write_text("".join(lines))
    (root / f"{name}.stderr.log").write_text("")
    return MODULE.InputRoot(label, root)


class MoeCompileAuditTests(unittest.TestCase):
    def test_normalises_backend_lines_and_bare_torch_uuid(self) -> None:
        evidence = MODULE.backend_evidence(
            "Using FLASH_ATTN attention backend\n"
            "Using FlashAttention version 3\n"
            "Using TRITON Unquantized MoE backend\n"
            "Using MoEPrepareAndFinalizeNoDPEPModular\n"
            "Using TritonExperts MoE backend\n",
            "",
            {"requested": {"runner": "v2"}, "use_v2_model_runner": True},
        )
        self.assertTrue(evidence["complete"])
        self.assertEqual(evidence["signature"]["moe_experts"], "TritonExperts")
        self.assertEqual(MODULE.canonical_gpu_uuid(f"UUID('{UUID}')"), f"GPU-{UUID}")

    def test_audits_controlled_cuda_build_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [
                make_root(base, "cu129", "129"),
                make_root(base, "cu130", "130", token_offset=1),
            ]
            report = MODULE.audit(roots)
            self.assertEqual(report["evidence_state"], "CONTROLLED")
            self.assertEqual(len(report["comparisons"]), 1)
            comparison = report["comparisons"][0]
            self.assertEqual(comparison["kind"], "cuda_build")
            self.assertTrue(comparison["controlled"])
            self.assertEqual(comparison["tokens_identical_prompts"], 0)
            self.assertEqual(comparison["prompt_count"], 16)
            self.assertTrue(comparison["prompt_token_ids_identical"])
            self.assertEqual(len(comparison["distribution_differences"]), 2)
            self.assertEqual(report["global_checks"]["selected_gpu_uuids"], [f"GPU-{UUID}"])
            rendered = MODULE.markdown(report)
            self.assertIn("first-token margin range r0", rendered)
            self.assertIn("1..1", rendered)

    def test_missing_backend_is_confounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_root(Path(temporary), "cu130", "130", omit_backend=True)
            report = MODULE.audit([root])
            self.assertEqual(report["evidence_state"], "CONFOUNDED")
            self.assertEqual(report["unusable_successful_cells"], [report["cells"][0]["id"]])

    def test_separate_roots_with_one_stack_compare_runners_or_fresh_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = make_root(base, "cu130_v2_a", "130")
            second = make_root(base, "cu130_v2_b", "130")
            v1 = make_root(base, "cu130_v1", "130", runner="v1")
            report = MODULE.audit([first, second, v1])
            kinds = sorted(item["kind"] for item in report["comparisons"])
            self.assertEqual(
                kinds,
                ["fresh_process_repeat", "v1_vs_v2", "v1_vs_v2"],
            )

    def test_historical_comparison_is_bounded_to_retained_target_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_root(base, "cu130", "130")
            old_root = base / "e4"
            arm = old_root / "Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_compile_v2_graphs0_prefix0"
            arm.mkdir(parents=True)
            current = make_result()["repeats"][0][0]
            prompts = make_result()["prompts"]
            write_json(
                arm / "run.json",
                {
                    "input_sha256": sha_json(prompts),
                    "model_revision": "ec052fda178e241c7c443468d2fa1db6618996be",
                    "resolved_compile": 3,
                    "resolved_cudagraph": "NONE",
                },
            )
            write_json(
                arm / "runs.json",
                [
                    {
                        "kind": "original",
                        "repeat": 0,
                        "target_rid": "0",
                        "target": {
                            "tokens": current["token_ids"],
                            "logprobs": [
                                [
                                    [candidate["token_id"], float(candidate["logprob"]).hex()]
                                    for candidate in position["candidates"]
                                ]
                                for position in current["logprobs"]
                            ],
                        },
                    }
                ],
            )
            report = MODULE.audit([root], old_root)
            historical = report["historical_e4"]
            self.assertEqual(historical["errors"], [])
            self.assertEqual(len(historical["comparisons"]), 1)
            comparison = historical["comparisons"][0]
            self.assertTrue(comparison["generated_tokens"]["exact"])
            self.assertTrue(comparison["top5_logprobs"]["all_exact"])
            self.assertIn("target 0 only", historical["scope"])

    def test_failed_cell_is_incomplete_not_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_root(Path(temporary), "cu130", "130", failed=True)
            report = MODULE.audit([root])
            self.assertEqual(report["evidence_state"], "INCOMPLETE")
            self.assertEqual(report["invalid_cells"], [])
            self.assertEqual(report["failed_cells"], [report["cells"][0]["id"]])


if __name__ == "__main__":
    unittest.main()
