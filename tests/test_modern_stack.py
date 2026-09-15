"""The hook and harness on vLLM 0.29.0 as well as 0.28.0 (CPU only, vLLM stubbed).

0.29.0 keeps every call site and field the hook reads; what changes is the default model runner,
an opt-in V2 sampling path the hook cannot see, and the FP8 kernel priority list. These tests cover
the three additions that answer those: the refusal of batch-sharded sampling, the recorded
``linear_quant``, and the per-tensor forcing that now fails closed instead of running the per-token
default unnoticed. They also cover the cache and artefact records the cold E6 replays lacked.
"""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes/shape"))
sys.path.insert(0, str(ROOT / "probes/shape/shape_hook_pkg"))
import shape_common
import shape_hook


# ---------------------------------------------------------------------------
# Stub runners with the surface the hook reads, on each release's layout
# ---------------------------------------------------------------------------

def quant_method(name, activation_quant_key=None, kernel=None):
    cls = type(name, (), {"apply": lambda self, *a, **kw: None})
    qm = cls()
    qm.activation_quant_key = activation_quant_key
    if kernel is not None:
        qm.fp8_linear = type(kernel, (), {})()
    return qm


class Model:
    def __init__(self, modules):
        self._modules_by_name = modules

    def named_modules(self):
        return list(self._modules_by_name.items())


def runner(*, v2=False, batch_sharded=False, model=None):
    """A stand-in for the runner object the hook reads, with only the attributes it touches."""
    parallel = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                  enable_expert_parallel=False, enable_batch_sharded_sampling=batch_sharded)
    cfg = NS(compilation_config=NS(mode=3, cudagraph_mode=NS(name="FULL_AND_PIECEWISE")),
             parallel_config=parallel,
             scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=2048, enable_chunked_prefill=True, async_scheduling=False))
    r = NS(vllm_config=cfg, input_batch=NS(req_ids=["0-a"], num_reqs=1, num_computed_tokens_cpu=[0], num_prompt_tokens=[3]),
           requests={"0-a": NS(get_token_id=lambda p: [10, 11, 12][p])}, attn_groups=[[NS(backend=NS(get_name=lambda: "FLASH_ATTN"))]],
           model=model or Model({}))
    r.get_model = lambda: r.model
    if v2:
        r.req_states = NS(req_id_to_index={}, all_token_ids=NS(gpu=None), num_computed_tokens_np=[0])
    return r


def scheduler_output():
    return NS(num_scheduled_tokens={"0-a": 3}, total_num_scheduled_tokens=3, scheduled_new_reqs=[],
              scheduled_cached_reqs=None, finished_req_ids=set(), preempted_req_ids=set())


@pytest.fixture(autouse=True)
def clean_hook_state():
    saved = dict(shape_hook._state)
    shape_hook._state.update({"linear_quant": None, "fp8_forcing": None, "hashes": None, "batch": None,
                              "dispatch": None, "moe_counts": None, "seen": set(), "prompts": {}})
    yield
    shape_hook._state.clear()
    shape_hook._state.update(saved)


# ---------------------------------------------------------------------------
# 0.29.0 batch-sharded sampling: refuse rather than record without hashes
# ---------------------------------------------------------------------------

def test_batch_sharded_sampling_is_refused_because_the_hook_does_not_see_those_logits():
    with pytest.raises(ValueError, match="batch-sharded sampling"):
        shape_hook._record(runner(v2=True, batch_sharded=True), scheduler_output())


def test_the_default_v2_configuration_is_still_recorded():
    rec = shape_hook._record(runner(v2=True), scheduler_output())
    assert rec["model_runner"] == "v2" and rec["schema"] == shape_hook.SCHEMA


def test_a_runner_without_the_0290_attribute_is_recorded_as_before():
    """0.28.0 has no enable_batch_sharded_sampling; the check must not reject its records."""
    r = runner()
    del r.vllm_config.parallel_config.enable_batch_sharded_sampling
    assert shape_hook._record(r, scheduler_output())["model_runner"] == "v1"


# ---------------------------------------------------------------------------
# linear_quant: the resolved FP8 scheme, recorded rather than inferred
# ---------------------------------------------------------------------------

def test_linear_quant_reports_the_first_quantised_layer_not_the_first_layer():
    model = Model({"model.embed": NS(quant_method=None),
                   "model.layers.0.qkv_proj": NS(quant_method=quant_method("UnquantizedLinearMethod")),
                   "model.layers.1.qkv_proj": NS(quant_method=quant_method(
                       "Fp8PerTensorOnlineLinearMethod", "kFp8DynamicTensorSym", "CutlassFP8ScaledMMLinearKernel"))})
    assert shape_hook.linear_quant(model) == {"layer": "model.layers.1.qkv_proj", "method": "Fp8PerTensorOnlineLinearMethod",
                                              "activation_quant_key": "kFp8DynamicTensorSym", "kernel": "CutlassFP8ScaledMMLinearKernel"}


def test_linear_quant_distinguishes_per_tensor_from_per_token():
    """The whole point of the field: the two X2 arms differ only in the activation key."""
    per_token = Model({"l": NS(quant_method=quant_method("Fp8PerTensorOnlineLinearMethod", "kFp8DynamicTokenSym", "CutlassFP8ScaledMMLinearKernel"))})
    per_tensor = Model({"l": NS(quant_method=quant_method("Fp8PerTensorOnlineLinearMethod", "kFp8DynamicTensorSym", "CutlassFP8ScaledMMLinearKernel"))})
    assert shape_hook.linear_quant(per_token)["activation_quant_key"] != shape_hook.linear_quant(per_tensor)["activation_quant_key"]


def test_linear_quant_falls_back_to_the_first_unquantised_linear_method():
    model = Model({"l": NS(quant_method=quant_method("UnquantizedLinearMethod"))})
    assert shape_hook.linear_quant(model) == {"layer": "l", "method": "UnquantizedLinearMethod", "activation_quant_key": None, "kernel": None}


def test_linear_quant_is_none_when_no_linear_layer_exists_and_is_recorded_once():
    r = runner()
    rec = shape_hook._record(r, scheduler_output())
    assert rec["linear_quant"] is None and "linear_quant" in rec
    # A later pass reuses the cached value rather than walking the model again.
    r.get_model = lambda: (_ for _ in ()).throw(AssertionError("walked twice"))
    shape_hook._record(r, scheduler_output())


def test_a_failing_model_walk_is_recorded_as_an_error_not_silently_dropped():
    r = runner()
    r.get_model = lambda: (_ for _ in ()).throw(RuntimeError("no model"))
    del r.model
    assert "error" in shape_hook._record(r, scheduler_output())["linear_quant"]


# ---------------------------------------------------------------------------
# Per-tensor FP8 forcing: reach modules that already imported the name, or fail
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_vllm_fp8(monkeypatch):
    """The three modules of 0.28.0 and 0.29.0 that hold cutlass_fp8_supported, the online one preloaded."""
    made = {}
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.quantization",
                 "vllm.model_executor.layers.quantization.utils", "vllm.model_executor.layers.quantization.online"):
        made[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, made[name])
    for name in ("vllm.model_executor.layers.quantization.utils.w8a8_utils",
                 "vllm.model_executor.layers.quantization.fp8",
                 "vllm.model_executor.layers.quantization.online.fp8"):
        mod = types.ModuleType(name)
        mod.cutlass_fp8_supported = lambda: True  # every module took its own reference, as both releases do
        made[name] = mod
        monkeypatch.setitem(sys.modules, name, mod)
    return made


def test_forcing_rebinds_the_name_in_the_online_module_that_already_imported_it(stub_vllm_fp8):
    online = stub_vllm_fp8["vllm.model_executor.layers.quantization.online.fp8"]
    assert online.cutlass_fp8_supported() is True
    shape_hook._force_fp8_per_tensor()
    assert online.cutlass_fp8_supported() is False
    for name in ("vllm.model_executor.layers.quantization.utils.w8a8_utils", "vllm.model_executor.layers.quantization.fp8"):
        assert stub_vllm_fp8[name].cutlass_fp8_supported() is False


def test_forcing_records_which_modules_it_patched(stub_vllm_fp8):
    shape_hook._force_fp8_per_tensor()
    state = shape_hook._state["fp8_forcing"]
    assert state["patched_modules"] == sorted(["vllm.model_executor.layers.quantization.utils.w8a8_utils",
                                               "vllm.model_executor.layers.quantization.fp8",
                                               "vllm.model_executor.layers.quantization.online.fp8"])
    assert "vllm.model_executor.layers.quantization.online.fp8" in state["already_imported"]


def test_forcing_fails_closed_when_no_module_exposes_the_check(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("vllm"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    with pytest.raises(RuntimeError, match="per-token default"):
        shape_hook._force_fp8_per_tensor()


def test_the_forcing_record_reaches_the_pass_record(stub_vllm_fp8):
    shape_hook._force_fp8_per_tensor()
    rec = shape_hook._record(runner(), scheduler_output())
    assert rec["fp8_forcing"]["patched_modules"]


def test_an_unforced_arm_records_null_rather_than_omitting_the_field():
    assert shape_hook._record(runner(), scheduler_output())["fp8_forcing"] is None


# ---------------------------------------------------------------------------
# Cache and compiled-artefact records (the cold E6 replays had none)
# ---------------------------------------------------------------------------

def test_cache_roots_cover_every_compiled_artefact_store_and_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllmcache"))
    roots = shape_common.cache_roots()
    assert roots["vllm"] == str(tmp_path / "vllmcache")
    assert set(roots) >= {"vllm", "inductor", "triton", "flashinfer", "torch_extensions", "cuda_jit", "deep_gemm"}


def test_a_redirected_inductor_cache_inside_the_vllm_root_is_not_counted_twice(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllmcache"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "vllmcache" / "inductor"))
    assert "inductor_env" not in shape_common.cache_roots()
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert shape_common.cache_roots()["inductor_env"] == str(tmp_path / "elsewhere")


def test_cache_manifest_distinguishes_absent_empty_and_populated(tmp_path):
    missing = shape_common.cache_manifest(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    empty = shape_common.cache_manifest(tmp_path / "empty")
    assert missing["exists"] is False and empty["exists"] is True
    assert missing["files"] == empty["files"] == 0 and missing["sha256"] is empty["sha256"] is None
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "a.py").write_text("x")
    full = shape_common.cache_manifest(tmp_path / "full")
    assert full["files"] == 1 and full["bytes"] == 1 and full["sha256"]


def test_the_manifest_digest_changes_with_content_name_and_size(tmp_path):
    def digest(name, text):
        d = tmp_path / name
        d.mkdir()
        (d / "k.py").write_text(text)
        return shape_common.cache_manifest(d)["sha256"]
    assert digest("a", "x") != digest("b", "y")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "other.py").write_text("x")
    assert shape_common.cache_manifest(tmp_path / "c")["sha256"] != digest("d", "x")


def test_a_huge_cache_is_truncated_rather_than_walked_to_the_end(tmp_path):
    for i in range(5):
        (tmp_path / f"{i}.bin").write_text("x")
    m = shape_common.cache_manifest(tmp_path, max_files=2)
    assert m["truncated"] == 2 and m["files"] == 2 and m["sha256"] is None


def test_cache_is_empty_is_the_cold_replay_precondition(tmp_path):
    (tmp_path / "a").mkdir()
    assert shape_common.cache_is_empty({"a": shape_common.cache_manifest(tmp_path / "a")})
    (tmp_path / "a" / "k").write_text("x")
    assert not shape_common.cache_is_empty({"a": shape_common.cache_manifest(tmp_path / "a")})


def test_compiled_artefacts_report_the_autotune_choices_and_the_kernels_that_ran(tmp_path):
    root = tmp_path / "inductor"
    (root / "ab").mkdir(parents=True)
    (root / "ab" / "cxyz.best_config").write_text(json.dumps({"XBLOCK": 8, "RBLOCK": 64, "num_warps": 4}))
    (root / "ab" / "cxyz.py").write_text(
        "def triton_red_fused_fused_add_rms_norm_3(x):\n    pass\n"
        "def triton_poi_fused_mul_1(x):\n    pass\n"
        "    extern_kernels.mm(a, b, out=c)\n    extern_kernels.bmm(a, b, out=c)\n")
    art = shape_common.compiled_artefacts({"inductor": str(root)})
    assert art["best_configs"]["inductor/ab/cxyz.best_config"]["RBLOCK"] == 64
    assert art["extern_calls"] == {"bmm": 1, "mm": 1}
    # The numeric suffix is dropped so two compiles of the same graph compare by kernel family.
    assert art["triton_kernel_families"] == {"triton_poi_fused_mul": 1, "triton_red_fused_fused_add_rms_norm": 1}


def test_an_unreadable_autotune_choice_is_recorded_as_an_error_not_dropped(tmp_path):
    (tmp_path / "x.best_config").write_text("{not json")
    art = shape_common.compiled_artefacts({"inductor": str(tmp_path)})
    assert "error" in art["best_configs"]["inductor/x.best_config"]


def test_write_cache_state_and_write_artefacts_leave_the_files_an_arm_is_read_from(tmp_path, monkeypatch):
    monkeypatch.setattr(shape_common, "cache_roots", lambda: {"inductor": str(tmp_path / "cache")})
    out = tmp_path / "arm"
    out.mkdir()
    shape_common.write_cache_state(out, "before_engine")
    before = json.loads((out / "cache_before_engine.json").read_text())
    assert before["phase"] == "before_engine" and before["empty"] is True
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "k.py").write_text("extern_kernels.mm(a, b)\n")
    shape_common.write_artefacts(out)
    assert json.loads((out / "cache_after_run.json").read_text())["empty"] is False
    assert json.loads((out / "artefacts.json").read_text())["extern_calls"] == {"mm": 1}


def test_machine_identity_records_the_host_even_when_nvidia_smi_is_absent():
    ident = shape_common.machine_identity()
    assert ident["hostname"] and "kernel" in ident
    assert ident["gpus"] is None or isinstance(ident["gpus"], list)


# ---------------------------------------------------------------------------
# Harness options that keep two compile arms apart
# ---------------------------------------------------------------------------

def base_args(**kw):
    fields = {"model": "Qwen/Qwen2.5-7B-Instruct", "revision": None, "tp": 1, "quantization": None, "cudagraph": 1,
              "prefix_caching": 0, "max_tokens": 2, "repeats": 2, "out": Path("/tmp/out"), "fp8_per_tensor": False,
              "no_compile": False, "inductor_config": None, "custom_ops": None, "tag": None}
    return NS(**{**fields, **kw})


def test_inductor_config_and_custom_ops_reach_the_compilation_config():
    kw = shape_common.engine_kwargs(base_args(inductor_config='{"max_autotune": false}', custom_ops="+rms_norm, -quant_fp8"))
    cc = kw["compilation_config"]
    assert cc["inductor_compile_config"] == {"max_autotune": False}
    assert cc["custom_ops"] == ["+rms_norm", "-quant_fp8"]


def test_absent_options_leave_vllms_defaults_alone():
    assert set(shape_common.engine_kwargs(base_args())["compilation_config"]) == {"mode", "cudagraph_mode"}


def test_a_tag_keeps_arms_that_differ_only_in_those_options_apart():
    plain, tagged = base_args(), base_args(tag="autotune_off")
    assert shape_common.arm_name(plain) != shape_common.arm_name(tagged)
    assert shape_common.arm_name(tagged).endswith("_autotune_off")


def test_an_untagged_arm_keeps_the_name_the_committed_records_use():
    assert shape_common.arm_name(base_args()) == "Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0"
