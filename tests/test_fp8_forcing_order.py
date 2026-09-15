"""--fp8-per-tensor must reach the shape hook before it registers (CPU only, vLLM stubbed).

The hook reads SHAPE_FORCE_FP8_PER_TENSOR once, inside register(); a second register() (the
vllm.general_plugins load at engine start) returns at once. At TP=1 with the in-process engine
core the model runs in the runner's own process, so a variable set after the runner's register()
call is never seen and the arm silently runs vLLM's per-token default.
"""
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes/shape"))
sys.path.insert(0, str(ROOT / "probes/shape/shape_hook_pkg"))
import run_e2
import run_e3
import run_e4
import run_e6
import shape_hook

from test_e6_replay import RUN_META, make_record, outputs, write_arm


class EngineStart(Exception):
    """Raised where the stubbed engine would start; everything before it has run."""


@pytest.fixture
def seen(monkeypatch):
    """Stub the hook's register() and vLLM; record the variable at registration and at engine start."""
    events = []
    monkeypatch.setenv("SHAPE_FORCE_FP8_PER_TENSOR", "unset")  # records the original state so teardown restores it
    monkeypatch.delenv("SHAPE_FORCE_FP8_PER_TENSOR")
    monkeypatch.setattr(shape_hook, "register", lambda: events.append(("register", os.environ.get("SHAPE_FORCE_FP8_PER_TENSOR"))))

    def llm(**kw):
        events.append(("engine", os.environ.get("SHAPE_FORCE_FP8_PER_TENSOR")))
        raise EngineStart

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.TokensPrompt = llm, object, object
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for mod in (run_e2, run_e3, run_e4, run_e6):
        if hasattr(mod, "env_with_hook"):
            monkeypatch.setattr(mod, "env_with_hook", lambda out: Path(out).mkdir(parents=True))
    return events


def args(tmp_path, per_tensor, **kw):
    return NS(model="Qwen/Qwen2.5-7B-Instruct", revision=None, tp=1, quantization="fp8", cudagraph=1, prefix_caching=0,
              max_tokens=2, repeats=2, out=tmp_path / "out", fp8_per_tensor=per_tensor, no_compile=False, mixed=False, **kw)


def expected(per_tensor):
    return "1" if per_tensor else None


@pytest.mark.parametrize("per_tensor", [True, False])
def test_e6_record_sets_the_variable_before_register(tmp_path, seen, per_tensor):
    with pytest.raises(EngineStart):
        run_e6.record(args(tmp_path, per_tensor, record=True))
    assert seen[0] == ("register", expected(per_tensor))


@pytest.mark.parametrize("per_tensor", [True, False])
def test_e6_replay_sets_the_variable_before_register(tmp_path, seen, per_tensor):
    # The replay takes the flag from the recorded run.json, as a real replay does.
    meta = {**RUN_META, "args": {**RUN_META["args"], **{k: v for k, v in vars(args(tmp_path, per_tensor)).items() if k != "out"}}}
    write_arm(tmp_path / "arm/record", make_record("a"), outputs("a"), meta)
    with pytest.raises(EngineStart):
        run_e6.replay(tmp_path / "arm/record", tmp_path / "arm/replay", None)
    assert seen[0] == ("register", expected(per_tensor))


@pytest.mark.parametrize("per_tensor", [True, False])
def test_e3_run_sets_the_variable_before_register(tmp_path, seen, per_tensor):
    with pytest.raises(EngineStart):
        run_e3.run(args(tmp_path, per_tensor))
    assert seen[0] == ("register", expected(per_tensor))


@pytest.mark.parametrize("runner", [run_e2, run_e4])
@pytest.mark.parametrize("per_tensor", [True, False])
def test_e2_and_e4_set_the_variable_before_the_hook_or_engine_sees_it(tmp_path, seen, monkeypatch, runner, per_tensor):
    argv = [runner.__file__, "--model", "Qwen/Qwen2.5-7B-Instruct", "--quantization", "fp8", "--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + (["--fp8-per-tensor"] if per_tensor else []))
    with pytest.raises(EngineStart):
        runner.main()
    # run_e2 has no in-process register(); its engine core loads the plugin in a child that inherits the variable.
    assert seen[0][1] == expected(per_tensor)
