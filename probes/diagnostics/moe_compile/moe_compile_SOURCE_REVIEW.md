# vLLM issue 56900: v0.28.0 source and wheel audit

Audit date: 2026-09-16. This is a source review and diagnostic design, not a GPU result. No root cause is asserted. The installation recipes below have not yet been executed on a GPU host in this follow-up.

## Primary material checked

- [Issue 56900 and its comments](https://github.com/vllm-project/vllm/issues/56900) present at audit time. The original failure is H100 80 GB, driver 580.126.09, vLLM 0.28.0, torch 2.13.0+cu130. The independent non-reproduction is H20, driver 535, official vLLM 0.28.0+cu129 / torch 2.13.0+cu129. The latter changes CUDA build, GPU, driver, and installed package provenance.
- vLLM tag `v0.28.0`, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`, especially [`vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/vllm.py), [`vllm/config/attention.py`](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/attention.py), [`vllm/config/kernel.py`](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/kernel.py), the [unquantized MoE oracle](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/fused_moe/oracle/unquantized.py), and [CUDA platform backend selection](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/platforms/cuda.py).
- [v0.28.0 GitHub release/API metadata](https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.28.0), [PyPI vLLM 0.28.0 metadata](https://pypi.org/pypi/vllm/0.28.0/json), `wheels.vllm.ai` metadata for [cu129](https://wheels.vllm.ai/0.28.0/cu129/vllm/metadata.json) and [cu130](https://wheels.vllm.ai/0.28.0/cu130/vllm/metadata.json), and the PyTorch package indices for [cu129](https://download.pytorch.org/whl/cu129/torch/) and [cu130](https://download.pytorch.org/whl/cu130/torch/). The exact torch 2.13.0 CPython 3.12 x86-64 metadata is available for [cu129](https://download-r2.pytorch.org/whl/cu129/torch-2.13.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl.metadata) and [cu130](https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl.metadata).

## What the v0.28.0 source establishes

### Model runner

`Qwen2MoeForCausalLM` is explicitly in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (`vllm/config/vllm.py`). With Triton available and no unsupported feature, this architecture defaults to Model Runner V2. `VLLM_USE_V2_MODEL_RUNNER=0` or `1` is the direct override and is read before worker construction. `VLLM_ENABLE_V1_MULTIPROCESSING=0` controls process placement for the V1 engine; it does not select V1 versus V2.

This makes V1/V2 a valid diagnostic axis, but it is not a one-kernel comparison: the runner changes input preparation, workspace/warmup, sampling, and other execution machinery.

### Unquantized MoE backend

For CUDA, the unquantized backend oracle begins with FlashInfer TRTLLM, FlashInfer CUTLASS, Triton, and Batched Triton. On device capability family 90, it moves both FlashInfer choices to the back because the source says they are slower on Hopper. Therefore, if supported, automatic selection uses `TritonExperts` for this unquantized model. The issue's log confirms `TRITON`, `MoEPrepareAndFinalizeNoDPEPModular`, and `TritonExperts`.

The supported forced values relevant here are `moe_backend="triton"`, `"batched_triton"`, `"flashinfer_trtllm"`, and `"flashinfer_cutlass"`. A forced backend that does not support the configuration raises at startup. `batched_triton` also requests the batched activation format, so it changes prepare/finalize and layout as well as the expert class.

The Triton fused-MoE tuning lookup includes the complete sanitized GPU device name in its config filename. The original H100 log says its exact E=60,N=1408,H100 file was absent and the default config was used. That does not establish the H20's config choice; its backend/config log is still needed.

### Attention backend and FA3

On CUDA/Hopper non-MLA models, automatic attention priority is FlashAttention, FlashInfer, Triton attention, then flex attention. The issue log confirms FlashAttention version 3. Supported structured controls are:

```python
attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3}
attention_config={"backend": "TRITON_ATTN"}
```

Changing attention backend tests whether the compiled symptom survives outside FA3. It does not isolate MoE alone, and coherent output under another backend would localize the interaction rather than prove an FA3 defect.

### Compilation and dumps

Compilation modes are exactly: 0 `NONE`, 1 `STOCK_TORCH_COMPILE`, 2 `DYNAMO_TRACE_ONCE`, and 3 `VLLM_COMPILE`. Modes 1 and 2 are useful follow-ups if they construct successfully: they separate vLLM's custom backend from all compilation, but are not equivalent to mode 3.

`VLLM_DEBUG_DUMP_PATH` overrides `CompilationConfig.debug_dump_path`. In mode 3 it enables depyf/transformed-code and FX/pattern dump paths. It should not be described as a complete capture of every generated binary/kernel. `VLLM_LOGGING_LEVEL=DEBUG` is supported. `VLLM_LOG_MODEL_INSPECTION=1` is also supported and can record the selected layer implementations.

`VLLM_DISABLE_COMPILE_CACHE=1` disables vLLM's on-disk layer. Source tests combine it with Torch Inductor's `fresh_cache()` because it does not alone prove a cold Inductor/Triton compile. For subprocess diagnostics, set unique, initially empty `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR` before importing torch/vLLM. With vLLM caching disabled, vLLM's compiler adapter returns before redirecting the latter two, so explicit paths matter.

## H20 versus H100: what can and cannot be inferred

Both reported cards take vLLM's capability-family-90 branches, including the Hopper MoE priority and Hopper attention priority. That only establishes common coarse dispatch. It does not hold constant:

- exact GPU device name and any device-name-keyed tuning config;
- SM count, memory subsystem, and other device properties visible to Triton/Inductor;
- driver (reported 535 versus 580.126.09), CUDA driver/runtime compatibility behavior, or loaded shared libraries;
- cu129 versus cu130 torch and vLLM binaries;
- the effective Triton version and imported module paths.

Thus the H20 result shows the failure is not universal across all v0.28.0/sm90 environments. It does not identify cu130 as the cause. The controlled build test is cu129 and cu130 on the same GPU UUID and loaded driver, with runner/backend/config and imported modules recorded. If cu130 cannot import or allocate on that driver, that is a compatibility outcome and the model-output comparison is unavailable; it must not be converted into a correctness conclusion.

## Triton and TokenSpeed provenance

Both official torch 2.13.0+cu129 and +cu130 metadata require `triton==3.7.1` on Linux. vLLM 0.28.0 requires `tokenspeed-mla==0.1.8`; its [PyPI metadata](https://pypi.org/pypi/tokenspeed-mla/0.1.8/json) requires `tokenspeed-triton>=3.7.10.post20260531`. The original environment lists both `triton==3.7.1` and `tokenspeed-triton==3.8.10.post20260906`.

Direct inspection of the CPython 3.12 x86-64 wheel central directories resolves an initially suspected ambiguity: `tokenspeed-triton` owns `tokenspeed_triton/`, while `triton==3.7.1` owns `triton/` (including `triton/__init__.py`). They do not overlap the top-level `triton` namespace. The separate TokenSpeed package can still affect TokenSpeed MLA code paths, so pin and record it, but it is not evidence that `import triton` came from an ambiguous file owner.

Every result must therefore record:

- `triton.__version__`, `triton.__file__`, and the installed `triton` version;
- versions and dist-info locations for `triton`, `tokenspeed-triton`, and `tokenspeed-mla`;
- torch/vLLM module paths and versions, `torch.version.cuda`, `torch._C._show_config()`, and an installed-distribution inventory;
- NVIDIA driver, GPU UUID/name/capability/SM count/total memory, and preferably relevant loaded `.so` paths after generation.

Pin `triton==3.7.1`, `tokenspeed-triton==3.8.10.post20260906`, and `tokenspeed-mla==0.1.8` in both environments to match the original recorded stack. A future unconstrained `tokenspeed-triton>=...` resolution would add a package-version difference to the build comparison.

## Official 0.28.0 wheel identity and isolated installs

The release tag resolves to commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

- cu129 x86-64 release asset: `https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl`, SHA256 `8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397`.
- cu130/default x86-64 release asset: `https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl`; PyPI wheel: `https://files.pythonhosted.org/packages/87/d7/97f6ecc2ae883e601e08d7cef87cb54ececeefcfe6b5e12d5d92f8d06d6b/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl`. Both report SHA256 `addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889`. `vllm.envs.VLLM_MAIN_CUDA_VERSION` defaults to `13.0` at this tag, explaining the unsuffixed default wheel.
- `wheels.vllm.ai/0.28.0/{cu129,cu130}/vllm/metadata.json` maps both artifacts to the same commit. The cu130 metadata points to the unsuffixed wheel.

Example Python 3.12 x86-64 Linux installation, in separate environments:

```bash
python3.12 -m venv /opt/vllm-issue56900/cu129
/opt/vllm-issue56900/cu129/bin/python -m pip install --upgrade pip
/opt/vllm-issue56900/cu129/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu129 \
  'torch==2.13.0+cu129' 'torchvision==0.28.0+cu129' 'torchaudio==2.11.0+cu129'
/opt/vllm-issue56900/cu129/bin/python -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397' \
  --extra-index-url https://download.pytorch.org/whl/cu129
/opt/vllm-issue56900/cu129/bin/python -m pip install \
  'triton==3.7.1' 'tokenspeed-triton==3.8.10.post20260906' 'tokenspeed-mla==0.1.8'

python3.12 -m venv /opt/vllm-issue56900/cu130
/opt/vllm-issue56900/cu130/bin/python -m pip install --upgrade pip
/opt/vllm-issue56900/cu130/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' 'torchaudio==2.11.0+cu130'
/opt/vllm-issue56900/cu130/bin/python -m pip install \
  'https://files.pythonhosted.org/packages/87/d7/97f6ecc2ae883e601e08d7cef87cb54ececeefcfe6b5e12d5d92f8d06d6b/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889' \
  --extra-index-url https://download.pytorch.org/whl/cu130
/opt/vllm-issue56900/cu130/bin/python -m pip install \
  'triton==3.7.1' 'tokenspeed-triton==3.8.10.post20260906' 'tokenspeed-mla==0.1.8'
```

Installing torch first makes the intended local-version build explicit; the direct vLLM URLs prevent index selection from substituting another vLLM artifact. Still run `pip check` and capture actual imports. vLLM's own tag documentation contains stale wording that says binaries are CUDA 12.9 by default, while the same tag's source default and release page say the v0.28.0 PyPI/default artifact is CUDA 13.0. Artifact version/digest and runtime inspection are stronger evidence than that sentence.

The PyTorch cu129/cu130 indices were also checked for the exact CPython 3.12 x86-64 companion assets used above: `torchvision==0.28.0+cu129`, `torchvision==0.28.0+cu130`, `torchaudio==2.11.0+cu129`, and `torchaudio==2.11.0+cu130` all exist.

Minimum post-install proof (run separately in both environments):

```bash
python -m pip check
python - <<'PY'
import importlib.metadata as md, torch, triton, vllm
print("torch", torch.__version__, torch.version.cuda, torch.__file__)
print("triton import", triton.__version__, triton.__file__)
for name in ("triton", "tokenspeed-triton", "tokenspeed-mla", "vllm"):
    try:
        d = md.distribution(name)
        print(name, d.version, d._path)
    except md.PackageNotFoundError:
        print(name, "ABSENT")
print("vllm", vllm.__version__, vllm.__file__)
print("cuda allocation", torch.zeros(1, device="cuda"))
PY
```

## Testable diagnostic order

1. Same GPU UUID/driver, V2, graphs off: mode 3 versus mode 0 in cu129, then the same two cells in cu130. Use fresh per-cell caches and exact prompt token IDs. This establishes whether each build reproduces on the controlled host.
2. Preserve per-token top-5 logprobs. At the first divergent token, compare top-1 ids and margin to second place; exact token mismatch alone can be legitimate near a tie, while the reported broad degeneracy remains a separate signal.
3. If only compiled mode fails, run modes 1 and 2 versus 0. This tests custom vLLM compile versus other compile paths; construction failures are results.
4. On the reproducing build, force V1 then V2 with graphs off. Treat this as localization across the runner stack, not proof of a runner bug.
5. On the reproducing V2 build, hold mode 3/graphs off and force `moe_backend="triton"` (expected to confirm the logged auto choice), then supported alternate MoE backends. Separately force FA3 and Triton attention. Do not combine the backend changes in one arm.
6. Add graph-enabled cells only after graphs-off behavior is established. The posted issue already reports token identity between graph-on and graph-off compiled arms, so this is confirmation rather than the first discriminator.
7. A Hugging Face reference or a known accuracy set can establish external correctness; vLLM eager coherence alone is a useful internal reference but not an accuracy oracle.
