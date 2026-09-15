#!/usr/bin/env bash
# Install the shape_hook plugin into the venv, verify the vllm.general_plugins entry point and schema 3, record the stack,
# the GPU topology and the NCCL version.
set -x
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pip-cache PATH=/root/venv/bin:$PATH
pip install -q -e /root/census/probes/shape/shape_hook_pkg && echo HOOK_INSTALL_OK || echo HOOK_INSTALL_FAILED
python - <<'PY'
import importlib.metadata as md
eps = [ep for ep in md.entry_points(group="vllm.general_plugins")]
print("ENTRY_POINTS", [(e.name, e.value) for e in eps])
assert any(e.name == "shape_hook" and e.value == "shape_hook:register" for e in eps), "shape_hook entry point missing"
ep = [e for e in eps if e.name == "shape_hook"][0]
fn = ep.load(); import shape_hook, inspect
print("SHAPE_HOOK_IMPORT_OK", shape_hook.__file__, "register_resolves_to", fn.__module__ + "." + fn.__name__)
src = open(shape_hook.__file__).read()
print("SCHEMA", shape_hook.SCHEMA, "new_token_ids_occurrences", src.count("new_token_ids"))
assert shape_hook.SCHEMA == 3 and "new_token_ids" in src
import torch, vllm, platform
print("STACK python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
print("TORCH_NCCL_VERSION", ".".join(map(str, torch.cuda.nccl.version())) if isinstance(torch.cuda.nccl.version(), tuple) else torch.cuda.nccl.version())
for p in ("flashinfer-python", "triton", "transformers", "ninja", "nvidia-cublas-cu13", "nvidia-cublas", "nvidia-nccl-cu13", "hf_transfer"):
    try: print("PKG", p, md.version(p))
    except Exception as e: print("PKG", p, None)
print("DRIVER", open("/proc/driver/nvidia/version").read().split("\n")[0])
print("GPU_COUNT", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if torch.cuda.device_count() > 1:
    print("P2P 0->1", torch.cuda.can_device_access_peer(0, 1), "1->0", torch.cuda.can_device_access_peer(1, 0))
PY
cd /root/census/probes/shape && python -c 'import sys; sys.path.insert(0, "."); import teacher_forcing; print("TEACHER_FORCING_VLLM_INTERFACE", teacher_forcing.VLLM_INTERFACE)'
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink -s 2>&1 | head -40
nvidia-smi -q | grep -iE "Product Name|Bus Id|Link Width|PCIe Generation|Current|Max" | head -40
pip freeze > /root/logs/pip-freeze.txt; wc -l /root/logs/pip-freeze.txt
# ---- additions for the cold second-machine replay: machine identity, JIT tool chain, cache locations, stack diff
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,serial,pci.bus_id,driver_version,vbios_version --format=csv
python -c 'import torch; print("TORCH_CUDA_NCCL", torch.cuda.nccl.version())'
hostname; grep -m1 "model name" /proc/cpuinfo; nproc; free -g | head -2
env | grep -E "^RUNPOD_(POD_ID|POD_HOSTNAME|DC_ID|GPU_COUNT|CPU_COUNT|PUBLIC_IP|TCP_PORT_22)=" | sort
cat /etc/os-release | head -3; uname -r
ls -l /usr/local/cuda 2>&1; /usr/local/cuda/bin/nvcc --version 2>&1 | tail -2; which -a nvcc; nvcc --version 2>&1 | tail -2
python - <<'PY'
import importlib, importlib.metadata as md, os, inspect
for m in ("vllm.third_party.deep_gemm", "deep_gemm"):
    try:
        mod = importlib.import_module(m); print("DEEP_GEMM_MODULE", m, os.path.dirname(mod.__file__), "version", getattr(mod, "__version__", None))
    except Exception as e: print("DEEP_GEMM_MODULE", m, "not importable:", type(e).__name__, str(e)[:120])
for d in ("deep-gemm", "deep_gemm"):
    try: print("DEEP_GEMM_DIST", d, md.version(d))
    except Exception: print("DEEP_GEMM_DIST", d, None)
import vllm.envs as ve
print("VLLM_CACHE_ROOT", ve.VLLM_CACHE_ROOT)
try:
    import flashinfer.jit.env as fe
    print("FLASHINFER_JIT", {k: str(getattr(fe, k)) for k in dir(fe) if k.isupper() and ("DIR" in k or "BASE" in k)})
except Exception as e: print("FLASHINFER_JIT import failed", e)
import torch._inductor.config as ic, torch._inductor.runtime.cache_dir_utils as cu
print("INDUCTOR_CACHE_DIR", cu.cache_dir())
import triton.knobs as tk
try: print("TRITON_CACHE_DIR", tk.cache.dir)
except Exception as e: print("TRITON_CACHE_DIR?", e)
PY
grep -rn "DG_JIT_CACHE_DIR\|\.deep_gemm" /root/venv/lib/python3.12/site-packages/vllm/utils/deep_gemm.py /root/venv/lib/python3.12/site-packages/vllm/third_party/deep_gemm/*.py 2>/dev/null | head -8
grep -rln "DG_JIT_CACHE_DIR" /root/venv/lib/python3.12/site-packages/vllm 2>/dev/null | head -5
# installed distributions against every record's env.json
python - <<'PY'
import json, glob, importlib.metadata as md
inst = {d.metadata["Name"]: d.version for d in md.distributions()}
print("INSTALLED_DISTRIBUTIONS", len(inst))
for f in sorted(glob.glob("/root/census/probes/shape/results_cold/e6/*/record/env.json")):
    e = json.load(open(f)); rec = e["installed_distributions"]
    diff = {k: (rec.get(k), inst.get(k)) for k in sorted(set(rec) | set(inst)) if rec.get(k) != inst.get(k)}
    print("DIST_DIFF", f.split("/")[-3], "record_n", len(rec), "diff", diff)
    print("PKGS_DIFF", f.split("/")[-3], {k: (v, inst.get(k)) for k, v in e["packages"].items() if v != inst.get(k)}, "driver", e["driver"], "gpu_count", e["gpu_count"])
PY
echo POST_BOOTSTRAP_DONE
