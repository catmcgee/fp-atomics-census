#!/usr/bin/env bash
# Install the shape_hook plugin, verify the entry point and schema 3, and record the stack, machine identity and cache roots.
set -x
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pip-cache PATH=/root/venv/bin:$PATH
pip install -q -e /root/census/probes/shape/shape_hook_pkg && echo HOOK_INSTALL_OK || echo HOOK_INSTALL_FAILED
python - <<'PY'
import importlib.metadata as md
eps = [ep for ep in md.entry_points(group="vllm.general_plugins")]
print("ENTRY_POINTS", [(e.name, e.value) for e in eps])
assert any(e.name == "shape_hook" and e.value == "shape_hook:register" for e in eps), "shape_hook entry point missing"
fn = [e for e in eps if e.name == "shape_hook"][0].load()
import shape_hook, torch, vllm, platform
print("SHAPE_HOOK_IMPORT_OK", shape_hook.__file__, "register_resolves_to", fn.__module__ + "." + fn.__name__)
src = open(shape_hook.__file__).read()
print("SCHEMA", shape_hook.SCHEMA, "new_token_ids_occurrences", src.count("new_token_ids"))
assert shape_hook.SCHEMA == 3 and "new_token_ids" in src
print("STACK python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
for p in ("flashinfer-python", "triton", "transformers", "ninja"):
    try: print("PKG", p, md.version(p))
    except Exception: print("PKG", p, None)
print("DRIVER", open("/proc/driver/nvidia/version").read().split("\n")[0])
print("GPU_COUNT", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
cd /root/census/probes/shape && python -c 'import sys; sys.path.insert(0,"."); import teacher_forcing; print("TEACHER_FORCING_VLLM_INTERFACE", teacher_forcing.VLLM_INTERFACE)'
cd /root/census/probes/shape && python - <<'PY'
import sys; sys.path.insert(0, ".")
from shape_common import cache_roots, cache_state, cache_is_empty, machine_identity
import json
print("CACHE_ROOTS", json.dumps(cache_roots(), indent=1))
print("MACHINE", json.dumps(machine_identity(), indent=1))
print("CACHE_EMPTY_NOW", cache_is_empty())
PY
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,serial,pci.bus_id,driver_version,vbios_version --format=csv
hostname; grep -m1 "model name" /proc/cpuinfo; nproc; free -g | head -2; uname -r; cat /etc/os-release | head -2
env | grep -E "^RUNPOD_(POD_ID|POD_HOSTNAME|DC_ID|GPU_COUNT|CPU_COUNT|PUBLIC_IP|TCP_PORT_22)=" | sort
curl -s --max-time 10 https://api.ipify.org; echo
pip freeze > /root/logs/pip-freeze.txt; wc -l /root/logs/pip-freeze.txt
echo POST_BOOTSTRAP_DONE
