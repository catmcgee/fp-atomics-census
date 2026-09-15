#!/usr/bin/env bash
# Pod bootstrap: venv on the container disk with the pinned stack. Run under nohup; log: /root/logs/bootstrap.log
set -x
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pip-cache PATH=/root/venv/bin:$PATH
mkdir -p /root/hf /root/pip-cache /root/census /root/logs
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 --version
[ -x /root/venv/bin/python ] || python3 -m venv /root/venv
pip install -q --upgrade pip
if pip install -q vllm==0.28.0 flashinfer-python==0.6.16.post3 hf_transfer ninja; then echo "PIP_PINNED_OK"; else echo "PIP_PINNED_FAILED"; fi
python -c 'import torch, vllm, triton; print("STACK torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__, "triton", triton.__version__)'
for p in flashinfer-python ninja transformers hf_transfer; do echo "PKG $p $(pip show $p 2>/dev/null | grep -i ^version | cut -d' ' -f2)"; done
which ninja; ninja --version
date -u
echo "BOOTSTRAP_DONE"
