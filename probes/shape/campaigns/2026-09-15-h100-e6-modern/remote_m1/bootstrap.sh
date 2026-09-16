#!/usr/bin/env bash
# Pod bootstrap for C1: venv with vLLM 0.29.0. vLLM 0.29.0 pins flashinfer-python==0.6.18, which under PEP 440
# excludes 0.6.18.post1, so the post release the campaign targets is installed over it with --no-deps and the
# resulting pip conflict is recorded rather than hidden. Log: /root/logs/bootstrap.log
set -x
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pip-cache PATH=/root/venv/bin:$PATH
mkdir -p /root/hf /root/pip-cache /root/census /root/logs /root/artefacts
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 --version
[ -x /root/venv/bin/python ] || python3 -m venv /root/venv
pip install -q --upgrade pip
if pip install -q vllm==0.29.0 hf_transfer ninja; then echo "PIP_VLLM_OK"; else echo "PIP_VLLM_FAILED"; fi
pip show flashinfer-python | head -2
if pip install -q --no-deps --force-reinstall flashinfer-python==0.6.18.post1; then echo "PIP_FLASHINFER_POST1_OK"; else echo "PIP_FLASHINFER_POST1_FAILED"; fi
pip check; echo "PIP_CHECK_RC $?"
python -c 'import torch, vllm, triton; print("STACK torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__, "triton", triton.__version__)'
for p in flashinfer-python ninja transformers hf_transfer; do echo "PKG $p $(pip show $p 2>/dev/null | grep -i ^version | cut -d' ' -f2)"; done
which ninja; ninja --version
date -u
echo "BOOTSTRAP_DONE"
