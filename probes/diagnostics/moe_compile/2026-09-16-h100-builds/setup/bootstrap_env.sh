#!/usr/bin/env bash
set -euo pipefail
build="$1"
case "$build" in cu129|cu130) ;; *) exit 2;; esac
export PIP_CACHE_DIR=/root/pip-cache PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p /root/logs /root/pip-cache /opt/issue56900
python3 -m venv "/opt/issue56900/$build"
py="/opt/issue56900/$build/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install --index-url "https://download.pytorch.org/whl/$build" "torch==2.13.0+$build" "torchvision==0.28.0+$build" "torchaudio==2.11.0+$build"
if [ "$build" = cu129 ]; then
 wheel='https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397'
else
 wheel='https://files.pythonhosted.org/packages/87/d7/97f6ecc2ae883e601e08d7cef87cb54ececeefcfe6b5e12d5d92f8d06d6b/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889'
fi
"$py" -m pip install "$wheel" --extra-index-url "https://download.pytorch.org/whl/$build" 'triton==3.7.1' 'tokenspeed-triton==3.8.10.post20260906' 'tokenspeed-mla==0.1.8' 'flashinfer-python==0.6.16.post3' 'transformers==5.17.0' 'ninja==1.13.2'
"$py" -m pip check
"$py" -m pip freeze --all > "/root/logs/$build.freeze.txt"
"$py" -c 'import torch,triton,vllm; print("STACK",torch.__version__,torch.version.cuda,triton.__version__,vllm.__version__);print("ALLOC",torch.zeros(1,device="cuda"))'
printf 'complete\n' > "/root/logs/$build.ready"
