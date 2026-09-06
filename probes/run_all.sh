#!/usr/bin/env bash
# Run the probe suite on one machine, each probe twice in fresh processes
# (RUN_TAG=a and RUN_TAG=b), then print the report. Engine probes need the
# models below on the Hugging Face hub; set HF_TOKEN for gated ones.
#
#   bash probes/run_all.sh            # everything that fits one GPU
#   bash probes/run_all.sh quick      # engine-free probes only
#   bash probes/run_all.sh engines    # engine probes only
#   bash probes/run_all.sh cublaslt   # the cuBLASLt sweeps only
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
MODE=${1:-full}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-}

twice() {  # run a probe under RUN_TAG=a then RUN_TAG=b; never abort the suite
  for tag in a b; do
    echo "== RUN_TAG=$tag $*"
    RUN_TAG=$tag $PY "$@" || echo "!! probe exited $? : $*"
  done
}

if [ "$MODE" = "full" ] || [ "$MODE" = "quick" ]; then
twice probe_torch_ops.py
fi
if [ "$MODE" != "engines" ]; then
twice probe_cublaslt_algo.py --sweep 7b
twice probe_cublaslt_algo.py --sweep 70b
fi
if [ "$MODE" = "full" ] || [ "$MODE" = "quick" ]; then
twice probe_flashinfer.py --which renorm
twice probe_flashinfer.py --which topk
twice probe_flashinfer.py --which moe
twice probe_flashinfer.py --which attention --backend fa2
twice probe_flashinfer.py --which attention --backend trtllm-gen
twice probe_kernels.py --which sglang_fp8_blockwise
twice probe_kernels.py --which deepgemm_bmk_bnk_mn
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "engines" ]; then
  twice probe_engine_logits.py --engine vllm --model Qwen/Qwen3-8B
  twice probe_engine_logits.py --engine vllm --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
  twice probe_engine_logits.py --engine vllm --model Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --quantization moe_wna16 --name vllm_qwen1.5_moe_wna16
  twice probe_engine_logits.py --engine vllm --model HuggingFaceH4/zephyr-7b-beta --lora typeof/zephyr-7b-beta-lora --name vllm_zephyr_lora_split_k
  VLLM_BATCH_INVARIANT=1 twice probe_engine_logits.py --engine vllm --model HuggingFaceH4/zephyr-7b-beta --lora typeof/zephyr-7b-beta-lora --name vllm_zephyr_lora_batch_invariant
fi

$PY report.py

echo SUITE_DONE
