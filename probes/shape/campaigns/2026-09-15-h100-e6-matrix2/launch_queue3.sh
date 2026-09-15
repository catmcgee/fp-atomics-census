#!/usr/bin/env bash
# Third queue (the second, Q3, misfired on existing arm directories), after the first queue's TP=2 arms failed at NCCL init (NVLS multicast bind, CUDA error 401).
# TP=2 arms with NCCL_NVLS_ENABLE=0 (environment fix, record and replay alike); then Q4 without it; then the SGLang arm.
# Log /root/logs/queue_Q5.log (TP=2) and /root/logs/queue_Q6.log (Q4 and SGLang).
NO_SGL=1 NCCL_NVLS_ENABLE=0 bash /root/remote/matrix.sh Q5 \
  "T1b|0,1|--model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --tp 2 --cudagraph 1 --prefix-caching 0" \
  "T2b|0,1|--model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --tp 2 --cudagraph 1 --prefix-caching 0 --mixed" > /root/logs/queue_Q5.log 2>&1
bash /root/remote/matrix.sh Q6 \
  "Q4|0|--model Qwen/Qwen3-8B-FP8 --revision 220b46e3b2180893580a4454f21f22d3ebb187d3 --quantization fp8 --cudagraph 1 --prefix-caching 0 --mixed" > /root/logs/queue_Q6.log 2>&1
