#!/usr/bin/env bash
# Diagnostics after the eleven-arm queue (replaces launch_extra.sh, stopped before it launched anything). Log /root/logs/queue_extra.log
# 1. the 14 September plain record replayed cold in its own environment (VLLM_DISABLE_COMPILE_CACHE unset, vLLM cache directory empty);
# 2. the same record replayed cold again with VLLM_DISABLE_COMPILE_CACHE=1, for the repeatability of a fresh compile on this pod;
# 3. the 14 September mixed record replayed cold in its own environment, if the deadline allows.
DL=$1
for i in $(seq 1 400); do grep -q QUEUE_DONE /root/logs/queue_cold.log 2>/dev/null && break; [ -f /root/logs/STOP ] && exit 1; sleep 5; done
A=Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0
REPNAME=replay_cold_vcache SAVE_CACHES=1 KEEP_VLLM_CACHE_ENABLED=1 bash /root/remote/cold_extra.sh "$DL" "q25_plain_vcache|0|-|$A"
REPNAME=replay_cold2 SAVE_CACHES=1 bash /root/remote/cold_extra.sh "$DL" "q25_plain_c2|0|-|$A"
REPNAME=replay_cold_vcache SAVE_CACHES=0 KEEP_VLLM_CACHE_ENABLED=1 bash /root/remote/cold_extra.sh "$DL" "q25_mixed_vcache|0|-|${A}_mixed"
