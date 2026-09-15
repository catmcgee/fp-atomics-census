#!/usr/bin/env bash
# Diagnostic, after the eleven-arm queue: two more cold replays of the 14 September plain record on this pod, each from empty caches,
# with the caches each wrote archived, to tell whether a fresh compile on one machine is repeatable. Log /root/logs/queue_extra.log
DL=$1
for i in $(seq 1 400); do grep -q QUEUE_DONE /root/logs/queue_cold.log 2>/dev/null && break; [ -f /root/logs/STOP ] && exit 1; sleep 5; done
A=Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0
REPNAME=replay_cold2 SAVE_CACHES=1 bash /root/remote/cold_extra.sh "$DL" "q25_plain_c2|0|-|$A"
REPNAME=replay_cold3 SAVE_CACHES=1 bash /root/remote/cold_extra.sh "$DL" "q25_plain_c3|0|-|$A"
