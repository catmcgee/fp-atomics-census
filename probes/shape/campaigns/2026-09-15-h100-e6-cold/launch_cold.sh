#!/usr/bin/env bash
# The exact queue invocation (run under nohup; log /root/logs/queue_cold.log). DEADLINE_EPOCH is the first argument.
exec bash /root/remote/cold_queue.sh "$1" \
  "q25_plain|0|-|Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0" \
  "q25_mixed|0|-|Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0_mixed" \
  "q25_fp8_pertensor|0|-|Qwen_Qwen2.5-7B-Instruct_tp1_fp8_pertensor_compile_v2_graphs1_prefix0" \
  "q25_fp8_pertoken|0|-|Qwen_Qwen2.5-7B-Instruct_tp1_fp8_compile_v2_graphs1_prefix0" \
  "q3fp8_plain|0|-|Qwen_Qwen3-8B-FP8_tp1_fp8_compile_v2_graphs1_prefix0" \
  "q3fp8_mixed|0|-|Qwen_Qwen3-8B-FP8_tp1_fp8_compile_v2_graphs1_prefix0_mixed" \
  "llama_plain|0|-|NousResearch_Meta-Llama-3.1-8B-Instruct_tp1_none_compile_v2_graphs1_prefix0" \
  "moe_graphs0|0|-|Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs0_prefix0" \
  "moe_graphs1|0|-|Qwen_Qwen1.5-MoE-A2.7B-Chat_tp1_none_nocompile_v2_graphs1_prefix0" \
  "q25_tp2_plain|0,1|0|Qwen_Qwen2.5-7B-Instruct_tp2_none_compile_v2_graphs1_prefix0" \
  "q25_tp2_mixed|0,1|0|Qwen_Qwen2.5-7B-Instruct_tp2_none_compile_v2_graphs1_prefix0_mixed" > /root/logs/queue_cold.log 2>&1
