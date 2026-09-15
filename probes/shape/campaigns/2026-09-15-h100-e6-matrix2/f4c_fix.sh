#!/usr/bin/env bash
# Rerun of arm 4c with the harness fix (SHAPE_FORCE_FP8_PER_TENSOR set before shape_hook.register()).
# Record and replay run from /root/census_fix (a copy of the pod tree with only run_e6.py patched);
# --compare runs from the unpatched /root/census tree so the summary's analyser digest is the main checkout's.
# The first 4c arm (forcing not applied) is moved, not deleted, to results_matrix2/e6_failed_attempts_pertensor_not_forced/.
# Log: /root/logs/queue_F4cfix.log
set -u
LOGS=/root/logs; ROOT=/root/census/probes/shape/results_matrix2/e6; LABEL=F4cfix
ARM=Qwen_Qwen2.5-7B-Instruct_tp1_fp8_pertensor_compile_v2_graphs1_prefix0
export PATH=/root/venv/bin:$PATH HF_HOME=/root/hf HF_HUB_OFFLINE=1 PIP_CACHE_DIR=/root/pip-cache
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_DISABLE_COMPILE_CACHE=1 CUDA_VISIBLE_DEVICES=0
stamp() { date -u +%FT%TZ; }
gpu_idle() { for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; echo "WARNING GPU still busy"; }
echo "F4CFIX WAIT_Q7 $(stamp)"
for i in $(seq 1 240); do grep -q "QUEUE_DONE Q7" $LOGS/queue_Q7.log && break; [ -f $LOGS/STOP ] && { echo "STOP present"; exit 1; }; sleep 5; done
mkdir -p /root/census_fix && rsync -a --exclude '/probes/results' --exclude 'shape/results*' --exclude '__pycache__' /root/census/probes /root/census_fix/
cp /root/census/scan-manifest.json /root/census_fix/ 2>/dev/null
cp /root/remote/run_e6.fixed.py /root/census_fix/probes/shape/run_e6.py
diff -u /root/census/probes/shape/run_e6.py /root/census_fix/probes/shape/run_e6.py > $LOGS/harness_changes_pod.patch; echo "F4CFIX PATCH_LINES $(wc -l < $LOGS/harness_changes_pod.patch)"
mkdir -p /root/census/probes/shape/results_matrix2/e6_failed_attempts_pertensor_not_forced
[ -d "$ROOT/$ARM" ] && mv "$ROOT/$ARM" /root/census/probes/shape/results_matrix2/e6_failed_attempts_pertensor_not_forced/
gpu_idle; rm -rf /root/.cache/vllm/torch_compile_cache
echo "F4CFIX RECORD_START $(stamp) :: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cd /root/census_fix/probes/shape
python run_e6.py --record --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --quantization fp8 --fp8-per-tensor --cudagraph 1 --prefix-caching 0 --out "$ROOT" > $LOGS/$LABEL.record.log 2>&1; echo "F4CFIX RECORD_EXIT rc=$? $(stamp)"
grep -E "^E6 " $LOGS/$LABEL.record.log | tee -a $LOGS/verdicts.txt
echo "F4CFIX AOT_CHECK record direct_load=$(grep -c 'Directly load AOT compilation' $LOGS/$LABEL.record.log)"
[ -f "$ROOT/$ARM/record/outputs.json" ] || { echo "F4CFIX no usable record"; exit 1; }
python /root/remote/text_check.py "$ROOT/$ARM/record" "$LABEL" > "$LOGS/text_check_$LABEL.md" 2> "$LOGS/text_check_$LABEL.err"; grep TEXT_CHECK "$LOGS/text_check_$LABEL.md"
gpu_idle; rm -rf /root/.cache/vllm/torch_compile_cache
python run_e6.py --replay "$ROOT/$ARM/record" --out "$ROOT/$ARM/replay" > $LOGS/$LABEL.replay.log 2>&1; echo "F4CFIX REPLAY_EXIT rc=$? $(stamp)"
grep -E "^E6 " $LOGS/$LABEL.replay.log | tee -a $LOGS/verdicts.txt
echo "F4CFIX AOT_CHECK replay direct_load=$(grep -c 'Directly load AOT compilation' $LOGS/$LABEL.replay.log)"
cd /root/census/probes/shape
python run_e6.py --compare "$ROOT/$ARM/record" "$ROOT/$ARM/replay" > $LOGS/$LABEL.compare.log 2>&1; echo "F4CFIX COMPARE_EXIT rc=$? $(stamp)"
grep -E "^E6 " $LOGS/$LABEL.compare.log | tee -a $LOGS/verdicts.txt
grep -hE "Selected" $LOGS/$LABEL.record.log $LOGS/$LABEL.replay.log
echo "F4CFIX_DONE $(stamp)"
