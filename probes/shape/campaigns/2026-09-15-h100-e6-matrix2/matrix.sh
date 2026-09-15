#!/usr/bin/env bash
# E6 matrix 2: for each ARMSPEC "LABEL|CUDA_VISIBLE_DEVICES|record-args", run --record, then --replay in a fresh process, then --compare.
# Then, unless /root/logs/SKIP_SGL exists, the SGLang residual arm (sgl_arm.sh).
# Logs: /root/logs/<LABEL>.{record,replay,compare}.log; verdicts: /root/logs/verdicts.txt; text checks: /root/logs/text_check_<LABEL>.md
# A file /root/logs/STOP stops the queue before the next arm is launched.
set -u
QN=$1; shift
ROOT=/root/census/probes/shape/results_matrix2/e6
LOGS=/root/logs
export PATH=/root/venv/bin:$PATH HF_HOME=/root/hf HF_HUB_OFFLINE=1 PIP_CACHE_DIR=/root/pip-cache
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_DISABLE_COMPILE_CACHE=1
cd /root/census/probes/shape
mkdir -p "$ROOT"
stamp() { date -u +%FT%TZ; }
gpu_idle() {  # no engine process may survive into the next invocation
  for i in $(seq 1 30); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2
  done
  echo "WARNING GPU still busy: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
}
clear_cache() { rm -rf /root/.cache/vllm/torch_compile_cache; }
aot() { echo "QUEUE $QN AOT_CHECK $1 direct_load=$(grep -c 'Directly load AOT compilation' "$2") saved_aot=$(grep -c 'saved AOT compiled function' "$2")"; }
verdicts() { grep -E "^E6 " "$1" | sed "s|^|[$(stamp)] [$2] |" | tee -a "$LOGS/verdicts.txt"; }
for spec in "$@"; do
  if [ -f "$LOGS/STOP" ]; then echo "QUEUE $QN STOP present; not launching $spec $(stamp)"; break; fi
  LABEL=${spec%%|*}; REST=${spec#*|}; CVD=${REST%%|*}; ARGS=${REST#*|}
  export CUDA_VISIBLE_DEVICES=$CVD
  gpu_idle; clear_cache
  echo "QUEUE $QN ARM_START $LABEL $(stamp) :: CUDA_VISIBLE_DEVICES=$CVD NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-unset} :: $ARGS"
  # shellcheck disable=SC2086
  python run_e6.py --record $ARGS --out "$ROOT" > "$LOGS/$LABEL.record.log" 2>&1; rc=$?
  echo "QUEUE $QN RECORD_EXIT $LABEL rc=$rc $(stamp)"; verdicts "$LOGS/$LABEL.record.log" "$LABEL record"; aot "$LABEL.record" "$LOGS/$LABEL.record.log"
  ARM=$(grep -oE "^E6 [^ ]+ record:" "$LOGS/$LABEL.record.log" | head -1 | awk '{print $2}')
  if [ -z "$ARM" ]; then
    echo "QUEUE $QN $LABEL record printed no E6 line; no arm directory is assumed (the newest-directory fallback was removed after it picked another arm)"
    grep -E "Error|error|Traceback|RuntimeError" "$LOGS/$LABEL.record.log" | grep -v "^INFO" | tail -8
  fi
  if [ -z "$ARM" ] || [ ! -f "$ROOT/$ARM/record/run.json" ] || [ ! -f "$ROOT/$ARM/record/outputs.json" ]; then echo "QUEUE $QN ARM_FAILED $LABEL (no usable record) $(stamp)"; echo "QUEUE $QN ARM_END $LABEL $(stamp)"; continue; fi
  echo "QUEUE $QN ARM_DIR $LABEL $ARM"
  python /root/remote/text_check.py "$ROOT/$ARM/record" "$LABEL" > "$LOGS/text_check_$LABEL.md" 2> "$LOGS/text_check_$LABEL.err"; grep TEXT_CHECK "$LOGS/text_check_$LABEL.md"
  gpu_idle; clear_cache
  echo "QUEUE $QN REPLAY_START $LABEL $(stamp)"
  python run_e6.py --replay "$ROOT/$ARM/record" --out "$ROOT/$ARM/replay" > "$LOGS/$LABEL.replay.log" 2>&1; rc=$?
  echo "QUEUE $QN REPLAY_EXIT $LABEL rc=$rc $(stamp)"; verdicts "$LOGS/$LABEL.replay.log" "$LABEL replay"; aot "$LABEL.replay" "$LOGS/$LABEL.replay.log"
  [ $rc -ne 0 ] && grep -E "Error|error|Traceback|RuntimeError|refused" "$LOGS/$LABEL.replay.log" | grep -v "^INFO" | tail -8
  if [ -d "$ROOT/$ARM/replay" ]; then
    python run_e6.py --compare "$ROOT/$ARM/record" "$ROOT/$ARM/replay" > "$LOGS/$LABEL.compare.log" 2>&1; rc=$?
    echo "QUEUE $QN COMPARE_EXIT $LABEL rc=$rc $(stamp)"; verdicts "$LOGS/$LABEL.compare.log" "$LABEL compare"
  fi
  echo "QUEUE $QN ARM_END $LABEL $(stamp)"
done
unset CUDA_VISIBLE_DEVICES
if [ -f "$LOGS/STOP" ] || [ -f "$LOGS/SKIP_SGL" ] || [ "${NO_SGL:-0}" = 1 ]; then echo "QUEUE $QN SGL skipped (STOP or SKIP_SGL present) $(stamp)"; else
  gpu_idle
  echo "QUEUE $QN SGL_WAIT_INSTALL $(stamp)"
  for i in $(seq 1 180); do grep -q SGL_INSTALL_DONE $LOGS/sgl_install.log 2>/dev/null && break; [ -f "$LOGS/STOP" ] && break; sleep 5; done
  if grep -q SGL_PIP_OK $LOGS/sgl_install.log && grep -q SGL_INSTALL_DONE $LOGS/sgl_install.log && [ ! -f "$LOGS/STOP" ]; then
    echo "QUEUE $QN SGL_START $(stamp)"
    bash /root/remote/sgl_arm.sh > $LOGS/sgl_arm.log 2>&1
    echo "QUEUE $QN SGL_END $(stamp)"; grep -E "^SGL |GUARD|should_use|RESULT|status" $LOGS/sgl_arm.log | tail -20
  else echo "QUEUE $QN SGL_NOT_RUN install incomplete or STOP $(stamp)"; fi
fi
echo "QUEUE_DONE $QN $(stamp)"
