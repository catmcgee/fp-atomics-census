#!/usr/bin/env bash
# Diagnostic extra cold replays (not part of the eleven-arm queue): the same clearing, checks and logging as cold_queue.sh,
# into a replay directory named by REPNAME; SAVE_CACHES=1 also archives the caches each replay wrote, for a kernel-level diff.
# Usage: REPNAME=replay_cold2 SAVE_CACHES=1 [KEEP_VLLM_CACHE_ENABLED=1] cold_extra.sh DEADLINE_EPOCH "LABEL|CUDA_VISIBLE_DEVICES|NVLS|ARM" ...
# after it, the caches it wrote and every file it wrote outside the results are logged.
# Logs: /root/logs/<LABEL>.{replay_cold,compare_cold}.log, <LABEL>.cache_{before,after}.txt, <LABEL>.writes*.txt, verdicts_extra.txt
# A file /root/logs/STOP stops the queue before the next arm; so does passing DEADLINE_EPOCH.
set -u; REPNAME=${REPNAME:?set REPNAME}; SAVE_CACHES=${SAVE_CACHES:-0}
DEADLINE=$1; shift
LOGS=/root/logs; ROOT=results_cold/e6
export PATH=/root/venv/bin:$PATH HF_HOME=/root/hf HF_HUB_OFFLINE=1 PIP_CACHE_DIR=/root/pip-cache
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_DISABLE_COMPILE_CACHE=1
# KEEP_VLLM_CACHE_ENABLED=1 reproduces the 14 September plain and mixed records' environment, which did not set VLLM_DISABLE_COMPILE_CACHE;
# the vLLM cache directory is still deleted and checked empty first, so the replay compiles fresh and can only save, not load.
[ "${KEEP_VLLM_CACHE_ENABLED:-0}" = 1 ] && unset VLLM_DISABLE_COMPILE_CACHE
unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR DG_JIT_CACHE_DIR FLASHINFER_WORKSPACE_BASE CUDA_CACHE_PATH VLLM_CACHE_ROOT XDG_CACHE_HOME NCCL_NVLS_ENABLE
CACHE_DIRS="/root/.cache /root/.triton /root/.nv /root/.deep_gemm /root/.tilelang /root/.cutlass /tmp/torchinductor_root"
cd /root/census/probes/shape
stamp() { date -u +%FT%TZ; }
gpu_idle() { for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; echo "WARNING GPU still busy"; }
extra_dirs() { [ -f $LOGS/extra_cache_dirs.txt ] && grep -vE '^\s*(#|$)' $LOGS/extra_cache_dirs.txt; }
clear_caches() {
  for d in $CACHE_DIRS $(extra_dirs); do rm -rf "$d"; done
  find /tmp /var/tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null
}
cache_state() {  # prints one line per location; CACHE_TOTAL_FILES n
  local total=0
  for d in $CACHE_DIRS $(extra_dirs) /tmp /var/tmp; do
    if [ -e "$d" ]; then n=$(find "$d" -type f 2>/dev/null | wc -l); b=$(du -sb "$d" 2>/dev/null | cut -f1); echo "CACHE $d exists files=$n bytes=$b"
      find "$d" -maxdepth 3 2>/dev/null | head -60 | sed 's/^/  /'
    else n=0; echo "CACHE $d absent"; fi
    total=$((total + n))
  done
  echo "CACHE_TOTAL_FILES $total"
}
writes() {  # every regular file written since the marker outside weights, code/results, logs and pip cache
  find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /root/hf -o -path /root/census -o -path /root/logs -o -path /root/pip-cache -o -path /root/hfenv \) -prune \
    -o -newer "$1" -type f -print 2>/dev/null | grep -v '/__pycache__/' | sort
}
for spec in "$@"; do
  IFS='|' read -r LABEL CVD NVLS ARM <<< "$spec"
  if [ -f "$LOGS/STOP" ]; then echo "EXTRA STOP present; not launching $LABEL $(stamp)"; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "EXTRA DEADLINE passed; not launching $LABEL $(stamp)"; break; fi
  [ -f "$ROOT/$ARM/record/run.json" ] || { echo "EXTRA ARM_MISSING $LABEL $ARM"; continue; }
  gpu_idle
  clear_caches
  cache_state > "$LOGS/$LABEL.cache_before.txt" 2>&1
  tot=$(awk '/^CACHE_TOTAL_FILES/{print $2}' "$LOGS/$LABEL.cache_before.txt")
  if [ "$tot" != "0" ]; then echo "EXTRA CACHE_NOT_EMPTY $LABEL files=$tot; arm skipped $(stamp)"; continue; fi
  touch "$LOGS/.mark_$LABEL"; sleep 1
  if [ "$NVLS" = "0" ]; then NV="NCCL_NVLS_ENABLE=0"; else NV=""; fi
  echo "EXTRA ARM_START $LABEL $(stamp) :: CUDA_VISIBLE_DEVICES=$CVD ${NV:-NCCL_NVLS_ENABLE=unset} VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-unset} caches_before=0 :: $ARM"
  env CUDA_VISIBLE_DEVICES=$CVD $NV timeout 720 python run_e6.py --replay "$ROOT/$ARM/record" --out "$ROOT/$ARM/$REPNAME" > "$LOGS/$LABEL.$REPNAME.log" 2>&1; rc=$?
  echo "EXTRA REPLAY_EXIT $LABEL rc=$rc $(stamp)"
  cache_state > "$LOGS/$LABEL.cache_after.txt" 2>&1; [ "$SAVE_CACHES" = 1 ] && tar czf "$LOGS/$LABEL.caches.tgz" /tmp/torchinductor_root /root/.triton /root/.cache 2>/dev/null
  writes "$LOGS/.mark_$LABEL" > "$LOGS/$LABEL.writes.txt"
  awk -F/ '{print "/"$2"/"$3"/"$4}' "$LOGS/$LABEL.writes.txt" | sort | uniq -c | sort -rn > "$LOGS/$LABEL.writes_summary.txt"
  L=$LOGS/$LABEL.$REPNAME.log
  echo "EXTRA FACTS $LABEL direct_load=$(grep -c 'Directly load AOT compilation' $L) saved_aot=$(grep -c 'saved AOT compiled function' $L) cache_disabled_msg=$(grep -c "torch.compile cache is disabled" $L) caches_after=$(awk '/^CACHE_TOTAL_FILES/{print $2}' $LOGS/$LABEL.cache_after.txt) writes=$(wc -l < $LOGS/$LABEL.writes.txt)"
  grep -hE "Compiling a graph|torch.compile took|torch.compile and initial|init engine|Graph capturing finished|FlashAttention|Selected |all-reduce|KV cache size|Using .*backend|^E6 " $L | cut -c1-260 | sed "s/^/  [$LABEL] /"
  grep -oE "DeepGEMM warmup: 100%[^]]*\]" $L | tail -1 | sed "s/^/  [$LABEL] /"
  grep -E "Error|Traceback|refused" $L | grep -v "^INFO" | tail -5 | sed "s/^/  [$LABEL ERR] /"
  if [ -d "$ROOT/$ARM/$REPNAME" ]; then
    python run_e6.py --compare "$ROOT/$ARM/record" "$ROOT/$ARM/$REPNAME" > "$LOGS/$LABEL.compare_$REPNAME.log" 2>&1; rc=$?
    echo "EXTRA COMPARE_EXIT $LABEL rc=$rc $(stamp)"
    grep -E "^E6 " "$LOGS/$LABEL.compare_$REPNAME.log" | sed "s|^|[$(stamp)] [$LABEL compare] |" | tee -a "$LOGS/verdicts_extra.txt"
  fi
  echo "EXTRA ARM_END $LABEL $(stamp)"
done
echo "EXTRA_DONE $(stamp)"
