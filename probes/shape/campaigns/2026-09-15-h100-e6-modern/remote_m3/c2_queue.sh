#!/usr/bin/env bash
# C2 queue. Usage: c2_queue.sh DEADLINE_EPOCH JOBFILE
# Job line: KIND|LABEL|CACHE|ENV|REST
#   KIND  record | replay | boundary
#   CACHE cold (clear every cache root and refuse unless 0 files) | warm | restore:<archive label>
#   ENV   space-separated VAR=VAL, or -
#   REST  record:   the full run_e6.py --record arguments (model, revision, topology, quantisation, graphs)
#         replay:   <arm dir name> <replay subdir name>
#         boundary: <arm dir name> <replay subdir name> <pass index>
# /root/logs/STOP stops the queue before the next job; so does the deadline.
set -u
source /root/remote/lib.sh
DEADLINE=$1; JOBFILE=$2
ROOT=results_c2/e6
cd /root/census/probes/shape
mkdir -p /tmp_art
while IFS='|' read -r KIND LABEL CACHE ENVS REST; do
  case "$KIND" in ''|'#'*) continue;; esac
  if [ -f "$LOGS/STOP" ]; then echo "QUEUE STOP present; not launching $LABEL $(stamp)"; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "QUEUE DEADLINE passed; not launching $LABEL $(stamp)"; break; fi
  # A replay may be queued before its record has finished arriving from the record machine: wait for the
  # record to be complete (artefacts.json is the last file run_e6.py writes) rather than failing the job.
  if [ "$KIND" = "replay" ] || [ "$KIND" = "boundary" ]; then
    set -- $REST; WARM=$1
    for i in $(seq 1 90); do [ -f "$ROOT/$WARM/record/artefacts.json" ] && break; sleep 10; done
    if [ ! -f "$ROOT/$WARM/record/artefacts.json" ]; then echo "QUEUE RECORD_MISSING $LABEL $WARM $(stamp)"; continue; fi
  fi
  gpu_idle
  case "$CACHE" in
    cold)
      clear_caches; cache_state > "$LOGS/$LABEL.cache_before.txt" 2>&1
      tot=$(awk '/^CACHE_TOTAL_FILES/{print $2}' "$LOGS/$LABEL.cache_before.txt")
      if [ "$tot" != "0" ]; then echo "QUEUE CACHE_NOT_EMPTY $LABEL files=$tot; job skipped $(stamp)"; continue; fi ;;
    restore:*)
      clear_caches; cache_state > "$LOGS/$LABEL.cache_cleared.txt" 2>&1
      tot=$(awk '/^CACHE_TOTAL_FILES/{print $2}' "$LOGS/$LABEL.cache_cleared.txt")
      if [ "$tot" != "0" ]; then echo "QUEUE CACHE_NOT_EMPTY $LABEL files=$tot; job skipped $(stamp)"; continue; fi
      restore_full "${CACHE#restore:}" || { echo "QUEUE RESTORE_FAILED $LABEL"; continue; }
      cache_state > "$LOGS/$LABEL.cache_before.txt" 2>&1 ;;
    warm)
      cache_state > "$LOGS/$LABEL.cache_before.txt" 2>&1 ;;
    *) echo "QUEUE BAD_CACHE_SPEC $LABEL $CACHE"; continue;;
  esac
  [ "$ENVS" = "-" ] && ENVS=""
  echo "QUEUE ARM_START $LABEL $(stamp) :: kind=$KIND cache=$CACHE env='$ENVS' caches_before=$(awk '/^CACHE_TOTAL_FILES/{print $2}' "$LOGS/$LABEL.cache_before.txt") :: $REST"
  rc=0
  case "$KIND" in
    record)
      eval "env $ENVS timeout 1500 python run_e6.py --record --out $ROOT --prefix-caching 0 $REST" > "$LOGS/$LABEL.run.log" 2>&1 || rc=$? ;;
    replay)
      set -- $REST; ARM=$1; OUT=$2
      eval "env $ENVS timeout 1500 python run_e6.py --replay $ROOT/$ARM/record --out $ROOT/$ARM/$OUT" > "$LOGS/$LABEL.run.log" 2>&1 || rc=$? ;;
    boundary)
      set -- $REST; ARM=$1; OUT=$2; P=$3
      eval "env $ENVS timeout 1500 python run_e6.py --replay $ROOT/$ARM/record --out $ROOT/$ARM/$OUT --boundary $P" > "$LOGS/$LABEL.run.log" 2>&1 || rc=$? ;;
  esac
  echo "QUEUE RUN_EXIT $LABEL rc=$rc $(stamp)"
  cache_state > "$LOGS/$LABEL.cache_after.txt" 2>&1
  archive_full "$LABEL"  >> "$LOGS/$LABEL.archive.log" 2>&1
  archive_light "$LABEL" >> "$LOGS/$LABEL.archive.log" 2>&1
  tail -2 "$LOGS/$LABEL.archive.log" | sed "s/^/  [$LABEL] /"
  L=$LOGS/$LABEL.run.log
  echo "QUEUE FACTS $LABEL direct_load=$(grep -c 'Directly load AOT compilation' $L) caches_after=$(awk '/^CACHE_TOTAL_FILES/{print $2}' $LOGS/$LABEL.cache_after.txt)"
  grep -hE "Compiling a graph|torch.compile took|init engine|Graph capturing finished|FlashAttention|Selected |KV cache size|all-reduce backends|^E6 " $L | cut -c1-240 | sed "s/^/  [$LABEL] /"
  grep -E "Error|Traceback|refused|RuntimeError" $L | grep -v "^INFO" | tail -6 | sed "s/^/  [$LABEL ERR] /"
  # comparison: every replay against its record
  if [ "$KIND" = "replay" ] || [ "$KIND" = "boundary" ]; then
    set -- $REST; ARM=$1; OUT=$2
    if [ -d "$ROOT/$ARM/$OUT" ]; then
      python run_e6.py --compare "$ROOT/$ARM/record" "$ROOT/$ARM/$OUT" > "$LOGS/$LABEL.compare.log" 2>&1; crc=$?
      echo "QUEUE COMPARE_EXIT $LABEL rc=$crc $(stamp)"
      grep -E "^E6 " "$LOGS/$LABEL.compare.log" | sed "s|^|[$(stamp)] [$LABEL compare] |" | tee -a "$LOGS/verdicts.txt"
    fi
  fi
  echo "QUEUE ARM_END $LABEL $(stamp)"
done < "$JOBFILE"
echo "QUEUE_DONE $(stamp)"
