#!/usr/bin/env bash
# Shared helpers for the C1 queues: cache clearing and listing, artefact archiving and restoring.
LOGS=/root/logs; ART=/root/artefacts
export PATH=/root/venv/bin:$PATH HF_HOME=/root/hf HF_HUB_OFFLINE=1 PIP_CACHE_DIR=/root/pip-cache
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_DISABLE_COMPILE_CACHE=1
unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR DG_JIT_CACHE_DIR FLASHINFER_WORKSPACE_BASE CUDA_CACHE_PATH VLLM_CACHE_ROOT XDG_CACHE_HOME NCCL_NVLS_ENABLE TORCHINDUCTOR_DETERMINISTIC
# Every root shape_common.cache_roots() names, plus the two extra locations the 15 September cold campaign found.
CACHE_DIRS="/root/.cache /root/.triton /root/.nv /root/.deep_gemm /root/.tilelang /root/.cutlass /root/.humming /root/.config/vllm /tmp/torchinductor_root"
# Roots archived and restored whole for the artefact-restored arms.
ARCHIVE_DIRS="/root/.cache /root/.triton /root/.nv /root/.deep_gemm /tmp/torchinductor_root"
stamp() { date -u +%FT%TZ; }
gpu_idle() { for i in $(seq 1 45); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; echo "WARNING GPU still busy"; }
clear_caches() {
  for d in $CACHE_DIRS; do rm -rf "$d"; done
  find /tmp /var/tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null
  true
}
cache_state() {  # one line per location, then CACHE_TOTAL_FILES n
  local total=0 n b
  for d in $CACHE_DIRS /tmp /var/tmp; do
    if [ -e "$d" ]; then n=$(find "$d" -type f 2>/dev/null | wc -l); b=$(du -sb "$d" 2>/dev/null | cut -f1); echo "CACHE $d exists files=$n bytes=$b"
      find "$d" -maxdepth 3 2>/dev/null | head -40 | sed 's/^/  /'
    else n=0; echo "CACHE $d absent"; fi
    total=$((total + n))
  done
  echo "CACHE_TOTAL_FILES $total"
}
# Full archive of the compiled artefacts, for restoring on either machine.
archive_full() {  # archive_full LABEL
  local L present
  L="$1"; present=""
  mkdir -p $ART
  for d in $ARCHIVE_DIRS; do [ -d "$d" ] && present="$present $d"; done
  tar -I 'gzip -1' -cf "$ART/$L.full.tgz" -P $present 2>/dev/null
  echo "ARCHIVE_FULL $L bytes=$(stat -c %s "$ART/$L.full.tgz" 2>/dev/null) dirs=$present"
}
# The Inductor output code, its autotune choices, the vLLM compile cache and the Triton kernel sources: the
# small archive that is synced off the pod for the record/replay artefact diff.
archive_light() {  # archive_light LABEL: the Inductor output code, its autotune choices, the vLLM compile cache
  local L T LIST                       # metadata and the Triton kernel sources, plus a per-file digest manifest
  L="$1"; LIST=/root/logs/$L.artefact_files.txt
  : > "$LIST"
  for d in $ARCHIVE_DIRS; do
    [ -d "$d" ] || continue
    find "$d" \( -name '*.py' -o -name '*.best_config' -o -name '*.json' -o -name '*.ttir' -o -name '*.ttgir' -o -name '*.llir' -o -name '*.ptx' -o -name '*.txt' -o -name '*.cpp' -o -name '*.cu' \) -type f >> "$LIST" 2>/dev/null
  done
  sort -o "$LIST" "$LIST"
  xargs -r -a "$LIST" sha256sum > "$ART/$L.artefact_sha256.txt" 2>/dev/null
  tar -I 'gzip -1' -cf "$ART/$L.light.tgz" -P -T "$LIST" 2>/dev/null
  echo "ARCHIVE_LIGHT $L files=$(wc -l < "$LIST") bytes=$(stat -c %s "$ART/$L.light.tgz" 2>/dev/null)"
}
restore_full() {  # restore_full LABEL  (caches must have been cleared first)
  local L
  L="$1"
  [ -f "$ART/$L.full.tgz" ] || { echo "RESTORE_MISSING $L"; return 1; }
  tar -I 'gzip -1' -xf "$ART/$L.full.tgz" -P -C /
  echo "RESTORE_FULL $L files=$(for d in $ARCHIVE_DIRS; do find $d -type f 2>/dev/null; done | wc -l)"
}
