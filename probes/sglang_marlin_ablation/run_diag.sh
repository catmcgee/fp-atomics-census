#!/bin/bash
# Diagnostic follow-up (not census probes): sampled-token change counts per arm.
set -u
export HF_HOME=/root/hf PATH=/root/venv-sgl/bin:$PATH
cd /workspace/census/probes
LOGS=/workspace/logs
F=/root/venv-sgl/lib/python3.12/site-packages/sglang/srt/layers/quantization/marlin_utils.py
PYC_DIR=/root/venv-sgl/lib/python3.12/site-packages/sglang/srt/layers/quantization/__pycache__
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cleanup() { pkill -f "sglang::" 2>/dev/null; sleep 2; for i in $(seq 1 15); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done; }
TEST='import os, torch, sglang.srt.layers.quantization.marlin_utils as m
print("SGLANG_MARLIN_USE_ATOMIC_ADD =", os.environ.get("SGLANG_MARLIN_USE_ATOMIC_ADD"), "| should_use_atomic_add_reduce(16,1536,8960,cuda,fp16) =", m.should_use_atomic_add_reduce(16, 1536, 8960, torch.device("cuda"), torch.float16))'
pyccheck() { python - "$F" "$PYC_DIR/marlin_utils.cpython-312.pyc" <<'EOF'
import sys, os, struct
src, pyc = sys.argv[1], sys.argv[2]
if not os.path.exists(pyc): print("pyc absent (will be regenerated from the current source on next import)"); sys.exit()
b = open(pyc, "rb").read(16); flags = struct.unpack("<I", b[4:8])[0]
if flags & 1: print("pyc hash-based; flags", flags)
else:
    mtime, size = struct.unpack("<II", b[8:16]); print(f"pyc mtime={mtime} size={size}; source mtime={int(os.stat(src).st_mtime)} size={os.path.getsize(src)}; match={size == os.path.getsize(src) and mtime == int(os.stat(src).st_mtime)}")
EOF
}
run_diag() { local name=$1; shift; cleanup; log "DIAG $name (SGLANG_MARLIN_USE_ATOMIC_ADD=${SGLANG_MARLIN_USE_ATOMIC_ADD:-unset})"; python -c "$TEST" 2>&1 | tail -1; python $LOGS/diag_token_diffs.py "$name" "$LOGS/diag_$name.json" 6 log_level=info "$@" 2>&1 | grep -E "^\{|DIAG_DONE|Traceback|Error|gptq_marlin kernel"; log "DIAG $name exit=${PIPESTATUS[0]}"; }

unset SGLANG_MARLIN_USE_ATOMIC_ADD
cp -p $F $LOGS/marlin_utils.py.patched
log "state at start:"; sha256sum $F $LOGS/marlin_utils.py.stock $LOGS/marlin_utils.py.patched; pyccheck
log "restore the stock file"; cp -p $LOGS/marlin_utils.py.stock $F; rm -f $PYC_DIR/marlin_utils.cpython-312*.pyc; sha256sum $F
run_diag stock
log "re-apply the patch"; cp -p $LOGS/marlin_utils.py.patched $F; rm -f $PYC_DIR/marlin_utils.cpython-312*.pyc; sha256sum $F; diff -u $LOGS/marlin_utils.py.stock $F | diff - $LOGS/wheel_guard.patch >/dev/null && echo "patch identical to wheel_guard.patch" || echo "PATCH MISMATCH"
run_diag guard_off
export SGLANG_MARLIN_USE_ATOMIC_ADD=1
run_diag guard_env_on
unset SGLANG_MARLIN_USE_ATOMIC_ADD
run_diag guard_off_det enable-deterministic-inference=true
log "final pyc check:"; pyccheck; sha256sum $F
cleanup
log "DIAG_DONE_ALL"
