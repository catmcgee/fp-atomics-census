#!/bin/bash
# SGLang Marlin atomic-add ablation (fp-atomics-census, control for row sglang-0008).
# Four arms, each RUN_TAG=a then RUN_TAG=b, same wheel/venv/model revision/prompts/tokens.
set -u
export HF_HOME=/root/hf PATH=/root/venv-sgl/bin:$PATH
cd /workspace/census/probes
LOGS=/workspace/logs
F=/root/venv-sgl/lib/python3.12/site-packages/sglang/srt/layers/quantization/marlin_utils.py
PYC_DIR=/root/venv-sgl/lib/python3.12/site-packages/sglang/srt/layers/quantization/__pycache__
REV=4f5a0e31008e2966e8f787964bcc4e1105e03dc3
COMMON="--engine sglang --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 --revision $REV --repeats 6 --tp 1"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
touch $LOGS/ablation_start_marker
unset SGLANG_MARLIN_USE_ATOMIC_ADD

TEST='import os, torch, sglang.srt.layers.quantization.marlin_utils as m
print("marlin_utils:", m.__file__)
print("SGLANG_MARLIN_USE_ATOMIC_ADD =", os.environ.get("SGLANG_MARLIN_USE_ATOMIC_ADD"))
print("should_use_atomic_add_reduce(16, 1536, 8960, cuda, fp16) =", m.should_use_atomic_add_reduce(16, 1536, 8960, torch.device("cuda"), torch.float16))
m.maybe_warn_marlin_atomic_add_env(); print("maybe_warn_marlin_atomic_add_env() returned without error")'

cleanup() {  # no engine subprocess may survive into the next arm
  pkill -f "sglang::" 2>/dev/null; sleep 2
  for i in $(seq 1 15); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2
  done
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
}

twice() {
  local name=$1; shift
  for tag in a b; do
    cleanup
    log "== RUN_TAG=$tag $name $* (SGLANG_MARLIN_USE_ATOMIC_ADD=${SGLANG_MARLIN_USE_ATOMIC_ADD:-unset})"
    RUN_TAG=$tag python invoke.py probe_engine_logits.py $COMMON --name "$name" "$@" 2>&1 | grep -v "^\s*$" | grep -E "PROBE |Traceback|Error|error|Warning: .*atomic|marlin|Marlin|use_atomic|Load weight|max_total_num_tokens|Capture cuda graph end|revision|Traceback" | tail -40
    log "== exit=${PIPESTATUS[0]} $name $tag"
  done
}

log "guard test on the stock wheel:"
python -c "$TEST" 2>&1 | grep -v "^\s*$" | tail -6
sha256sum $F
cp -p $F $LOGS/marlin_utils.py.stock

log "ARM 1 stock"
twice sglang_qwen2.5_1.5b_gptq_marlin_stock_x6

log "patching the installed wheel"
python - "$F" <<'EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '    if not True:\n        maybe_warn_marlin_atomic_add_env()\n        return False\n'
new = '    if not os.environ.get("SGLANG_MARLIN_USE_ATOMIC_ADD", "0") == "1":\n        maybe_warn_marlin_atomic_add_env()\n        return False\n'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
assert s.count('\nimport logging\n') == 1 and '\nimport os\n' not in s
s = s.replace('\nimport logging\n', '\nimport logging\nimport os\n', 1)
p.write_text(s)
print("patched", p)
EOF
rm -fv $PYC_DIR/marlin_utils.cpython-312*.pyc
find /root/venv-sgl/lib/python3.12/site-packages/sglang -name "marlin_utils.*.pyc" -print -delete
diff -u $LOGS/marlin_utils.py.stock $F > $LOGS/wheel_guard.patch
log "patch ($(wc -l < $LOGS/wheel_guard.patch) lines):"
cat $LOGS/wheel_guard.patch
sha256sum $F
log "guard test on the patched wheel, env unset:"
python -c "$TEST" 2>&1 | grep -v "^\s*$" | tail -6
log "guard test on the patched wheel, env=1:"
SGLANG_MARLIN_USE_ATOMIC_ADD=1 python -c "$TEST" 2>&1 | grep -v "^\s*$" | tail -6
log "guard test on the patched wheel, env=0:"
SGLANG_MARLIN_USE_ATOMIC_ADD=0 python -c "$TEST" 2>&1 | grep -v "^\s*$" | tail -6
log "pycache now:"; ls -la $PYC_DIR | grep "marlin_utils\."

log "ARM 2 guard_off"
twice sglang_qwen2.5_1.5b_gptq_marlin_guard_off_x6

log "ARM 3 guard_env_on"
( while true; do for p in /proc/[0-9]*; do if grep -qz "^SGLANG_MARLIN_USE_ATOMIC_ADD=1$" $p/environ 2>/dev/null; then echo "$(date -u +%T) ${p#/proc/} $(tr '\0' ' ' < $p/cmdline 2>/dev/null | cut -c1-120)"; fi; done; sleep 4; done ) > $LOGS/arm3_env_watch.log 2>&1 &
WATCH=$!
export SGLANG_MARLIN_USE_ATOMIC_ADD=1
twice sglang_qwen2.5_1.5b_gptq_marlin_guard_env_on_x6
unset SGLANG_MARLIN_USE_ATOMIC_ADD
kill $WATCH 2>/dev/null
log "processes that carried SGLANG_MARLIN_USE_ATOMIC_ADD=1 during arm 3 (count, cmdline):"
awk '{ $1=""; $2=""; print }' $LOGS/arm3_env_watch.log | sort | uniq -c | sort -rn | head -20

log "ARM 4 guard_off_det"
twice sglang_qwen2.5_1.5b_gptq_marlin_guard_off_det_x6 --extra-arg enable-deterministic-inference=true

log "pip freeze"
pip freeze > $LOGS/pip-freeze-ablation.txt
log "new result files:"
find results -type f -newer $LOGS/ablation_start_marker | sort
log "ABLATION_DONE"
