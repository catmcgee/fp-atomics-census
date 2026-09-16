#!/usr/bin/env bash
# Prefetch the pinned model revisions C2 needs. $1 = "tp1" (all three TP=1 models) or "one" (Qwen2.5-7B only).
# HF_TOKEN arrives in the environment of the launching shell; never written to a file or echoed.
set -x
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1 PIP_CACHE_DIR=/root/pip-cache
mkdir -p /root/hf /root/logs
date -u
python3 -m venv /root/hfenv && /root/hfenv/bin/pip install -q huggingface_hub hf_transfer 2>&1 | tail -2
WHICH=${1:-one} /root/hfenv/bin/python - <<'PY'
import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
print("HF_TOKEN present:", bool(os.environ.get("HF_TOKEN")), flush=True)
repos = [("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")]
if os.environ.get("WHICH") == "tp1":
    repos.append(("Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3"))
    repos.append(("Qwen/Qwen1.5-MoE-A2.7B-Chat", "ec052fda178e241c7c443468d2fa1db6618996be"))
def get(rr):
    repo, rev = rr
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken", "*.jinja"], ignore_patterns=["original/*"])
    print("PREFETCHED", repo, rev, p, flush=True)
list(ThreadPoolExecutor(3).map(get, repos))
PY
date -u
echo "PREFETCH_DOWNLOADED"
/root/hfenv/bin/python - <<'PY'
import hashlib, json, os
from concurrent.futures import ThreadPoolExecutor
base = "/root/hf/hub"
jobs = []
for m in sorted(os.listdir(base)):
    if not m.startswith("models--"): continue
    snaps = os.path.join(base, m, "snapshots")
    for rev in os.listdir(snaps):
        for root, _, files in os.walk(os.path.join(snaps, rev)):
            for f in files:
                p = os.path.join(root, f)
                jobs.append((m[8:].replace("--", "/", 1), rev, os.path.relpath(p, os.path.join(snaps, rev)), p))
def h(j):
    repo, rev, rel, p = j
    real = os.path.realpath(p); size = os.path.getsize(real)
    s256 = hashlib.sha256(); s1 = hashlib.sha1(b"blob %d\0" % size)
    with open(real, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            s256.update(chunk); s1.update(chunk)
    return {"repo": repo, "revision": rev, "file": rel, "size": size, "sha256": s256.hexdigest(), "git_blob_sha1": s1.hexdigest()}
res = list(ThreadPoolExecutor(16).map(h, jobs))
json.dump(sorted(res, key=lambda r: (r["repo"], r["file"])), open("/root/logs/weights_digests.json", "w"), indent=1)
print("HASHED", len(res))
PY
du -sh /root/hf/hub/models--* 2>/dev/null; df -h /root | tail -1; date -u
echo "PREFETCH_DONE"
