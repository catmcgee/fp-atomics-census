# Graph-safe actual MoE routing telemetry

This standalone diagnostic uses the pinned vLLM `0.29.0` source at
`98dff2a81d747d1dba01a47f939f48c3526d4206`. It does not modify the frozen
shape hook or any archived record.

`--enable-return-routed-experts` is the instrumentation switch. In this source
it binds every supported MoE router to `RoutedExpertsCapturer`: the router copies
its selected `topk_ids` into a preallocated device tensor during the forward.
The worker transfers that tensor only after the forward. This captures actual
selected IDs, supports static CUDA-graph replay, and does not reconstruct IDs
from router logits.

## GPU command

Transfer this causal-followup directory outside the clean frozen checkout and
use new output paths on a compatible single-H100 pod. The harness refuses a
pre-existing hook directory, so a later run cannot be mistaken for this one.

```bash
export HF_HOME=/root/hf HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0
PREP=/root/causal_followup
SOURCE=/root/c2-source
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DATA="/root/causal/routing-$STAMP"
mkdir -p "$DATA/acks"
set -o pipefail
python "$PREP/routing_harness.py" \
  --source-root "$SOURCE" \
  --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --revision ec052fda178e241c7c443468d2fa1db6618996be \
  --shape-hook-root "$DATA/hook" \
  --output "$DATA/report.json" \
  --require-ack \
  2>&1 | tee "$DATA/controller.log"
```

The harness runs the same two prompts in three fresh subprocesses: graph
without telemetry, graph with telemetry, then matched eager telemetry. All
three use compilation mode 0, `enforce_eager=False`, seed 0, disabled prefix
caching, and a 2048-token model limit. The graph conditions request CUDA graph
mode `FULL`; the eager comparison requests mode `NONE`. Fixed-length decoding
ignores EOS so every request exercises several decode steps.

It passes only when graph telemetry preserves generated token IDs and hidden/
logit hashes, graph routing matches eager routing, and the native selected IDs
change across decode steps within each request. It also requires the two
dynamic prompts to differ in output-token hash, selected-ID hash, and per-layer
expert counts. It trims static-capture padding before all counts and digests.

Before importing vLLM, the harness requires a clean frozen census checkout,
an installed vLLM 0.29.0 wheel whose native capture/router/runner/scheduler/
output files match the reviewed commit byte for byte, and an installed frozen
`shape_hook` that resolves into the census checkout. A vLLM source checkout is
not required on the runtime host.

The required shape-hook directory is an execution witness, not a routing
source. Its JSONL records retain the resolved compilation and CUDA-graph modes,
the actual dispatch mode and padding, consumed token IDs, and hidden/logit
hashes. Each condition JSON is atomically updated after engine creation and
after every sample; subprocess and aggregate failures get separate failure
JSON. The report fails unless graph telemetry has a graph dispatch, eager
telemetry has no graph dispatch, and graph control and graph telemetry have
identical hook input/hidden/logit rows. Therefore a capture-time tensor or a
stale static buffer cannot pass merely because `requested_graph_mode` is true.
Keep the JSON, the three hook subdirectories, and the engine log together.

With `--require-ack`, each successful condition publishes an immutable
manifest under `handoff/report_<condition>/manifest.json`. Its files include
the condition JSON and log, that condition's hook evidence, and the retained
harness, validator, and frozen shape-hook source snapshots. The controller
waits up to 180 seconds for an exact four-field ACK at
`acks/report_<condition>.json` before starting the next engine. After all
three ACKs and aggregation, it writes `COMPLETE.json`; an invalid scientific
result is still marked complete with `valid: false` and returns status 1.

Before running, verify the installed wheel exposes both
`LLM(enable_return_routed_experts=...)` and `CompletionOutput.routed_experts`.
The feature is incompatible with pipeline parallelism and some KV-transfer
configurations in the pinned source; use TP=1 and no KV connector. The harness
constructs a new process and engine for each condition. It also requires the frozen 0.29
shape-hook package to be installed before the first `vllm` import.

## CPU-only archive audit

After copying the completed data root, rerun every integrity and scientific
check without a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 python routing_audit.py /path/to/routing-data-root
```

The auditor verifies the three canonical handoff manifests and exact ACKs,
rehashes every referenced file, rebuilds hook evidence from raw JSONL, and
recomputes every expert count and selected-ID digest from the retained IDs. It
then reruns the graph/eager comparisons, decode-step stale-buffer checks,
resolved dispatch checks, and source-byte attestations. It does not accept the
stored aggregate `valid` field as evidence. Exit status 0 means the archive and
scientific checks both pass; status 1 means integrity or recomputation failed;
status 2 means the archive is internally valid but its recomputed scientific
result is false.
