# MoE compile boundary localiser

This harness narrows the vLLM 0.28.0 Qwen1.5-MoE compiled-output failure at
the opaque MoE custom-op boundaries. It does not alter retained evidence.

The recorder is installed after engine construction on each instantiated
`MoERunner._forward_impl`. That method already runs behind
`torch.ops.vllm.moe_forward_shared`; recording therefore does not add a new
Dynamo/Inductor graph operation. The default uninstrumented compiled cell checks
whether the recorder preserves the observed generated tokens.

Run the first scan in the pinned CUDA 13.0 environment on one H100:

```sh
CUDA_VISIBLE_DEVICES=0 /opt/issue56900/cu130/bin/python \
  probes/diagnostics/moe_compile/localise/moe_boundary_localise.py \
  --output-dir /project/moe-localise-scan --require-ack
```

The controller creates three fresh processes. Each cell gets its own directory
containing distinct `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, and
`TRITON_CACHE_DIR` roots (nine named roots across a complete scan):
uninstrumented compiled, recorded compiled, and recorded eager. It uses the
original 16 raw prompts, V2, TP=1, graph mode `NONE`, and one generated token.
`comparison.json` reports output preservation, duplicate-pair behaviour and the
first compiled/eager layer/stage digest difference.

Every recorded call must contain MoE input, router logits, top-k weights and
IDs, both shared and routed outputs, and the post-call input. Missing stages,
duplicate event keys, a non-V2 resolved runner, or any failed cell stop the
controller before another cell starts. Each worker retains package `RECORD`
hashes, import-file hashes, direct wheel metadata, and `pip freeze` output in
`provenance.json`.

`--require-ack` is the paid-run mode. After each successful cell it emits
`HANDOFF_READY`, having written
`handoff/CELL_NAME_WITH_UNDERSCORES/manifest.json`. The manifest hashes the
cell directory (including all three caches), stdout/stderr, immutable copies of
the controller manifest and comparison at that point, and an exact copy of
this script. Before starting the next process, the controller waits up to 180
seconds for
`acks/CELL_NAME_WITH_UNDERSCORES.json` containing exactly:

```json
{
  "schema": 1,
  "label": "compiled_baseline",
  "handoff_manifest_sha256": "<manifest sha256>",
  "verified_manifest_sha256": "<manifest sha256>"
}
```

After each ACK, and later without a GPU, rehash every past manifest and validate
the exact acknowledgements with:

```sh
/opt/issue56900/cu130/bin/python \
  probes/diagnostics/moe_compile/localise/moe_boundary_localise.py \
  --output-dir /project/moe-localise-scan --audit-only
```

After the scan identifies a boundary, retain raw bf16 tensors only around that
layer in a second run:

```sh
CUDA_VISIBLE_DEVICES=0 /opt/issue56900/cu130/bin/python \
  probes/diagnostics/moe_compile/localise/moe_boundary_localise.py \
  --output-dir /project/moe-localise-layer-N \
  --layers N --save-tensors --no-include-baseline
```

If recording changes the compiled first token, or the boundary traces bracket
the difference after an opaque MoE call, one prospective control disables
Inductor's input/output in-place reuse and unrelated buffer reuse before model
construction:

```sh
CUDA_VISIBLE_DEVICES=0 /opt/issue56900/cu130/bin/python \
  probes/diagnostics/moe_compile/localise/inductor_reuse_control.py \
  --output-dir /project/moe-reuse-off --require-ack
```

This is one compiled-only process with the same 16 prompts and one token. It
fail-closes unless generated Inductor Python contains `moe_forward_shared` and
contains no `# reuse` markers. It records the two resolved Inductor switches,
the generated-source hashes, and timestamped `nvidia-smi` output. Run it only
after the baseline scan makes storage reuse the relevant discriminator.

When the first MoE input already differs, probe layer 0 attention instead:

```sh
CUDA_VISIBLE_DEVICES=0 /opt/issue56900/cu130/bin/python \
  probes/diagnostics/moe_compile/localise/attention_boundary_localise.py \
  --output-dir /project/attention-layer0 --require-ack
```

This runs recorded compiled and eager cells. After model construction it wraps
the instantiated layer-0 attention implementation called from inside the
existing opaque `unified_attention_with_output` op. It records Q, K, V and the
attention output, requires exactly one complete call, and checks the compiled
tokens against the already retained uninstrumented first-token vector. If Q/K/V
first differ, the boundary moves to embedding, input normalisation, QKV
projection, or rotary embedding. Equal Q/K/V with a differing output selects
the attention implementation. Equal attention output with a differing first
MoE input brackets the next probe to O projection and fused add/RMS norm.

The retained attention run found Q, K, V, and attention output exactly equal
between compiled and eager execution. The next targeted probe wraps the
already-loaded generated Inductor partition and the eager layer-0
post-attention normalisation boundary:

```sh
CUDA_VISIBLE_DEVICES=0 /opt/issue56900/cu130/bin/python \
  probes/diagnostics/moe_compile/localise/post_attention_operator_localise.py \
  --output-dir /project/post-attention-operator --require-ack
```

The compiled cell captures the real attention output, O-projection matrix and
output, residual, normalisation weight, and fused RMS-normalisation output.
The eager cell captures the corresponding O-projection output, residual,
weight, and normalisation output. It compares these stages directly and
checks that instrumentation preserves the retained compiled first-token
vector. This brackets a difference among O projection, the layer-0 residual,
and post-attention RMS normalisation without treating duplicate blocks as a
request-order witness.

The trace captures exact bytes and per-row hashes for MoE input, router logits,
top-k weights/IDs, input after the call, and shared/routed outputs. A distinct
shared input is captured separately; when it is the same view as the routed
input, alias metadata records that fact without copying the tensor twice.
For this targeted pair, `comparison.json` also reports differing-element counts,
maximum and mean absolute error, and the L2 delta for every matched tensor.
Duplicate-block reports are structurally formed only when the first-pass
packed row count equals the sum of all recorded prompt lengths. That equality
does not attest scheduler row order. Semantic interpretation of those blocks
requires a separate ordering witness; the current boundary conclusion instead
uses the exact full-tensor compiled/eager attention comparison.

Limits:

- vLLM eager output is an internal reference.
- A clone used to preserve each pre-call tensor can perturb allocation and
  timing. The uninstrumented cell detects token-level changes but cannot prove
  that timing is irrelevant.
- Boundary equality localizes a difference between recorded stages; it does
  not by itself identify a kernel cause.
- Use a new output directory for every invocation. The script refuses to
  overwrite prior data.
- One-token cells exercise prompt prefill and the first sampled logit. Retained
  CUDA 13.0 [compiled output](../2026-09-16-h100-builds/cu130_v2_graphs_off_nodump/compile-on_graphs-off_runner-v2/result.json.gz)
  and [eager output](../2026-09-16-h100-builds/cu130_v2_graphs_off_nodump/compile-off_graphs-off_runner-v2/result.json.gz)
  show that this is sufficient for the 0/8 compiled versus 8/8 eager
  duplicate-pair discriminator: 15 of 16 first tokens differ between the two
  modes. It does not observe decode-time shapes, the remaining prompt whose
  mismatch starts at token 1, short cycles, or repeat stability.

The retained graph statement can be regenerated without a GPU:

```sh
python3 probes/diagnostics/moe_compile/localise/retained_graph_audit.py \
  probes/diagnostics/moe_compile/2026-09-16-h100-builds/cu129_v2_graphs_off_nodump.compiler-sources.tar.gz \
  probes/diagnostics/moe_compile/2026-09-16-h100-builds/cu130_v2_graphs_off_nodump.compiler-sources.tar.gz
```

## Measured localization result

The 2026-09-16 H100 run preserved the compiled first-token vector under both
the MoE and attention recorders. Layer-0 Q, K, V, attention output,
O-projection output, residual, and normalisation weight were byte-identical
between compiled and eager execution. The first measured difference was the
post-attention RMS-normalisation output: all 962 rows differed (437,414 bf16
elements, maximum absolute difference 0.03125).

Replay of the captured inputs isolated the numerical rule. Eager output was
byte-exact when the fp32 residual sum was rounded to bf16 before the RMS
variance calculation. The generated compiled kernel normalized the unrounded
fp32 sum. This difference is causally active but was not sufficient to explain
the bad-output phenotype:

- Enabling Inductor's eager-numerics precision-cast control retained 0/8
  duplicate prompt pairs.
- Matched opaque-barrier controls, differing only in whether the residual sum
  was rounded through bf16, produced different token sequences but both
  retained 0/8 duplicate pairs and repetitive output. The original cycle
  classifier marked 6/16 identity-arm outputs and 7/16 rounded-arm outputs as
  period-1-to-3 cycles in their last 16 tokens.
- Disabling Inductor input/output in-place buffers and all unrelated buffer
  reuse produced exactly the same 16-by-32 token matrix as the retained
  default CUDA 13.0 compiled run. Both canonical token matrices hash to
  `1378169337ac946fea2c771aba7108af3d7e844b06793de234c1e873eb1b32e6`,
  including the same 9/16 short-cycle classification.

These controls rule out residual-sum rounding and Inductor buffer reuse as
complete explanations. They do not identify the remaining cause. The matched
barrier itself changed three first tokens relative to the uninstrumented
baseline, so only the rounded-versus-identity comparison supports a causal
statement about rounding. The buffer-reuse control retained token IDs but did
not retain logprobs, so its exact-identity claim is limited to decoded tokens.
The retained comparison can be regenerated on CPU with
`control_token_audit.py LEFT_RESULT RIGHT_RESULT`.
