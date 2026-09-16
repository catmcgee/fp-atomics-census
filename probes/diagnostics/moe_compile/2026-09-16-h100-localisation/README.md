# Localising the compiled MoE symptom

This investigation continues [the controlled CUDA-build comparison](../2026-09-16-h100-builds/README.md)
for [vLLM issue 56900](https://github.com/vllm-project/vllm/issues/56900).
The pinned Qwen1.5-MoE model runs on a second H100 with vLLM 0.28.0,
PyTorch 2.13.0+cu130, Triton 3.7.1, V2, TP=1 and CUDA graphs disabled.
The recorded package list, all 16 model-file hashes, driver 580.126.09 and
VBIOS 96.00.DA.00.0C match the previous cu130 experiment. The GPU UUID differs.

## Measured boundaries

| Diagnostic | Observation |
|---|---|
| Uninstrumented compiled baseline | All 16 first tokens match the earlier compiled run. None of the eight duplicate prompt pairs agrees. |
| MoE boundary recorder | Recording preserves every compiled first token. Compiled/eager inputs to layer 0's MoE differ in all 962 packed rows. The recorded MoE call leaves its input unchanged. |
| Layer-0 attention recorder | Recording preserves every compiled first token. Q, K, V and the attention output match eager execution byte for byte. |
| Layer-0 post-attention operations | Output projection, residual input, norm weight and updated residual sum match exactly. RMSNorm output differs in 437,414 elements across all 962 rows; maximum absolute difference 0.03125. |
| Global precision-cast control | Enabling Inductor precision-cast emulation changes first tokens but leaves 0/8 duplicate pairs agreeing. |
| Matched opaque residual controls | Identity and BF16-rounding barriers each generate 32 tokens per prompt. Both leave 0/8 duplicate pairs agreeing, with 6/16 and 7/16 short-cycle outputs respectively. |
| Buffer reuse disabled | Both Inductor reuse settings resolve false and generated sources contain no reuse markers. All 512 generated tokens match the earlier compiled cu130 run exactly; 0/8 duplicate pairs agree and 9/16 outputs have short cycles. |
| Eager reference | All eight duplicate output pairs agree; only one of the 16 first tokens matches the compiled output. |

The earliest differing recorded operation is layer 0's post-attention RMSNorm.
The eager result matches a reference that rounds the residual sum to BF16 before
computing its variance; the compiled result is close to the unrounded FP32-sum
reference. This is a numerical localisation, not an established cause of the
model's degenerate text. Both precision emulation and the matched residual
rounding intervention leave the failure in place. The opaque identity barrier
itself changes three first tokens, so the rounding intervention must be compared
with that matched identity control rather than only with the original baseline.

Boundary scans and global precision emulation generate one token per prompt.
The paired residual controls generate 32, with the original greedy sampling and
ignored EOS. Neither experiment measures repeat stability across fresh processes.
The earlier controlled build experiment supplies that separate observation.

**Packing limitation:** the retained intermediate `duplicate_blocks` fields
split tensors using the original prompt lengths. Their `valid` flag checks
only the total row count; it does not establish the scheduler's packing order.
Do not interpret those fields as duplicate-request agreement. The output-token
pair comparisons use the returned request outputs and remain applicable.
The attention comparison checks entire tensor bytes across the two modes.

## Evidence and offline verification

The [MoE scan](moe-scan/) retains three immutable handoffs with 1,105 file
entries. The [attention scan](attention-layer0/) retains two with 690 entries.
Every original file was copied and SHA-verified before its exact ACK allowed
another GPU process to begin. Public bundles retain compressed raw JSON, logs,
executed source snapshots and textual compiler source/IR. Full binary caches
remain in the durable private archive; their original hashes are retained.

The [post-attention bundle](post-attention-op/) includes nine captured tensors
needed for its five direct comparisons. These comparisons reproduce from the
public data. The additional FP32 reference formula is sensitive to the CPU
PyTorch stack: its saved two-element difference recomputed as 14 elements on
the local ARM CPU. The audit reports both instead of treating that formula as
a portable bit-exact result. Unused large tensors retain their original hashes.

The auditor verifies exact public checksum coverage and original handoffs,
then reruns each incremental comparison with its frozen executed comparator.
It does not accept a saved verdict as proof of reproduction. Audit output paths
must be new and outside the bundles.

## Campaign status

The operation diagnostics are complete for this batch. The first recorded
numerical difference has been narrowed, but the failure has not been fixed.

## Provenance and lifecycle

[Provenance](provenance/) retains hardware, package lists, source hashes, model
file hashes and deletion evidence. The torch provenance objects have different
schemas: the current recorder omits several runtime fields. All eight shared torch fields, including the
distribution RECORD and import-file hashes, match. These sampled hashes and
matching package versions do not prove every installed file identical. No
full-stack identity is claimed. The first-token baseline was reproduced on this
different GPU UUID.

All completed arms were copied and SHA-verified before their ACK. The final
buffer-reuse arm contains 445 verified files, followed by 41 operational logs
and seven final provenance files. The pod was deleted at 19:11 UTC and absence
confirmed twice. Full caches remain in the durable private archive.

[Operational failures](operational-failures/) retain the first opaque-operator
registration failure (before model work), plus two independent routing setup
failures. The first MoE scan's ACK directory was missing; creating it allowed
the already-verified handoff to complete without rerunning inference.

The [CPU token audits](analysis/) recompute exact sequence comparisons and the
original short-cycle classifier (period at most three over the final 16 tokens).
The paired identity/rounding controls differ in 14 of 16 complete sequences,
with six and seven short-cycle outputs respectively. The no-reuse control
matches the original compiled run in all 512 tokens and all nine short cycles.
Requested top-five log probabilities were not retained by these targeted
controls, so no logprob identity is claimed.

To verify every localisation bundle, run the following from the repository root:

```sh
for bundle in moe-scan attention-layer0 post-attention-op precision-cast-control \
  opaque-identity-control opaque-rounded-control no-buffer-reuse-control; do
  .venv/bin/python probes/diagnostics/moe_compile/localise/public_audit.py \
    --audit-public "probes/diagnostics/moe_compile/2026-09-16-h100-localisation/$bundle" \
    --output "/tmp/$bundle-audit.json"
done
```

After verifying the bundles, audit the matched controls with
`public_audit.py --audit-control-pair IDENTITY_BUNDLE ROUNDED_BUNDLE --output NEW_PATH`.
This checks matched inputs, sampling and runtime settings before comparing outputs.

Use new output paths. `localise/control_token_audit.py LEFT RIGHT` independently
compares the compressed `result.json.gz` files and reports their source hashes,
sequence lengths, duplicate-pair agreement and short cycles.


`localise/norm_rounding_audit.py POST_ATTENTION_BUNDLE --output NEW_PATH`
recomputes the four CPU rounding formulas from the public captured tensors.
On the recorded local torch 2.8.0 ARM CPU, rounding the residual sum to BF16
before variance, without rounding the normalised value before multiplying the
weight, reproduces the eager tensor exactly. This is a stack-specific numerical
comparison, not proof of the GPU kernel's implementation or the failure's cause.
