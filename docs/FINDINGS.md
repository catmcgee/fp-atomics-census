# Findings notes

## What the experiments establish

Bit-exact replay needs more than a model revision, seed and recorded request
schedule. The experiments retain examples where those fields agree but output
hashes differ. They also retain matching observations under specific controls.
A matching sample establishes what happened in those runs, not a deployment-wide
guarantee.

| Question | Measured result | Scope and interpretation |
|---|---|---|
| Can an atomic reduction change inference output? | The SGLang Marlin guard intervention changed token-level and cross-process outcomes on one fixed stack. | This is an isolated source-path intervention. Other differences with the atomic path disabled remain unexplained. |
| Does a fresh compile always replay a fixed history exactly? | C1's bf16 reference matched in 10 of 16 cold replays; reduction configurations tracked the output classes. | Strong association on the tested configuration; no single-choice compiler intervention yet establishes the cause. |
| Is FP8 exempt from cold replay differences? | C2 per-token FP8 matched 1/3 cold replays and blockwise FP8 2/3. | The earlier all-matching FP8 observations do not generalise. |
| Which compiler settings help the mixed schedule? | A fresh-reference eight-setting experiment found cold differences in all four determinism-off settings. Each of the three runnable determinism-on settings matched 3/3 cold replays. | Combo-off alone was insufficient. Determinism plus both combo options on failed at startup; this is not an IDENTICAL result. Small repeat counts remain bounded observations. |
| Do equal coarse backend names prove equal implementations? | The TP=2 record initialised `mnnvl`; all three cold replays fell back to `trtllm`. | Backend names and package equality are insufficient. These differing outputs do not isolate a cold-compile effect. |
| Does the compiled MoE symptom depend on the CUDA build? | vLLM 0.28.0 cu129 and cu130 reproduced exactly the same V2 failure on one H100, including saved top-five log probabilities. | Build choice did not remove this symptom. V1 also failed differently. The kernel cause and the contributor's H20 non-reproduction remain unresolved. |
| Where does compiled MoE first differ numerically? | Layer-0 attention and output projection match eager, but post-attention RMSNorm differs across all 962 packed rows. Precision emulation and matched residual-rounding controls retain the output failure. | This narrows a numerical difference; it does not establish the cause of the degenerate model output. |
| Does the compiled MoE symptom require vLLM's custom compile backend? | Mode 2 (`DYNAMO_TRACE_ONCE`) with plain Inductor reproduced 0/8 agreeing duplicate pairs and 8/16 short cycles; mode 0 had 8/8 and 0/16. | The custom `VllmBackend`, its passes and splitting are unnecessary. Mode 2 still uses vLLM's compile wrapper and traced model/operator code. Mode 1 is unsupported on V2, so attribution to stock `torch.compile` is not established. |
| Can actual MoE routing be observed inside CUDA graphs? | Native selected-expert capture matched eager routing, changed across seven decode steps per request, and preserved tokens and tensor hashes when enabled. | Validated on two prompts with compilation mode 0, FULL graphs and synchronous scheduling. Historical null Python-hook fields remain unobserved. |

The [README](../README.md) retains the inventory and complete runtime context.
Primary campaign records and offline reproduction commands are in
[C1](../probes/shape/campaigns/2026-09-15-h100-e6-attribution.md),
[C2](../probes/shape/campaigns/2026-09-15-h100-e6-modern.md),
[the C2 completion](../probes/shape/campaigns/2026-09-16-h100-e6-followup.md),
[the setting and routing follow-up](../probes/shape/campaigns/2026-09-16-h100-causal-followup.md),
[the controlled MoE investigation](../probes/diagnostics/moe_compile/2026-09-16-h100-builds/README.md), and
[its operation-level localisation](../probes/diagnostics/moe_compile/2026-09-16-h100-localisation/README.md), and
[the compilation-mode discriminator](../probes/diagnostics/moe_compile/2026-09-16-h100-compile-modes/README.md).

## What a replay claim must state

A useful claim identifies the reference, exact model files, runtime and source,
hardware and driver, request arrivals and consumed tokens, compilation and cache
state, graph execution, and the compared output fields. When collective or
attention implementations can change under the same broad name, their actual
selection must be recorded too. Rank-0 agreement does not establish agreement of
all internal rank states.

Teacher forcing holds the consumed continuation fixed so that one numerical
change does not cascade into a different input sequence. Hash differences then
measure differing tensor bits under those inputs. They do not by themselves
establish a meaningful answer change, an exploit, or an incorrect mathematical
result. Free-running argmax and output text provide separate diagnostics.

The census classifies potential order-dependent source paths; it does not prove
that every listed path runs in a deployment. Runtime measurements establish
bounded observations; they do not constitute an allowlist. The remaining
protocol argument below concerns a proposed verifier, not a measured attack or
a proved security property.


## Distinguishable batch histories

Suppose an operator can choose among K feasible **complete shape histories**, and those choices induce D distinct output trajectories for a fixed target. Then D ≤ K and the choice can encode at most log2(D) ≤ log2(K) bits per trajectory, assuming all choices can be deliberately selected and distinguished. This is an upper bound on a choice set, not an experimentally established channel capacity. Many histories may produce the same output, and scheduling constraints may prevent choices from being realised reliably.

A T-step bound of sum over t of log2(K_t), or T log2(K) when each step has K choices, requires a different assumption: independently realisable, distinguishable choices at each step. It cannot be obtained by multiplying the number of complete histories by the number of tokens. The current experiments do not establish those independence, feasibility or distinguishability conditions.

## Reproduction and scheduler freedom

Accepting any output in a set of admissible histories can leave freedom to an operator who controls scheduling. Recording or committing a history and checking only that history may bind a later replay to an earlier execution. It does not remove the information encoded by the operator's original choice of history.

Reducing that freedom would require an independently prescribed schedule, trusted or verifiably constrained scheduling, or a demonstrated batch-invariant output for the relevant deployment. Each option needs an explicit threat model, commitment timing, target-state authentication and treatment of model, sampler, cache and neighbour dependencies. A deterministic kernel alone does not provide these protocol properties. None is proved by the present census or repeated-run observations.
