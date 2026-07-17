# fp-atomics-census

A census of floating-point atomic operations in the GPU kernel sources of the
main open-source LLM inference engines, and what that census implies for
bit-exact verification of AI inference.

Status: work in progress. See the plan at the bottom of this file.

## Why this exists

Verifying that a datacenter ran the workload it claims to have run, whether for
AI governance, treaty verification or auditing an API provider, is simplest when
inference is bit-exactly reproducible. The verifier re-runs a sampled
computation and compares hashes. Cankaya (2026) argues that modern inference
engines are already deterministic once the configuration is recorded, and that
the only genuine run-to-run non-determinism comes from floating-point atomic
accumulation. He found one such kernel (the exllama path in vLLM's GPTQ
`q_gemm.cu`) and noted the atomics in FlashAttention-3's backward pass.

That claim was made from a handful of models and configurations. Nobody has
checked it systematically across the ecosystem. If it holds, verifiers can
demand bit-exact matches instead of tolerance schemes such as DiFR, which leave
an adversary slack for steganography and unreported computation. If it does not
hold, the census says which configurations a verification regime has to forbid
or gate.

The deliverable is a machine-readable inventory of every atomic site in the
scanned engines, a per-engine list of configurations that stay clear of
order-dependent floating-point atomics, and a short write-up.

## Layout

```
scan-manifest.json   pinned repo shas and in-scope directories
scan/                candidate scanner (regex plus tree-sitter for C++)
candidates/          raw scanner output, one jsonl per engine (not triaged)
triage/              inventory JSON Schema and the classification checklist
inventory/           triaged sites, one jsonl per engine, plus inventory.csv
allowlist/           per-engine deterministic configuration allowlists
docs/                taxonomy, background, method, findings, runtime probes
probes/              GPU scripts that settle the class-C and low-confidence rows
tests/               scanner unit tests on synthetic fixtures
```

## How to run

To be written once the scanner lands.

## How to read the inventory

To be written once the schema lands.

## Limits of static analysis

To be written.

## Citations

- N. Cankaya, "Bit-Exact AI Inference Verification Without Performance
  Tradeoffs", arXiv:2606.00279, 2026. https://arxiv.org/abs/2606.00279
- "DiFR: Inference Verification Despite Nondeterminism", arXiv:2511.20621,
  2025. https://arxiv.org/abs/2511.20621
- H. He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference",
  September 2025.
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

See docs/BACKGROUND.md for what each of these claims.

## Plan

1. Scaffold, licence, manifest, README skeleton, background reading. (this commit)
2. Taxonomy and inventory schema, with fixtures for each class.
3. Scanner for CUDA and PTX; tests; run on flash-attention and marlin.
4. Triage flash-attention and marlin; first allowlists.
5. Scanner support for Triton and torch patterns; run on vLLM; triage.
6. SGLang and FlashInfer, including the pre-compiled cubin surface.
7. DeepGEMM, DeepEP, CUTLASS reduction modes; B-indirect analysis.
8. Findings, runtime probes, final README pass, inventory.csv.

## Licence

MIT. See LICENSE.
