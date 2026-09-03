# fp-atomics-census

A census of floating-point atomic operations in the GPU kernel sources of
the main open-source LLM inference engines, and what that census implies
for bit-exact verification of AI inference.

Eight repositories are pinned in `scan-manifest.json` and scanned at those
commits: vLLM, SGLang, FlashInfer, FlashAttention, Marlin, DeepGEMM, DeepEP
and the split-K and stream-K reduction paths of CUTLASS. Every atomic site
found is classified by reading the surrounding code, and the result is a
machine-readable inventory, a per-engine list of configurations that stay
clear of order-dependent atomics, and a short write-up.

## Why this exists

Verifying that a datacenter ran the workload it claims to have run, whether
for AI governance, treaty verification or auditing an API provider, is
simplest when inference is bit-exactly reproducible: the verifier re-runs a
sampled computation and compares hashes. Cankaya (2026) argues that modern
inference engines are already deterministic once the configuration is
recorded, and that the only genuine run-to-run non-determinism comes from
floating-point atomic accumulation. He found one such kernel (the exllama
path in vLLM's GPTQ `q_gemm.cu`) and noted the atomics in
FlashAttention-3's backward pass.

That claim was made from a handful of models and configurations. Nobody
had checked it systematically across the ecosystem. If it holds, verifiers
can demand bit-exact matches instead of tolerance schemes such as DiFR,
which leave an adversary slack for steganography and unreported
computation. If it does not hold, the census says which configurations a
verification regime has to forbid or gate.

The short answer, argued in `docs/FINDINGS.md`: the premise holds for the
core dense forward pass (attention, unquantised and most quantised linear
layers, the Triton and Marlin MoE kernels, DeepGEMM, DeepEP), but not for
the ecosystem as a whole. Order-dependent atomics sit on default paths in
LoRA, in several MoE finalize kernels, in one quantised MoE kernel, in
SGLang's Marlin and CUTLASS FP8 configurations, in some sampling
renormalisation kernels, and in PyTorch library operators used for
pooling. A verifier can forbid every one of them with the allowlists here,
and is then left with the closed binaries (cuBLASLt, cuDNN, NCCL, TRT-LLM
cubins) that only a runtime probe can settle.

## Layout

```
scan-manifest.json   pinned repo shas and in-scope directories
scan/                candidate scanner (regex plus tree-sitter for C++, ast for Python)
candidates/          raw scanner output, one jsonl and one coverage json per engine
triage/              inventory JSON Schema, checklist, validation and export tools
inventory/           triaged sites, one jsonl per engine; inventory.csv at the root
allowlist/           per-engine deterministic configuration allowlists
docs/                TAXONOMY, BACKGROUND, METHOD, FINDINGS, RUNTIME_PROBES
probes/              GPU scripts that settle the class-C and low-confidence rows
tests/               scanner unit tests on synthetic fixtures
```

## How to run

```
make venv                     # Python 3.11 venv with jsonschema, tree-sitter, pytest
make clone                    # fetch every repo at its pinned sha into repos/
make test                     # scanner unit tests on tests/fixtures
make census REPO=vllm         # re-run the scanner on one repo (SHA=... to override)
make census-all               # re-run on all eight
make validate                 # check inventory/*.jsonl against the schema and the checkouts
make csv                      # regenerate inventory.csv
python -m triage.coverage_table candidates/*.coverage.json   # coverage tables
```

`make census` refuses to run on a checkout whose HEAD differs from the
manifest, so the candidates are reproducible from the manifest alone.
Re-running the scanner regenerates `candidates/`; the inventory is a
human product and is not regenerated.

## How to read the inventory

Each line of `inventory/<engine>.jsonl` is one site. The fields are
defined in `docs/TAXONOMY.md` and enforced by `triage/inventory.schema.json`.
The ones that matter most:

- `class`: `A` (order-dependent float accumulation, no mitigation found),
  `A1` (no contention), `A2` (serialised), `A3` (gated by a flag),
  `B` (exact atomic), `B-indirect` (exact atomic that decides a slot or
  order for later float data), `C` (opaque binary).
- `default_path`: whether a stock configuration reaches the site.
- `gate`: for A3, the flag, where it is read, and its default at the sha.
- `downstream`: for B-indirect, whether a different slot order can change
  a floating-point reduction order, with the argument.
- `evidence`: `file:line` references that justify the class.
- `confidence` and `notes`: `low` means the class is a best reading, not a
  settled fact; the notes say what is missing.

`candidates/` keeps everything the scanner matched, including matches in
comments and strings (with `excluded_reason`), so the triage can be
audited: every inventory row cites candidate ids or a file:line that
appears in the candidates.

Class-A rows with `default_path: true` are the findings. `A3` rows on the
default path are findings when the gate's default is the atomic side; the
allowlists spell out which flag to set.

## Limits of static analysis

- The scanner finds sites; it cannot decide contention. Every A1/A2/A3
  sub-case was decided by reading, and a wrong reading is possible. Rows
  with `confidence: low` say so.
- Binaries are opaque. cuBLAS/cuBLASLt (every unquantised linear layer),
  cuDNN, NCCL, NVSHMEM and the TensorRT-LLM kernels that FlashInfer
  downloads as cubins are class C. The census records where they are
  called and which knobs change their reduction strategy; `probes/` has
  the experiments that would settle them.
- Runtime-generated kernels are out of scope: `torch.compile` output
  (on by default in vLLM), Triton autotune choices, and JIT variants are
  seen only through their templates.
- Batch invariance is a different property. A kernel can be
  run-to-run deterministic and still give different results for the same
  request at different batch sizes. This census is about run-to-run
  determinism; Cankaya 2026 and the engines' batch-invariance modes cover
  the other property.
- The pinned shas are from 6 July 2026. Kernels move; the inventory line
  numbers are valid only at those shas.

## Citations

- N. Cankaya, "Bit-Exact AI Inference Verification Without Performance
  Tradeoffs", arXiv:2606.00279, 2026. https://arxiv.org/abs/2606.00279
- "DiFR: Inference Verification Despite Nondeterminism", arXiv:2511.20621,
  2025. https://arxiv.org/abs/2511.20621
- H. He and Thinking Machines Lab, "Defeating Nondeterminism in LLM
  Inference", September 2025.
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- SGLang team, "Towards Deterministic Inference in SGLang and Reproducible
  RL Training", September 2025.
  https://lmsys.org/blog/2025-09-22-sglang-deterministic/
- PyTorch, `torch.use_deterministic_algorithms` reference.
  https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html

`docs/BACKGROUND.md` summarises what each of these claims.

## Licence

MIT. See LICENSE.
