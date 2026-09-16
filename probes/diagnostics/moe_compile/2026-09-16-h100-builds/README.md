# Same-H100 CUDA-build controls for vLLM #56900

**16 September 2026: the compiled MoE failure reproduces with both official
CUDA 12.9 and CUDA 13.0 build bundles on the same H100.** V2 generated tokens,
text and all saved top-five log probabilities are exactly equal across the two
builds, separately for compilation enabled and disabled. Switching to V1
changes the compiled output but does not remove the short-cycle and
duplicate-prompt failures.

This follows the [H20 non-reproduction](https://github.com/vllm-project/vllm/issues/56900#issuecomment-5697547861)
and our [proposed controls](https://github.com/vllm-project/vllm/issues/56900#issuecomment-5699822266).
The result narrows that discussion: the CUDA-build choice alone does not
explain the H100/H20 discrepancy. The GPU, driver and other differences between
those machines remain unresolved. No specific kernel fix is established.

## Results

All cells use Qwen/Qwen1.5-MoE-A2.7B-Chat revision
`ec052fda178e241c7c443468d2fa1db6618996be`, the original 16 raw prompts (eight
pairs), TP=1, bf16, seed 0, greedy 32-token generation, prefix caching disabled,
and CUDA graphs disabled. Compilation means resolved mode 3; the uncompiled
reference resolves to mode 0. Each process makes two identical generation
calls. Every completed cell is token-identical across its two calls.

| Build | Runner | Compilation | Independent processes | Outputs ending in short cycles | Matching duplicate pairs |
|---|---|---|---:|---:|---:|
| CUDA 12.9 | V2 | on | 2 | 9/16 in each | 0/8 |
| CUDA 12.9 | V2 | off | 1 | 0/16 | 8/8 |
| CUDA 13.0 | V2 | on | 2 | 9/16 in each | 0/8 |
| CUDA 13.0 | V2 | off | 1 | 0/16 | 8/8 |
| CUDA 13.0 | V1 | on | 1 | 5/16 | 0/8 |
| CUDA 13.0 | V1 | off | 1 | 0/16 | 8/8 |

A short cycle is a period of at most three tokens over the final 16 generated
tokens. This is a narrow output diagnostic, not a general accuracy benchmark.
Some uncompiled continuations repeat their already repetitive input in longer
phrases.

- **Within either V2 build:** compilation on versus off matches 0/16 complete
  token sequences. Fifteen prompts first diverge at token 0; the remaining
  prompt diverges at token 1. Divergence is not limited to near-tied first-token
  choices.
- **Across CUDA builds:** compiled matches compiled, and uncompiled matches
  uncompiled, in 16/16 complete token sequences, texts and full saved logprob
  payloads. Each fresh compiled repeat also matches its first process exactly
  on all three measures.
- **Across runners on CUDA 13.0:** V1 versus V2 compiled matches 0/16 token
  sequences. Uncompiled matches 16/16 token sequences; its log probabilities
  have small differences (maximum absolute candidate-logprob change
  `2.384185791015625e-6`). Do not conflate token identity with logprob identity.
- **Historical check:** current V2 compiled/uncompiled target 0 each match
  their corresponding retained E4 target in all 32 tokens and every saved
  top-five logprob value. E4 retained one target, with hook instrumentation.
  Separate raw output from the later original hook-free check was not found;
  this is not a direct 16-prompt comparison with that missing file. See
  [the source paths and hashes](historical_e4_output_comparison.json).

## Controls and scope

All current processes used GPU UUID
`GPU-fcd67bc9-4348-93cc-aba6-22a2076f2fdc`, NVIDIA H100 80GB HBM3,
driver `580.126.09`, VBIOS `96.00.DA.00.0C`, and Python 3.12.3. The model-file
hashes, exact prompts and token IDs are retained. The prompt-token batch SHA-256
is `348746304bca8894b76318c9b9206970102bf30f9e6ff163d75385cfa322949a`;
the raw prompt-text batch SHA-256 is
`573b7902fcbc93c2adeccc88deb0fa5d5a6611693153e6e6018a3b48fa00e912`.

The resolved selections were FlashAttention 3, the TRITON unquantised MoE
backend, `TritonExperts`, and `MoEPrepareAndFinalizeNoDPEPModular`. Runner,
compilation and graph settings are asserted after construction. Matching
backend names do not prove that every generated kernel is identical.

The official vLLM 0.28.0/CUDA 13.0 and 0.28.0+cu129 wheels were installed in
separate environments with their matching torch 2.13.0 build. Both use Triton
3.7.1, FlashInfer 0.6.16.post3, transformers 5.17.0 and tokenspeed-triton
3.8.10.post20260906. Both environments passed `pip check`. Full package lists,
wheel URLs and hashes, imported-module provenance and distribution RECORD
hashes are retained. The bundle comparison shares 169 normalised package
entries and differs in 44 entries; it is not a single-package intervention or
proof of complete identity with the historical installation.

Each worker starts in a fresh process with distinct, initially empty
`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` directories,
`VLLM_DISABLE_COMPILE_CACHE=1`, `VLLM_PLUGINS=""`, and
`TORCHINDUCTOR_COMPILE_THREADS=1`. This proves emptiness of those three roots
only. Other library and driver caches were not independently cleared.

These are eight successful worker processes on one GPU. They do not establish
behaviour on H20, another driver, other models, or vLLM 0.29.0. The uncompiled
reference is internal to vLLM, not an external correctness oracle.

## Initial failed setup and debug instrumentation

Two initial CUDA 13.0 workers failed before producing model output and are
retained separately under `cu130_v2_graphs_off_debugdump_failed/`:

1. With `VLLM_DEBUG_DUMP_PATH` enabled, depyf 0.20.0's
   `patched_load_by_key_path` rejected torch's `set_sys_modules` keyword during
   compiler autotuning. This is an instrumentation failure, not an output
   observation.
2. The uncompiled worker could not execute `ninja` because the launch command
   omitted the virtual environment's `bin` directory from `PATH`. The package
   was installed; the subsequent launch corrected `PATH`.

All successful cells retain `VLLM_LOGGING_LEVEL=DEBUG` and omit the incompatible
depyf dump instrumentation. The harness now exposes it as opt-in
`--debug-dump`. Native generated compiler sources and vLLM computation graphs
remain in the source archives; they are not depyf/FX debug dumps.

## Evidence and reproduction

- [Joint audit](moe_compile_final.audit.md) and
  [machine-readable audit](moe_compile_final.audit.json).
- Raw per-cell results, prompt IDs, logprobs, resolved settings, provenance,
  cache-before/after manifests and compressed stdout/stderr in each run folder.
- `*.compiler-sources.tar.gz`: generated source and textual compiler artefacts.
  Opaque binaries are omitted from this public bundle. Complete copies of the
  three named cache trees were SHA-verified into durable local project storage
  before subsequent runs.
- `*.verified.json`: hashes of the complete transferred source trees, including
  files intentionally omitted from this public bundle. `SHA256SUMS` covers the
  public bundle itself.
- `setup/`: pinned bootstrap, package freezes, hardware and model-file hashes.
- `source/`: harness, auditor and tests used for this investigation.

Use `audit_command.sh` to reproduce the audit from the retained, compressed raw
files. No GPU is required. Initial setup failures are audited separately and
are excluded from the completed output comparisons.
