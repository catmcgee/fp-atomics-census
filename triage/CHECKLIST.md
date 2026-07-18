# Triage checklist

Every candidate in `candidates/<engine>.jsonl` is either turned into a row in
`inventory/<engine>.jsonl` or left in place with an `excluded_reason`. Work
through the steps in order and write down the `file:line` for each answer;
the row's `evidence` array is those answers.

## 1. Is it code?

- The match is not inside a comment, a string literal, or a `#if 0` block.
  The scanner marks obvious cases (`in_comment`, `in_string`), but check.
- The match is device code, not a host `std::atomic`, and not a name that
  merely contains the pattern (`atomicAddFloat` wrapper definitions count;
  `non_atomic_add` does not).
- If it is a wrapper definition (`atomicMaxFloat`, `atomic_add_bf16x2`), the
  row is the definition site and `notes` lists the callers. Callers are not
  separate rows unless they differ in gating or contention.

## 2. What exactly is the operation?

- Primitive: `kind` field. For `atomicCAS`, decide loop-implementing-add
  (`atomicCAS-loop`) versus lock/flag (`atomicCAS-lock`) by reading the loop.
- Operand dtype: from the pointer type, the template argument, the PTX
  suffix, the Triton pointer's `dtype.element_ty`, or the torch tensor dtype
  at the call site. If the dtype is a template parameter, list every
  instantiation that is actually launched, or write `unknown`.
- Memory space: `__shared__` declaration, `smem_` naming with a
  `SharedStorage` struct, or a global pointer argument. Distributed shared
  memory (cluster) is its own value.

## 3. Which class?

- Float add/sub or float TMA reduce: class A, then go to step 4.
- Integer add/sub/inc/dec/or/and/xor, any max/min, exch, cas-as-lock:
  class B, then go to step 5.
- Call into a closed library: class C, then go to step 6.

## 4. Class A sub-cases

Do not stop at the first plausible sub-case. Record the evidence for the one
that applies, or record that none does.

- A1 (no contention): write down how the address is computed. Every
  contributor to the same address must be the same thread. A prior kernel
  that produces the index set is part of the evidence; state what makes the
  indices unique (a cumsum of counts, an argsort, a `blockIdx`-only address).
- A2 (serialised): find the lock or semaphore, the wait, and the release.
  All three must bracket the float accumulation. Note what happens if the
  serialisation is a template parameter defaulting to off; that is A3, not A2.
- A3 (gated): name the flag, the line that reads it, and its default at the
  pinned sha. State what the other branch does (workspace plus reduce kernel,
  or a different kernel entirely).
- If none applies, or the answer depends on inputs you cannot bound, class
  stays `A`, `confidence` is `low` or `medium`, and `notes` says what is
  unknown.

## 5. Class B follow-through

- Does the atomic's return value, or the counter it updates, choose a slot,
  offset, packet position, or "last block" role for data that a later
  floating-point reduction consumes? If yes, class is `B-indirect` and the
  `downstream` field is required.
- For `downstream.order_invariant`, ask: if two contributors swapped slots,
  would any float sum be performed in a different order? A per-row weighted
  sum indexed by the token's own id is invariant. A block-sequential sum over
  slots in slot order is not.

## 6. Class C

- Record the library, the call site in the engine, and every configuration
  knob that changes the reduction strategy. Do not describe what the binary
  does internally.

## 7. Reachability

- `path.description`: from a user-facing configuration (quant format,
  attention backend, model type, parallelism, phase) to the launch of this
  kernel. Cite the Python or C++ dispatch lines in `path.entry_points`.
- `default_path`: true only if a stock configuration with no opt-in flag
  reaches the site on supported hardware. Backward-only sites are
  `default_path: false` for inference and say so in `path.direction`.

## 8. Confidence

- `high`: every step above is answered with a `file:line`.
- `medium`: reachability or dtype has a gap, but the class is certain.
- `low`: contention, gating, or downstream order could not be settled.

## 9. Write the row

- Snippet: 3 to 6 verbatim lines including the atomic.
- `candidate_ids`: every candidate this row absorbs.
- Default-path class-A rows (plain `A`, no sub-case) get a second read by a
  different session before `cross_checked: true` is set.
