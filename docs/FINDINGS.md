# Findings notes

The README records the bounded source and runtime findings. The following argument concerns a proposed verifier, not a measured attack or a proved security property.

## Distinguishable batch histories

Suppose an operator can choose among K feasible **complete shape histories**, and those choices induce D distinct output trajectories for a fixed target. Then D ≤ K and the choice can encode at most log2(D) ≤ log2(K) bits per trajectory, assuming all choices can be deliberately selected and distinguished. This is an upper bound on a choice set, not an experimentally established channel capacity. Many histories may produce the same output, and scheduling constraints may prevent choices from being realised reliably.

A T-step bound of sum over t of log2(K_t), or T log2(K) when each step has K choices, requires a different assumption: independently realisable, distinguishable choices at each step. It cannot be obtained by multiplying the number of complete histories by the number of tokens. The current experiments do not establish those independence, feasibility or distinguishability conditions.

## Reproduction and scheduler freedom

Accepting any output in a set of admissible histories can leave freedom to an operator who controls scheduling. Recording or committing a history and checking only that history may bind a later replay to an earlier execution. It does not remove the information encoded by the operator's original choice of history.

Reducing that freedom would require an independently prescribed schedule, trusted or verifiably constrained scheduling, or a demonstrated batch-invariant output for the relevant deployment. Each option needs an explicit threat model, commitment timing, target-state authentication and treatment of model, sampler, cache and neighbour dependencies. A deterministic kernel alone does not provide these protocol properties. None is proved by the present census or repeated-run observations.
