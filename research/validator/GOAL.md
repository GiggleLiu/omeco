# Goal — beat-existing-optimizers

**Publishable bar** (approved by user 2026-07-22, "either win counts"):

A candidate optimizer PASSES iff, on the full instance suite with all guards
satisfied, at least one of:

- **Quality win**: mean Δtc@budget ≥ **+1.0** (log2; ≥2× fewer flops than the
  TreeSA baseline at matched optimizer wall-clock), where the mean runs over
  all (instance, budget) pairs with budgets T(g), T(g)/4, T(g)/16 — AND no
  instance regresses worse than **−0.5** at full budget T(g).
- **Speed win**: at budget T(g)/4, candidate tc is within **0.3** of the
  baseline tc on **every** instance (finds baseline-quality trees 4× faster).

The same bar must hold on the sealed holdout (aggregate booleans only), plus
the generalization guard: holdout mean Δtc ≥ dev mean Δtc − 1.0.

Definitions: Δtc(g, b) = tc_baseline(g) − tc_candidate(g, b), tc recomputed
by the validator's own scorer from the emitted tree (candidate numbers are
never trusted). T(g) = measured wall-clock of baseline TreeSA (ntrials=1,
niters=100/50, `RAYON_NUM_THREADS=1`) from `research/database/baselines.json`
(dev) or the sealed private baselines (holdout). Budgets are floored at
100 ms to absorb process-launch overhead. Guards (all enforced, see
`research/topics.md`): valid-tree, sc-cap (sc ≤ baseline sc + 2), held-out
generalization, resource-cap (single-thread wall-clock; CPU-time check),
timeouts marked at budget × 1.05.

Failed/timeout/sc-violating (instance, budget) pairs contribute Δtc = −5.0
to the mean; any invalid tree rejects the whole candidate.

## Attempt contract

A candidate is an attempt worktree of this repo. The validator invokes:

1. `build.sh` if present, else `cargo build --release --offline --example
   attempt -p omeco` (sandboxed, no network, 300 s).
2. `attempt.sh <graph.json> <budget_ms> <out.json>` if present, else
   `target/release/examples/attempt` with the same arguments — sandboxed,
   `RAYON_NUM_THREADS=1`, killed at budget × 1.05.

The attempt must write its best contraction tree in omeco `writejson` format
to `<out.json>` before the deadline (write early, improve in place —
anytime). Instances are randomly relabeled/permuted per scored run, so
memorized answers keyed on input content do not transfer. `validate
<dir> --precheck` is free and unlimited; scored runs are what count.
