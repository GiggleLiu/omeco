# Attempt 061 — Targeted waist-band reheating

- **attempt**: 061
- **date**: 2026-08-20
- **kind**: draft
- **parent (machinery)**: none (uses the 052 pipeline as chassis; informed
  by 042's failure and 054's waist analysis)

## Hypothesis (pre-registered)

The cold ladder converges everywhere but the binding constraint is the
WAIST — the argmax-cost node and the cuts on its root path. 054 showed the
waist cut is always globally improvable but two-sided rebuilds only pay on
separable networks; 042 showed steering MOVE SELECTION onto the argmax
collapses on peaky profiles. Untried middle ground: keep the move
distribution uniform and cold everywhere EXCEPT a heated BAND — all nodes
whose cost is within c bits of the current max, plus their root paths —
which anneals at a warmer beta_band (linear ramp per epoch; band and
temperatures recomputed every epoch from the current tree; c, betas,
epoch length functions of n). Local heating lets the tree renegotiate the
bottleneck's neighborhood without melting the rest, capturing part of
waist surgery's effect with no rebuild machinery, and composing with any
ladder. Claim: beats >= 1 record by > 0.05 (most likely ksg or
surfacecode-family style separable primaries via reg3_250 is unlikely —
honest expectation is the sycamore_m20 TTF or a primary tc tie; the
mechanism data is the main deliverable if no record falls).

## Novelty check

042 heated nothing — it biased which node to TRY by congestion softmax
and starved on peaky profiles; 048 tempered whole replicas; 054/PR#40
rebuild the waist structurally. Band-local temperature with uniform move
selection is untried.

## Expected evidence

Validator primaries (90 s): record beat or TTF >= 20%. Dev bench (huawei,
<= 600 s): on ksg + sycamore_m20 at matched sweeps, (a) acceptance rate
and realized tc-gain inside vs outside the band; (b) waist node cost
trajectory vs the cold-only parent; (c) sensitivity to band width c in
{1, 2, 4} bits — the 042 pathology check (band must not collapse to the
argmax alone).

## Falsification

If in-band heating never converts to retained global tc improvement
(gains melt back when the band cools) on both dev instances, the
bottleneck is genuinely structural (only rebuilds move it) — a clean
negative that sharpens the surgery-vs-annealing boundary from PR #40.

## Constraints (validator contract — non-negotiable)

- Binary: `omeco/examples/attempt.rs`, registered as example `attempt`
  (the validator builds `cargo build --release --offline --example attempt
  -p omeco` and runs `target/release/examples/attempt`).
- Contract: `attempt <graph.json> <budget_ms> <out.json>`; eager atomic
  best-so-far writes (tmp+rename, rate-limited ~150 ms + forced final flush);
  single thread (no Rayon); relabeling-invariant; pure tc objective;
  every knob a function of n; fixed RNG seed; LINEAR beta ramps.
- Hard wall limit enforced by the harness; treat budget_ms as the deadline.
- Dev instances readable at /Users/liujinguo/rcode/omeco/research/benchmark/targets/
  (never touch research/benchmark/private/).
- Baseline engine to modify: the 052-style pipeline as shipped in this
  worktree's library (see omeco/src/waist_surgery.rs `gated_sweep`,
  omeco/src/treesa.rs `fine_tune_tree_sa`) and the reference attempt
  pipeline in /Users/liujinguo/rcode/omeco/.worktrees/attempt-052/omeco/examples/attempt.rs
  (read-only): SIMPLIFY -> greedy seed portfolio -> warm kick + cold
  span-gated ladder ratchet loop.
- Also produce `dev_bench.sh <instances_dir> <out.jsonl>`: a deterministic
  mechanism-diagnostic benchmark, HARD-CAPPED at 600 s wall total (user
  directive), runnable on a 2-core Linux host (huawei).
