# Attempt 062 — Phase-switched composite: band reheat then freeze-out front

- **attempt**: 062
- **date**: 2026-08-20
- **kind**: improve
- **parent**: 061 (band reheat chassis in this worktree); imports the
  cold-phase mechanism of 059 (freeze-out front)

## Hypothesis (pre-registered)

Cycle-11's confirmed lesson: 061's band reheat wins the early
bottleneck-negotiation phase (reg3_250 TTF 7.1 s vs 16.1 s) but melts
gains late; 059's continuous freeze-out front wins matched-work
refinement (ksg −1.7..−4.7 bits) but starts slow. A PHASE-SWITCHED
composite — run 061's band-reheated ladder for the first
`f(n) = clamp(round(0.25 * total_planned_sweeps), min 2 epochs, max 40%)`
of the cold-phase sweep budget, then switch permanently to 059's
continuous freeze-out ladder (band heating off) — inherits both wins:
TTF at least as good as 061 AND final matched-sweep tc at least as good
as 059, on the same chassis.

ATOMIC CHANGE vs 061: one scheduler branch — before the switch point the
cold pass is 061's (band betas); after it, the cold pass is 059's
(continuous front, verbatim port). Kick, seeds, ratchet untouched.

## Expected evidence

Validator primaries (90 s): TTF <= 7.1 s on reg3_250 (retain the record)
AND tc delta >= 059's on sycamore_m20; any new record is a bonus. Dev
bench (huawei, <= 600 s): on ksg + surfacecode_d13 at matched sweeps,
composite final tc within noise of 059's arm AND early-phase tc(t)
within noise of 061's arm; report the switch-point sensitivity at
{15%, 25%, 40%}.

## Falsification

If the composite is worse than BOTH parents on either axis (the switch
destroys the band phase's structure before the front can refine it), the
phases do not compose by concatenation — record the tc(t) curves around
the switch point; that redirects to 064's event-triggered switch.

## Constraints (validator contract — non-negotiable)

- Binary: `omeco/examples/attempt.rs`, example name `attempt` (validator
  builds `cargo build --release --offline --example attempt -p omeco`).
- Contract: `attempt <graph.json> <budget_ms> <out.json>`; eager atomic
  best-so-far writes (tmp+rename, ~150 ms rate limit + forced final
  flush); single thread; relabeling-invariant; pure tc; knobs functions
  of n; fixed RNG seed; LINEAR beta ramps.
- Parent code: THIS worktree already contains attempt-061's
  omeco/examples/attempt.rs and dev_bench.sh — modify them (the atomic
  change below), keep `ATT_PARENT=1` reproducing the UNMODIFIED 061
  behavior byte-for-byte.
- 059's freeze-out front (for reference/porting):
  /Users/liujinguo/rcode/omeco/.worktrees/attempt-059/omeco/examples/attempt.rs (read-only).
- Dev instances: /Users/liujinguo/rcode/omeco/research/benchmark/targets/
  (never touch research/benchmark/private/).
- `dev_bench.sh <instances_dir> <out.jsonl>` hard-capped at 600 s total
  wall, runnable on 2-core Linux; print the budget plan first and abort
  if it would exceed 600 s.
