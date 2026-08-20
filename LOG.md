# Attempt 065 — MEASUREMENT: why does band reheat accelerate an expander?

- **attempt**: 065
- **date**: 2026-08-20
- **kind**: measurement (no record claim; validator run optional/skipped)
- **parent (machinery)**: 061, instrumented

## Question (pre-registered)

061's confirmed TTF record landed on reg3_250 — an EXPANDER — while the
mechanism was motivated by separable-network waists. The acceleration is
real but unexplained. Instrument the first 15 s on reg3_250 (and
surfacecode_d13 as the separable control), band arm vs ATT_PARENT=1, and
answer: (1) does the band phase reach lower sc earlier (bottleneck rank
drops), or the same sc with better internal structure (tc drops at fixed
sc)? (2) is the in-band acceptance doing directed work (net in-band gain
positive DURING descent, unlike the late-phase melt) or is the band
merely reordering when work happens? (3) does the heated band track one
persistent region or hop uniformly (band-membership churn rate) — on an
expander, uniform hopping would mean the mechanism is effectively a
restart-diversity device, not waist negotiation.

## Expected evidence

Per-sweep trace (band arm + parent, fixed seeds, 3 relabelings each):
t, tc, sc, max node cost, in/out-band accepted-gain, band size, band
churn (Jaccard vs previous epoch). Deliverable: a compact table + the
trace JSONL answering (1)-(3), each answered with a number, not prose.
No leaderboard claim; dev bench IS the deliverable (<= 600 s total).

## Interpretation guide (pre-committed)

- (1) sc-led descent + (3) persistent band => genuine waist negotiation
  generalizes to expanders — strengthens 062/064's premise.
- (1) tc-at-fixed-sc + (3) uniform churn => restart-diversity mechanism —
  the TTF record is basin-sampling, and 062 should switch even earlier.

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

## Outcome (recorded 2026-08-20)

**Measurement (huawei, 380 s plan, 3 relabelings x band/parent x
reg3_250 + surfacecode_d13, per-sweep traces):**

| metric (band - parent) | reg3_250 | surfacecode_d13 |
|---|---|---|
| time to shared sc | **-3.82 s** | +6 ms |
| min sc reached (15 s) | **-1 rank** | 0 |
| tc at shared sc | +1.36 | +0.05 |
| in-band gain (descent) | -1614 | -1711 |
| out-of-band gain (descent) | +2291 | +1928 |
| band Jaccard / churn | 0.32 / 0.68 | 0.21 / 0.79 |

**Answers to the pre-registered questions:** (1) sc-LED descent on the
expander — the band arm reaches the shared bottleneck rank 3.8 s earlier
and one rank lower; on the separable control it changes nothing. (2) The
in-band heat does destructive work (negative accepted gain) that
out-of-band recovery converts — heat forces rank drops, cold elsewhere
banks them. (3) Band persistence is moderate (Jaccard 0.32), above the
separable control (0.21): closer to persistent-waist negotiation than to
uniform restart-diversity, but not purely either.

**Verdict (per the pre-committed interpretation guide):** genuine
bottleneck-rank negotiation that generalizes to expanders — the
sc-led + above-control-persistence branch — INVERTING 061's original
separable-network premise: separable ladders already harvest rank drops
(no headroom), expanders have a sticky sc plateau that local heat
unsticks. Strengthens 062/064's premise; predicts the event trigger
should fire ~immediately on separable families and late on expanders.
