# Attempt 063 — Confirmation-robust early descent (sycamore TTF claim)

- **attempt**: 063
- **date**: 2026-08-20
- **kind**: improve
- **parent**: 061

## Hypothesis (pre-registered)

061 claimed sycamore_m20 TTF 1.3 s (record 39.6 s) but the confirmation
run hit the wall limit, so the claim stands unconfirmed. Root cause
hypothesis: 061's first snapshot write and early epoch overhead are
budget-shape-sensitive — under the confirmation run's fresh relabeling
the first frontier-quality snapshot lands later than the harness poll
needs. Making early descent CONFIRMATION-ROBUST — (a) write the first
valid tree immediately after the greedy portfolio (before any anneal);
(b) halve the first two band epochs' sweep counts so the first ladder
descent lands earlier; (c) snapshot on every improvement during the
first 5 s regardless of rate limit — preserves the mechanism while
making the TTF measurement land inside the wall limit. Claim: the
sycamore_m20 TTF record confirms (>= 20 percent speedup, worse-of-two)
and reg3_250's 7.1 s record is retained.

ATOMIC CHANGE vs 061: snapshot cadence + first-two-epoch sweep halving
only. Betas, band logic, ladder untouched.

## Expected evidence

Validator primaries: confirmed sycamore_m20 TTF record; reg3_250 TTF
<= 7.1 s retained. Dev bench: on sycamore_m20 + reg3_250, time of first
snapshot within 0.5 s of start, and tc(t) at {1,3,10,30} s not worse
than 061's arm.

## Falsification

If with robust snapshotting the sycamore TTF reverts toward the old
record, 061's 1.3 s was a relabeling-lottery artifact, not a mechanism
property — record both relabelings' curves; the claim dies cleanly.

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

**Validator (canonical host):** score -0.1254, NO RECORD. reg3_250
tc=39.882 (tie), ttf first-run 2.9 s but confirmation 7.73 s -> claim
not confirmed (record stays 7.069 s, attempt-061). sycamore_m20
tc=61.774 (-0.252), **ttf = inf: with robust snapshotting the 1.3 s
claim vanishes entirely.**

**Dev bench (huawei, 415 s plan):** the modification FAILED its own
evidence gate — the robust arm's tc(t) is consistently WORSE than pure
061 (sycamore tc@1 79.98 vs 62.42; reg3 tc@30 40.02 vs 39.88): halving
the first two band epochs damages early descent rather than making it
measurement-robust. First-snapshot latency was already ~30 ms in the
parent, so change (a)/(c) had nothing to fix.

**Verdict (honest, both pre-registered falsifications hit):**
(1) 061's sycamore_m20 TTF 1.3 s was a relabeling-lottery artifact, not
a mechanism property — the claim dies. (2) The early epochs are
load-bearing for descent; shortening them is harmful. Blacklist:
epoch-shortening as a confirmation-robustness device. The reg3 7.1 s
record (061) stands as the mechanism's real, confirmed effect.
