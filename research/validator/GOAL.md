# Goal — beat-existing-optimizers (v2, optimization mode)

**User direction (2026-07-22):** the goal is a better *algorithm*, pursued as
open-ended optimization — no pass/fail bar. The loop maintains a list of
candidate algorithms and targets better and better results on two fixed
instances. Cycle 1's v1 bar (either-win) was met by a schedule-hygiene
candidate (attempt-004); v2 is hardened so that only algorithm-level changes
can score.

## Objective

Minimize validator-recomputed **tc** (log2 flops) on the two target
instances in `research/benchmark/targets/`:

- `reg3_250` — random 3-regular, 250 tensors (expander; the hard dev graph)
- `sycamore_m20` — Sycamore-m20-scale RQC proxy, 561 tensors, 963 indices

at a fixed optimizer budget of **90 s per instance**, single-threaded
(`RAYON_NUM_THREADS=1`), instance randomly relabeled per run.

## Leaderboard and records

`research/validator/leaderboard.json` holds the best-known tc per instance
(the *record*), writable only by the validator (attempts are sandbox-denied
from `research/`). Records are seeded by reference rows measured with the
same pipeline (best of 3 runs each):

- `ref:treesa-baseline` — the unmodified greedy+TreeSA doubling attempt
- `ref:treesa-hygiene` — attempt-004's schedule-hygiene variant

Because the hygiene reference is in the floor, schedule/parameter tuning
gains ≈ 0 by construction; only search-mechanism changes can beat the
record. A run beating a record by > 0.05 triggers **one confirmation run**
(fresh relabeling); the *worse* of the two is recorded. A scored run's
`score` = mean over instances of (pre-run record tc − candidate tc).

## Rules

1. **Algorithm-level changes only**: batch planning rejects hypotheses whose
   only change is parameters/schedules; the schedule-only control
   (attempt-004's binary) must never set a record — it is a standing
   negative control.
2. Guards (all enforced as instance failure or rejection): valid tree
   (validator recomputes tc/sc from topology; candidate numbers never
   trusted), sc-cap (reference sc + 2 per instance), resource cap
   (single-thread, CPU-time check), timeout at budget × 1.05, wall limit
   300 s per scored run.
3. Termination is the user's decision, not a threshold's.

## Accepted residual risks (user-approved, 2026-07-22)

- No fresh-twin / holdout guard: records on the two fixed instances may in
  principle be achieved by instance-specific memorization or overfitting;
  mitigations are per-run relabeling, code review of record-setting
  attempts at cycle gates, and the confirmation-run rule.
- References are omeco-only (no cotengra rows yet); records measure
  progress vs omeco's own best, not external SOTA.

## Attempt contract (unchanged from v1)

Candidate = attempt worktree. Validator runs `build.sh` (else
`cargo build --release --offline --example attempt -p omeco`), then
`attempt.sh <graph.json> <budget_ms> <out.json>` (else
`target/release/examples/attempt`), sandboxed. Write the best tree in omeco
`writejson` format early and improve it in place (anytime). `validate
<dir> --precheck` is free and unlimited.
