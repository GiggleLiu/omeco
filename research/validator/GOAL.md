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

## v2.2 (2026-07-23) — anytime axis (time-to-frontier)

User direction ("push towards a better algorithm ... worth publication"),
grounded in the topic's own definition: an algorithm wins by better tc *or
by reaching the same tc faster*. Cycles 1–5 established that final tc at
90 s is saturated (13 mechanisms, 2 toolchains, budget-independent); the
open axis is *when* the frontier is reached.

- The harness polls the anytime `out.json` on **its own clock** during every
  run (candidate-declared times are never consulted; backdating is
  impossible by construction), rescores each distinct snapshot from
  topology, and derives the monotone tc(t) curve.
- **TTF** (time-to-frontier) per instance = first harness time at which the
  rescored tc ≤ pre-run tc record + 0.15 (the 0.15 band covers observed
  run-to-run spread of frontier-quality trees).
- `anytime_records` in the leaderboard holds the best confirmed TTF per
  instance, seeded from re-measured references (`treesa-inf`,
  `cotengra-sa`). A TTF claim needs a ≥20% speedup and passes the same
  confirmation-run / worse-of-two protocol; secondary reporting:
  tc@{1,3,10,30} s.
- Memorization risk (a candidate that recognizes the fixed instance and
  replays a stored tree instantly) inherits the accepted no-twins override;
  mitigations unchanged: per-run relabeling and mandatory code review of
  any record-setting attempt at the cycle gate.

## v2.1 (2026-07-23) — pure-tc objective

User direction: ignore sc entirely — minimize tc only, sc unbounded. The
sc-cap guard is removed from the validator (sc still reported,
informational). The cap-era board is archived at
`leaderboard_v2cap_archive.json`; records re-seeded from references
re-measured under pure tc (`ref:treesa-inf` = stock TreeSA with
`sc_target = ∞`, plus cotengra rows with flops-only objectives).
