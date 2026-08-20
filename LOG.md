# Attempt 064 — Event-triggered phase switch (knob-free reheat→refine)

- **attempt**: 064
- **date**: 2026-08-20
- **kind**: draft
- **parent (machinery)**: 061 chassis (+ 059 front, as in 062)

## Hypothesis (pre-registered)

062 fixes the reheat→refine switch at a sweep fraction; the right switch
point should be instance-dependent. The waist-cost trajectory provides
the signal: during the reheat phase the max node cost drops in steps;
when it stalls (no improvement of the max node cost by > 0.25 bits over
a window of W(n) = clamp(ceil(sqrt(n)), 8, 32) sweeps — the same epoch
scale 061 uses), bottleneck negotiation is over and further heat only
melts gains. Switching ON THE STALL EVENT (permanently, to the 059
front) matches or beats 062's best fixed fraction on both axes without
any switch knob, and adapts across instance families (expander vs
separable) where any fixed fraction must compromise.

## Expected evidence

Validator primaries: TTF <= 062's on reg3_250, tc >= 062's on
sycamore_m20 (or records). Dev bench (huawei): on ksg + reg3_250 +
surfacecode_d13 (3 instances, short budgets to fit 600 s), the triggered
switch time varies by family (later on expanders, immediate/early on separable)
and final tc >= the best fixed-fraction arm per instance; report the
stall-window sensitivity at W and 2W.

## Pre-implementation trigger calibration (attempt-065 trace)

Huawei, 15 s, 3 relabelings, band versus parent: on reg3_250 the band
phase reaches the shared sc 3.8 s earlier and one rank lower, with negative
accepted gain in-band, positive gain out-of-band, and band Jaccard 0.32.
On surfacecode_d13 it does nothing (d_t_sc +6 ms, d_min_sc 0, node cost
+1.0). Therefore the detector is allowed to fire on its first eligible
window: that is expected on the separable family, while continuing max-node
cost descent keeps expanders in reheat.

## Falsification

If the trigger fires degenerately (immediately or never) on any family,
or a single fixed fraction dominates it everywhere, the waist-stall
signal does not carry switch information — record trigger times and the
per-instance comparison; keep 062's fixed switch as the mechanism.

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

## Implementation and local verification

- Event rule: during the cold band phase, compare the current maximum node
  cost to its value at the start of each non-overlapping W-sweep window.
  Improvement must be strictly greater than 0.25 bits to open another
  window; otherwise switch permanently to the 059 front. The first eligible
  W-sweep window can trigger. `ATT_STALL_WINDOW_MULT=2` is the dev-only 2W
  sensitivity arm.
- Compatibility controls at 1024 matched sweeps on surfacecode_d13:
  `ATT_PARENT=1` is byte-identical to attempt-061 (tc 31.3013, 35,403
  accepts), and `ATT_FIXED_SWITCH=0.25` is byte-identical to attempt-062
  (tc 31.1804, 35,249 accepts).
- Required 10 s surfacecode_d13 smoke: event switched after 17 cold sweeps
  at 244.846 ms (`waist 30.0 -> 30.0`, gain 0.0 bits), ending at tc 30.4923;
  `ATT_PARENT=1` ended at tc 30.5169.
- Determinism: two event runs capped at 1024 sweeps produced byte-identical
  JSON (SHA-256 `ce867b037e6f8a5efce3607b597388439d981a6ddca8746dc7a642a09f15e834`),
  tc 31.2281, and 35,071 accepts. Only wall-time diagnostics differed.
- Reduced local harness plumbing check (`ATT_DEV_BUDGET_MS=10000
  ATT_DEV_SWEEPS=512`) completed all 15 rows. Both W and 2W switched in all
  three instances. On reg3_250, W switched on the first window (32 cold
  sweeps, 94.402 ms), while 2W continued through improving windows and
  switched at 128 cold sweeps (105.198 ms). This is not Huawei evidence and
  does not support the intended default-W family separation at that short
  sweep cap.
- Offline release build, example unit tests (7/7), rustfmt, clippy `-D
  warnings`, shell syntax, Python syntax, and the >600 s plan refusal path
  pass. Per instruction, the validator was not invoked.

## Outcome (recorded 2026-08-20)

**Validator (canonical host):** score -0.0328. **NEW ANYTIME RECORD
(confirmed, worse-of-two): sycamore_m20 TTF 5.3 s vs 39.6 s — 7.5x** —
the record 061 claimed and 063 disproved as a lottery artifact now lands
legitimately. reg3_250: tc tie, but TTF REGRESSES to 13.1 s (record
7.069 s, 061): the early switch cuts the band phase that reg3 needs.

**Dev bench (huawei, 565 s plan):** the stall TRIGGER IS DEGENERATE —
waist gain 0.0000 in the FIRST window on all three instances, so it
fires ~immediately everywhere (147-149 ms on reg3/d13; 2.6 s on ksg),
not family-adaptively as hypothesized. Event-W loses to the best fixed
fraction on final tc everywhere (+0.82 ksg / +0.57 reg3 / +0.05 d13);
2W beats W on tc everywhere (longer windows = later switch = better),
confirming the signal carries no per-family switch information at this
threshold.

**Verdict (honest, pre-registered falsification hit):** the waist-cost
stall signal is degenerate — the max node cost simply does not improve
within any sqrt(n) window under band heat, so "stall" is always true.
Per the LOG's abandonment clause, 062's fixed-fraction switch remains
the mechanism. The sycamore record is REAL but its attribution is
"switch-to-front almost immediately is optimal for sycamore's TTF" —
i.e. per-family switch points matter (sycamore ~0, reg3 late), which is
an argument FOR a working adaptive signal and AGAINST this particular
one. Candidate signal for a future attempt: sc-trajectory stall
(065 showed descent is sc-led), not max-node-cost stall.
