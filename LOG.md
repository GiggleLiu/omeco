# Attempt 059 — Continuous beta(span,t) freeze-out ladder

- **attempt**: 059
- **date**: 2026-08-20
- **kind**: improve
- **parent (machinery)**: 052 (span-gated cold ladder, via jg/surgery-v2
  library at eeaccf5); cycle-11 mechanism discussion (basin-structure /
  multigrid reading of the Set-A/Set-B ablation, PR #40 data)

## Hypothesis (pre-registered)

The 052 ladder is a HARD freeze: at ladder level S, nodes with subtree
span < S have beta = infinity (gated_sweep skips them) and eligible nodes
anneal at one cold beta. Hypothesis: the binary gate wastes moves at both
edges — just-unfrozen scales start at full cold beta (no fluctuation to
escape the coarse structure they were frozen under), while long-eligible
scales keep burning attempts they no longer improve. A CONTINUOUS
per-node inverse temperature beta_node = B(span, t) that sweeps a smooth
freeze-out front from coarse to fine (e.g. logistic in log2(span) with a
front position S(t) descending linearly in sweep count, width w, from
B_warm ~ 2.5 at the front to B_cold ~ 14 behind it; all parameters
functions of n only) yields strictly better anytime tc than the hard
ladder at matched sweep counts.

ATOMIC CHANGE vs parent: gated_sweep's `if span >= min_span {attempt at
beta}` becomes `attempt at beta = B(span, t)` (with B = +inf ahead of the
front so cost does not regress); everything else — move set, seed
portfolio, kick, ratchet, resync cadence — is byte-identical to the 052
pipeline.

## Novelty check

045 annealed elimination orders (different representation); 042 targeted
move SELECTION by congestion; 048 tempered whole replicas in time; 043
blocked sweeps by region for cache locality. None assigned a per-node,
scale-dependent temperature. Not a restatement of any prior attempt.

## Expected evidence

Validator (primaries reg3_250 + sycamore_m20, 90 s each): beat >= 1
current record by > 0.05, or TTF speedup >= 20%. Secondary (dev bench,
huawei, <= 600 s): at matched sweep budgets on surfacecode_d13 + ksg, the
continuous front's tc(t) curve dominates the hard ladder's, and the
accepted-improvement yield per span band shows fine-band gains
crystallizing only behind the front (the mechanism plot).

## Falsification

If matched-sweep tc(t) is within noise of the hard ladder on both dev
instances and both primaries, the binary gate is already an adequate
approximation of the freeze-out — record the yield-by-span curves and
close the temperature-shaping direction (it would also devalue 061's
premise, so say so explicitly).

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

## Outcome (recorded 2026-08-20)

**Validator (canonical host, v2.4, primaries 90 s):** status=scored,
score=-0.1333, NO RECORD. reg3_250 tc=39.882 (record 39.8833, delta +0.002
— tie, below RECORD_EPS), ttf 21.5 s (record 16.1 s). sycamore_m20
tc=61.790 (delta -0.268), ttf inf.

**Dev bench (huawei ecs-b88b, matched sweeps, 560 s plan):** the
pre-registered secondary evidence CONFIRMED on ksg — continuous front beats
the hard ladder by 1.73 / 4.71 / 2.81 bits at 380/620/940 sweeps — and is
a wash on surfacecode_d13 (-0.15 / +0.04 / n/a). Yield-rate per sweep is
LOWER for the continuous arm (0.02-0.04 vs 0.08-0.10): the win comes from
where the moves land (behind the front), not from more acceptances.

**Verdict (honest):** hypothesis SUPPORTED at matched sweeps on the
separable dev instance, NOT CONVERTED into a record on the primaries
(reg3_250 tie; sycamore slightly worse; TTF slower than record). The
freeze-out front is a better matched-work refiner but the 052 kick/ladder
chassis at 90 s budgets is already at the reg3 plateau. Candidate next
step (improve): continuous front INSIDE anneal_refine_rounds' cold pass
(PR #40) where matched-work refinement is exactly the use case.
Artifacts: dev-results-huawei.jsonl, devbench-huawei.log, report.json.
