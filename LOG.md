# Attempt 060 — Activity-based worklist freezing (active-set cold ladder)

- **attempt**: 060
- **date**: 2026-08-20
- **kind**: improve
- **parent (machinery)**: 052 (span-gated cold ladder, via jg/surgery-v2
  library at eeaccf5); cycle-11 mechanism discussion

## Hypothesis (pre-registered)

In late cold-ladder passes almost all rewrite attempts are rejected at
long-stale nodes; the sweep is O(n) but the acceptances cluster near
recently changed regions. Freezing by STALENESS — after each full ladder
pass, re-attempt only nodes within tree-distance d of a rewrite accepted
in the last pass (activity set; d and the refresh cadence functions of n;
periodic full sweeps every k passes to preserve ergodicity/correctness) —
preserves the dynamics' fixed points while making late sweeps O(active),
so at a fixed wall budget the engine executes strictly more effective
passes and reaches lower tc. This is the classic FM/KaHIP active-set trick
applied to tree rewrites; it strengthens the anytime (TTF) axis that
cycle 7 made primary.

ATOMIC CHANGE vs parent: sweep scheduling only (worklist instead of full
recursion); move set, betas, ladder, ratchet identical.

## Novelty check

043 (cache-blocked regional annealing) is the nearest prior: it PARTITIONED
sweeps into blocks for memory locality and failed because blocking cut the
frequency of global moves. 060 differs in kind: full-tree ladder passes
remain (global moves keep their cadence); only re-attempts at stale nodes
are pruned, and periodic full sweeps bound the staleness error. No prior
attempt did acceptance-driven active sets.

## Expected evidence

Validator primaries: beat >= 1 record by > 0.05 or TTF >= 20% faster
(TTF is the natural axis for a throughput mechanism). Dev bench (huawei,
<= 600 s): on ksg + surfacecode_d13, attempts/sec and accepted-moves/sec
vs the full-sweep parent at equal wall; the active-set fraction over time;
equal-final-tc-at-lower-wall or lower-tc-at-equal-wall.

## Falsification

If the active-set fraction stays near 1 (activity does not localize) or
pruned sweeps lose more quality than their speedup buys on both dev
instances, record the activity-localization curve and kill the direction.

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
score=-0.4954, NO RECORD, FALSIFIED on both pre-registered axes for the
primaries: reg3_250 tc=40.632 (delta -0.749), sycamore_m20 tc=61.764
(delta -0.242); TTF inf on both (never reached frontier).

**Dev bench (huawei, matched wall 30/75 s):** split verdict — ksg active
mode is -1.652 / -1.718 bits BETTER at 30/75 s with sweep-throughput
x1.97 / x0.98 and active fraction 0.33-0.41; surfacecode_d13 is a wash
(+0.009 / +0.001). Activity DOES localize (fraction ~0.4) and the
throughput win is real at short budgets on large sparse instances.

**Verdict (honest):** falsified as a record-setter on the primaries —
reg3_250 (expander) actively punishes staleness pruning (locality
assumption fails when improvements are spatially uniform), and the
long-budget advantage evaporates. The ksg short-budget gain suggests the
mechanism is an ANYTIME accelerant for large separable networks only;
any follow-up should gate the worklist on a measured activity-locality
statistic rather than enabling it unconditionally.
Artifacts: dev-results-huawei.jsonl, devbench-huawei.log, report.json.
