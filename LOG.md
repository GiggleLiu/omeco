# Attempt 028 — convergence-optimized single continuous anneal at non-converged scale

- **date:** 2026-07-23
- **kind:** draft
- **parent:** none
- **targets (scored):** reg3_1000 (tc rec 135.75), rqc_97_m24 (tc rec 106.47), 90 s each, single thread

## Hypothesis

At non-converged scale (n≈1000–1238) the stock doubling-restart TreeSA schedule wastes
budget on re-warmup and on uniform move proposals over mostly-cheap nodes. A
convergence-optimized SA should beat it by a wide margin via:

1. **ONE continuous anneal** — no ntrials restarts. A single trajectory with a
   wall-clock-indexed geometric cooling ramp β(t): 0.02 → 20 over 0.95×budget.
   Restarts pay re-warmup (re-melting an already-cooled good tree) which does not
   amortize inside 90 s at this scale.
2. **Profile-aware move targeting** — instead of a full post-order sweep that visits
   every node uniformly, keep a flat arena of the n−1 internal nodes tagged with
   each node's *local* contraction tc, and with p=0.7 draw the move's node from the
   current top-cost decile (nodes whose local tc dominates the global log2-sum-exp),
   else uniform. At n=1000 uniform sampling spends ~90% of proposals on nodes that
   contribute negligibly to total tc. Same mechanism sibling 027 tests at small
   scale; here the effect should be amplified ~n-fold.
3. **O(1) incremental cost maintenance** — arena moves are pure pointer rewires plus
   two `local_tc` recomputations (the mutated node and its restructured child); no
   full-tree recomputation anywhere in the hot loop. Global tc is derived lazily
   (every K moves, O(n) log-sum-exp over `local_tc`) only to decide new-best
   snapshots. Target > 10^6 moves/sec at n=1000.

Energy is pure tc (v2.1 sc-unbounded objective): ΔE = tc1 − tc0 (local), Metropolis
accept `rng < exp(−β·ΔE)`. Greedy seed written immediately; best-true-tc snapshots
flushed anytime.

## Expected evidence

- Confirmed records: reg3_1000 < 135.70 and/or rqc_97_m24 < 106.42 (reference
  variance is several log2 at this non-converged scale, so multi-log2 gains plausible).
- Local anytime curves (best tc at 5/10/20/40/90 s) for this method vs the
  `treesa_tuned` baseline binary on both targets — the paper's data either way.
- moves/sec achieved at n=1000.

## Implementation

`omeco/examples/attempt.rs` — a self-contained arena SA. Key pieces:

- **Flat arena** (`struct Arena`): the binary tree as parallel `Vec`s
  (`left/right/parent/out/tid/ltc`). The node *set* (n leaves + n−1 internal) is
  fixed for the whole run; moves only rewire pointers and rewrite `out`/`ltc` of
  two nodes. `internal` is a fixed list of internal-node indices in pre-order
  (parents before children — same traversal shape TreeSA sweeps in).
- **O(1) incremental moves** (`try_move`): the four rotation rules implemented
  directly on the arena using omeco's public `tcscrw` + `compute_intermediate_output`.
  An accepted move rewires ≤3 child pointers, updates 2 parents, rewrites one
  child's `out`, and recomputes exactly two `ltc` entries. Metropolis on the
  *local* Δtc = log2sumexp(ltc_child', ltc_x') − log2sumexp(ltc_child, ltc_x),
  identical to omeco's `rule_diff` energy. **Correctness verified**: arena's
  incremental global tc == an independent `tree_complexity` of the seed
  (279.0000 == 279.0000; and every emitted tree re-measured with
  `contraction_complexity` matches the arena's tc to <1e-4).
- **Lazy global tc**: `global_tc()` = log2 Σ 2^ltc over internal nodes, O(n),
  called once per sweep only to decide new-best snapshots — never inside the
  per-move loop (no full-tree recomputation in the hot path).
- **Systematic sweep + profile-aware boost**: each pass is a full sweep over all
  internal nodes (systematic base — this turned out to be essential; pure
  top-decile random targeting was far worse, see decisions) PLUS `boost = n_int`
  extra attempts drawn from the current top-cost decile (`ATTEMPT_BOOST=1`
  default). Top-decile list rebuilt every 4 sweeps by sorting `ltc`.
- **Single continuous anneal**: β(t) = 0.02·(f·ln(1000))^exp with f =
  elapsed/(0.95·budget) clamped, i.e. geometric 0.02→20; β_max held on the tail.
  No restarts. Greedy portfolio seed (best-of ≤24 stochastic-greedy within a
  ≤6 s / 6 %-budget slice) feeds the arena; deterministic greedy written
  immediately as the anytime fallback. Cycle/reheat machinery exists
  (`ATTEMPT_CYCLES`, `ATTEMPT_REHEAT`) but **default is cycles=1** (see below).

Also added `omeco/examples/measure.rs` (offline tc/sc re-measurement helper;
separate example, does not affect the `attempt` build).

### Decisions (empirical, on the two large targets)

1. **Pure top-decile random targeting FAILS.** The first draft (p=0.7 top-decile,
   p=0.3 uniform, no systematic sweep) reached only tc=256 on reg3_1000 in 8 s vs
   treesa_tuned's 146 — the systematic post-order coverage that TreeSA relies on
   is essential at n=1000. Pivoted to systematic sweep as the base.
2. **Profile-aware boost as an additive layer HELPS.** With systematic sweep as
   base, adding `boost = n_int` targeted attempts on the top-cost decile improved
   reg3_1000@20s from 141.9 (BOOST=0) to **139.4** (BOOST=1); BOOST=3 over-focused
   and hurt (149.7). So the top-decile mechanism does contribute — as a boost, not
   a replacement. `boost = n_int` (×1) is the default.
3. **Reheat cycles do NOT help at this budget.** cycles=2/3 (elitist reload+reheat)
   were within noise of / slightly worse than the single ramp at 90 s
   (135.8 vs 133.6–136.3). Kept cycles=1 (single continuous anneal), consistent
   with the hypothesis.
4. **Slow cooling (0.95) ≥ early cooling (0.85).** cool=0.85 gave 135.1; cool=0.95
   produced the best single result (133.6). Kept 0.95.

## Precheck

PASS. `validate … --precheck` → `status=scored`, `errors=[]` (build +
chain_10@2s structural check clean). No scored validation run (per protocol).

## Anytime curves (best tc at 5/10/20/40/90 s, single thread, 90 s budget)

reg3_1000 (record 135.75):

| t(s) | attempt-028 | treesa_tuned |
|-----:|------------:|-------------:|
|   5  |   264.32    |   145.91     |
|  10  |   264.32    |   137.07     |
|  20  |   264.32    |   137.07     |
|  40  |   179.79    |   137.07     |
|  90  | **133.57**  |   135.41     |

rqc_97_m24 (record 106.47):

| t(s) | attempt-028 | treesa_tuned |
|-----:|------------:|-------------:|
|   5  |    (seed)   |   124.95     |
|  10  |    (seed)   |   124.95     |
|  20  |    (seed)   |   106.83     |
|  40  |   149.41    |   106.83     |
|  90  | **106.30**  |   106.83     |

Repeat finals (validator relabels each run → this spread is representative):
- reg3_1000, attempt-028: 133.57 / 135.11 / 136.32  (mean ≈ 135.0)
- reg3_1000, treesa_tuned: 135.41
- rqc_97_m24, attempt-028: 106.43 / 106.30  (mean ≈ 106.36)
- rqc_97_m24, treesa_tuned: 106.83

Throughput: **~5.0–5.9 ×10^5 moves/s** at n=1000–1238 (below the 10^6 target —
the boosted moves target high-arity nodes near the root whose `out` can hold
~130 labels, so their `tcscrw`/`compute_intermediate_output` are O(d²) on the
linear membership path; the scratch-bitset path would help but was not wired in).

## Outcome vs hypothesis

Partially confirmed, honestly mixed:

- **rqc_97_m24: reproducible win.** 106.30–106.43 beats treesa_tuned (106.83) by
  ~0.4–0.5 and edges the record (106.47). A genuine terminal-tc improvement.
- **reg3_1000: a tie, not a wide margin.** attempt-028 mean ≈135.0 vs
  treesa_tuned 135.41 vs record 135.75 — comparable; individual runs straddle the
  record (best 133.57 < 135.70).
- **Anytime efficiency is WORSE early, not better.** The single stretched ramp
  banks its gain only in the last ~40 %; treesa_tuned's fast repeated anneals
  reach a near-final tree by 10–20 s. The hypothesis's "anytime efficiency wins"
  is contradicted for t≪budget; only the fixed-90 s terminal value is
  competitive-to-better. Since the scored budget is a fixed 90 s, the terminal
  values are what count.
- **Mechanisms individually validated:** O(1) arena moves are correct and fast
  enough to run a single uninterrupted anneal; profile-aware top-decile boost is
  a measurable within-method improvement over uniform sweeping.


## Scored outcome (validator v2.1 pure-tc, 2026-07-23)
- status: scored, score (mean Δtc vs pre-run records): -3.3618
- record_updates: none
- reg3_1000: pass — tc=140.779 sc=134.000 record=135.75412004851293 delta=-5.025
- rqc_97_m24: pass — tc=108.167 sc=101.000 record=106.46849426445668 delta=-1.698
