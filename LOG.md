# attempt 038

- date: 2026-07-24
- kind: draft
- parent: none

## Hypothesis (pre-registered before any implementation)

`dbn_13` (572 tensors over only 44 labels) and `nqueens_28` (4252 tensors)
have HYPEREDGE structure — labels shared by many tensors — where pairwise-greedy
seeds and uniform SA moves are misaligned with the real cost geometry (a shared
label is eliminated only when its LAST holder is contracted). A hyperedge-aware
pipeline wins:

1. **Seed by hyperedge scheduling** — order labels by an elimination heuristic
   over the LABEL hypergraph (variable elimination: min weighted fill /
   fewest-holders-first with lookahead). Contract each label's holder-group via a
   small optimal/greedy subtree when its turn comes.
2. **SA refinement whose move proposals are biased toward nodes holding
   high-degree labels near their elimination point** — systematic post-order
   sweeps as the base layer (pure targeting is blacklisted); the bias adds extra
   rule attempts at high-degree nodes on top of the systematic base.
3. **LINEAR beta schedule** (geometric blacklisted).

**Claim:** beat the `dbn_13` (29.09) or `nqueens_28` (121.37, median-of-3)
record by > 0.05.

**Falsification:** hyperedge-aware seed + biased SA lands within ±0.3 of the
records on both instances → hyperedge structure is already handled implicitly by
generic SA. Record label-degree histograms and seed-vs-final tc for attribution.

## Structure observed (read-only inspection of targets, pre-implementation)

- `dbn_13`: 44 labels, all dim 2; 88 arity-1 tensors + 484 arity-2 tensors;
  `iy` empty. The 484 arity-2 tensors form a 22-regular graph on the 44 labels.
  This is EXACTLY a partition-function / variable-elimination (treewidth)
  problem over 44 binary variables. Label-degree histogram: all 44 labels held
  by 24 tensors each.
- `nqueens_28`: 4086 labels, all dim 2; 1116 arity-1 + 3136 arity-3 tensors;
  `iy` empty. Variable-elimination over 4086 binary variables with 3-way
  factors; interaction-graph degree <= 8. Label-degree histogram: 784 labels of
  degree 5, 3302 of degree 2.

Both are graphical-model partition functions — the canonical match for a
variable-elimination (bucket-elimination) seed rather than pairwise greedy.

## Design

- Seed: variable-elimination contraction tree. Interaction over labels; a lazy
  priority queue drives the elimination order under two heuristics (min-cost /
  min weighted fill, and fewest-holders / min-degree). Each eliminated label's
  holder group is merged by a local greedy min-union pairwise order. Topology is
  converted to a `NestedEinsum` by exact outside-occurrence counting (matches the
  validator scorer bit-for-bit, so my measured tc == scored tc).
- SA: warm anneal reused from attempt-032 (linear beta 0.04..14, systematic
  post-order sweeps), seeded from the best VE tree (also races a greedy seed).
  Diversify restarts on stagnation use fresh RANDOMIZED VE seeds. Sweeps carry a
  light degree-bias: extra rule attempts at nodes whose labels have high
  hyperedge degree, on top of the systematic base pass.
- Pure-tc objective, single-threaded, atomic best-so-far writes, no per-instance
  constants, relabeling-robust.

## Results

All runs single-threaded (`RAYON_NUM_THREADS=1`), budget 90000 ms, scored by the
independent `research/validator/scorer.py`. tc == my measured tc (the VE builder
derives node outputs by the exact outside-occurrence rule the scorer uses).

### Label-degree histograms (hyperedge structure)

- `dbn_13`: all 44 labels held by 24 tensors each (pure hyperedge, dim 2).
- `nqueens_28`: 784 labels of degree 5, 3302 of degree 2 (dim 2); 4086 labels.

### dbn_13 (record 29.09) — WIN, confirmed twice

| stage | tc |
|-------|----|
| greedy seed | 44.10 – 44.71 |
| **VE seed (min-cost)** | **28.7944** |
| warm SA final, run 1 | 28.7505 |
| warm SA final, run 2 | 28.7568 |

Worse-of-two = **28.7568**, beating the 29.09 record by **0.333** (>> 0.05).
The win is the SEED: the variable-elimination order gives 28.79 immediately,
vs 44.7 for pairwise greedy — a 16-log2 gap. Warm SA adds only ~0.04. This is
`ve_fits=true`; the VE basin is refined by pure-intensify linear-beta SA. Every
run lands 28.71–28.79, all well under 29.04, so the confirm-twice margin is
robust to stochastic variation. **Hypothesis mechanism validated.**

### nqueens_28 (record 121.37, median-of-3) — mechanism FALSIFIED; generic fallback strong but high-variance

| stage | tc |
|-------|----|
| greedy seed | 384 – 460 |
| **VE seed** | **714.16 (fails)** |
| adaptive-doubling TreeSA final | 120.02 / 127.17 (two valid runs) |

The VE seed is USELESS here: `tc_ve=714 >> tc_greedy=384`, and on 4086 labels a
complete min-cost VE order does not even finish inside the time box (it aborts
and chain-merges to garbage). So `ve_fits=false` and the pipeline falls back to
the proven library TreeSA under an anytime doubling schedule. That fallback is
strong — one 90 s run reached **120.02, which beats the 121.37 record by 1.35** —
but it is high-variance across runs (120–153) and, crucially, owes NOTHING to the
hyperedge mechanism: it is generic simulated annealing. So for `nqueens_28` the
hypothesis is **falsified** (the hyperedge-aware seed does not help; any gains are
generic), and because of the variance the record is not reliably beaten
confirm-twice.

### Attribution / verdict

- The hyperedge-aware VE seed WINS decisively on `dbn_13` (small label hypergraph,
  44 vars) — seed 28.79 vs greedy 44.7, beating a tuned-TreeSA record. Confirmed.
- The VE seed FAILS on `nqueens_28` (4086 vars): too slow to complete and, when
  chain-aborted, far worse than greedy. Generic TreeSA does the work there.
- Boundary of the effect: the hyperedge/variable-elimination seed pays off when
  the LABEL hypergraph is small enough that a good elimination order can be found
  cheaply; on large label sets it is dominated by direct tree simulated annealing.
- The SA components (linear beta, systematic sweeps + light degree-bias) were kept
  from attempt-032 and are not the source of the dbn win (the seed is); on
  nqueens they are bypassed in favour of library TreeSA.

Claim status: **beat dbn_13 by 0.333 confirmed twice (> 0.05)** → primary claim
met. nqueens beaten in one run (120.02 < 121.37) but not reliably.

## Engineering notes

- Ported attempt-032's warm-anneal machinery (`WarmAnnealCtx`,
  `prepare_warm_anneal`, `warm_exprtree_to_nested`) into `omeco/src/treesa.rs`
  (only additions; doctests pass).
- Fixed a budget-overrun bug: the SA deadline check stride (`CLOCK_EVERY=8`) was
  too coarse when a single sweep costs 100s of ms on 4k-node trees (a 20 s budget
  ran to 41 s). Now the loop refuses to START a sweep whose measured duration
  would cross the deadline; verified wall <= budget on all runs.
- Large-graph outputs are inherently huge in `writejson` (nqueens tree ~0.6 GB;
  the reference `treesa_tuned` produces the same size) — not a defect of this
  attempt.
- clippy `-D warnings` clean, rustfmt clean on the two changed files, offline
  build OK, treesa doctests pass.
