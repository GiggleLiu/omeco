# attempt 053

- date: 2026-07-24
- kind: improvement
- parent: 038

## Hypothesis (pre-registered before any implementation)

attempt-038's variable-elimination (VE) seed set the `dbn_13` record (28.79
seed, ~29.00 scored) on a 44-label hypergraph, but on `nqueens_28` (4086 labels)
the complete min-cost VE order both (i) does not finish inside the time box and
(ii) when it does step, produces a terrible seed (tc_ve = 714 vs tc_greedy =
384) — because nqueens has large treewidth and unbounded min-cost VE walks
straight into the treewidth blow-up, creating ever-larger intermediate factors.

**Mechanism claim:** the win scales via a BOUNDED / cheap-first VE. Instead of
eliminating every label (which forces the blow-up), peel only the CHEAP labels
— those whose elimination creates a factor no larger than the current space
frontier (the tree-like periphery: arity-1 tensors, degree-2 labels, and other
no-regret eliminations). Stop before the hard core. This:

  (a) bounds each elimination to near-O(holders · avg_degree) and never grows
      the max live tensor (no treewidth blow-up, so VE is fast even at 4k
      labels);
  (b) hands the DEFERRED residual network (the expensive hard-core super-tensors
      only, a much smaller problem) to the proven library TreeSA, which anneals
      a few-hundred-tensor residual far better than the full 4252-tensor graph;
  (c) splices each super-tensor's peel subtree back under the residual tree; the
      scorer recomputes tc from topology alone (verified in scorer.py — node
      einsum metadata is ignored), so the spliced tree scores exactly as its
      topology deserves.

The full (unbounded) VE seed is retained as a separate candidate — it is the
`dbn_13` winner (dense small hypergraph where peeling reduces little). A full
library-TreeSA anytime run is retained as a safety candidate. Best-by-tc wins.

**Claim:** beat the `nqueens_28` record (121.37, median-of-3) by > 0.05 via
peel+residual-TreeSA (one of 038's fallback runs already reached ~120, so the
target is beatable and a smaller residual should reach it more reliably), OR
beat `dbn_13`'s record (29.00) further. Secondary targets: `qft_27` (29.61),
`rqc_97_m24` (106.47).

**Falsification:** cheap-first-peel + residual-TreeSA does NOT beat plain
full-graph TreeSA at equal budget on the large-label instances → the VE
mechanism is intrinsically small-hypergraph and peeling buys nothing over
generic SA. Report VE/peel time, residual-network size (# super-tensors, #
live labels), seed tc vs greedy tc, and residual-TreeSA tc vs full-TreeSA tc
(the attribution).

## Design

- Port attempt-038's self-contained `HyperGraph` (dense-interned labels, per-leaf
  id sets, hyperedge degrees) + `TopoTree` + `build_inner` (exact
  outside-occurrence NestedEinsum emission) into this attempt's `attempt.rs`.
  No warm-anneal machinery is ported: the ~0.04 refinement it added on dbn is
  not needed to beat the record, and the residual play uses library TreeSA.
- New `peel` method: lazy min-cost bucket queue over labels; eliminate a label
  only if the factor it creates has cost <= the running space frontier (a
  quantile of live tensor costs, capped so the max live tensor never grows).
  Produces (i) a partition of the original leaves into super-tensors, each
  carrying a `TopoTree` and a live-label set, and (ii) leaves the hard core
  un-eliminated.
- Residual driver: build an EinCode from super-tensor live sets, run anytime
  doubling library TreeSA on it, convert its NestedEinsum topology to a TopoTree
  and splice each residual leaf -> its super-tensor subtree, then `build_inner`
  the whole thing. Best-by-tc kept, atomic writes.
- Candidates kept in one best-so-far: greedy (immediate fallback), full VE seed
  (dbn winner), peel+residual-TreeSA (nqueens play), full-graph TreeSA doubling
  (safety). Pure-tc, single-threaded (one large-stack worker; main blocks on
  join), no per-instance constants, relabeling-robust, atomic best-so-far.

## Results

All runs single-threaded (`RAYON_NUM_THREADS=1`), scored by the independent
`research/validator/scorer.py`. Confirmed scorer.py uses tree TOPOLOGY only
(node einsum metadata ignored), so the spliced-topology emission scores exactly
as its topology deserves.

### dbn_13 (record 29.00) — WIN, deterministic

| stage | tc |
|-------|----|
| greedy seed | 44.40 – 44.68 |
| **full VE seed (min-cost)** | **28.7944** |
| full TreeSA (9 rounds, 90 s) | no improvement |

Scored tc = **28.7944** at both 8 s and 90 s (deterministic — the VE seed, jitter
0). Beats the 29.00 record by **0.21** (> 0.05). This is exactly the parent
(038) VE-seed win, reproduced without the warm-anneal machinery (the ~0.04 SA
refinement was not needed to beat the record). Full TreeSA never touches 28.79 —
the win is purely the variable-elimination order, as in 038.

### nqueens_28 (record 121.37, median-of-3) — NOT beaten

| stage | tc |
|-------|----|
| greedy seed | 365 – 410 (randomized) |
| VE seed (unbounded) | 719 – 738 (discarded — the treewidth blow-up) |
| **full TreeSA (auto), 90 s, single run** | **125.00** |

The full-graph TreeSA fallback reached 125.00 in one 90 s run — above the 121.37
record, consistent with 038's finding that the record is only touched on lucky,
high-variance runs (038 saw 120–153). Median-of-3 would not beat 121.37 by 0.05.

### Peel attribution (MODE=peel, cap sweep, 25 s each) vs full baseline

| lane | residual tensors | residual labels | peel_ms | final tc (25 s) |
|------|------------------|-----------------|---------|-----------------|
| **full TreeSA baseline** | — | — | — | **134.0** |
| peel cap=10 | 1438 | 1275 | 22 | 188.3 |
| peel cap=20 | 1038 | 875 | 37 | 336.7 |
| peel cap=30 | 948 | 785 | 201 | 410.0 (never beat greedy) |
| peel cap=45/60 | 947–948 | 784–785 | 57–188 | >= greedy |

## Verdict — hypothesis FALSIFIED (bounded peel does not scale the VE win)

- **The 038 bottleneck was correctly diagnosed and half-fixed.** Unbounded VE on
  nqueens both (i) blows up (tc 719) and (ii) is slow. Bounded cheap-first peel
  FIXES the speed: peeling completes in **22–260 ms** even at 4086 labels (near
  O(holders·degree), never growing the max factor). So mechanism part (a) works.
- **But the residual handoff LOSES at every cap.** peel+residual-TreeSA is
  strictly worse than full-graph TreeSA at equal budget (188–410 vs 134). Lower
  caps shrink the residual but the fixed peel boundaries carry a bad periphery
  order; higher caps leave the residual ~=full but with frozen boundaries. There
  is no cap where separating periphery from core helps.
- **Attribution / mechanism boundary.** Good orders on a large-treewidth
  partition function (nqueens) INTERLEAVE peripheral and core contractions; a
  variable-elimination peel commits to fixed super-tensor boundaries and removes
  exactly that freedom, so TreeSA on the residual can never recover the
  interleaved optimum. The VE mechanism is therefore **intrinsically
  small-hypergraph** — it wins on dbn (44 dense labels, no periphery to mis-cut)
  and is dominated by direct full-graph TreeSA the moment the instance has a
  large treewidth with peripheral structure. This is precisely the falsification
  the pre-registration named ("hierarchical VE seeds still lose to greedy/TreeSA
  on the large-label instances → the VE mechanism is intrinsically
  small-hypergraph").

Claim status: **dbn_13 beaten by 0.21 (deterministic, > 0.05)** — but this is the
inherited 038 win, not a new improvement. **nqueens_28 NOT beaten** (125.00
single run; the bounded-peel scaling idea does not deliver). Net: the scaling
hypothesis is falsified; the negative result (peel completes fast yet loses
because it freezes the periphery/core split) is the contribution.

### Production routing

`auto` routes to full-graph TreeSA anytime doubling (the proven strong lane),
keeping the deterministic full VE seed (dbn winner) and greedy fallback. The peel
lane is retained only behind `MODE=peel` for reproducing the attribution above;
it is never used by default (no per-instance constants; `MODE`/`PEEL_CAP` default
off). Required local runs done: dbn_13 @90 s = 28.7944, nqueens_28 @90 s = 125.00,
both verified with scorer.py. clippy `-D warnings` clean, rustfmt clean, offline
build OK, validator precheck = pass.

