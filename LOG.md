# Attempt 054 — Waist surgery: global cut improvement of the dominant contraction

- **attempt**: 054
- **date**: 2026-07-24
- **kind**: draft
- **parents (machinery)**: 047/052 (engines), 026/031 (cut/boundary machinery
  from the certification arc), 033 (proved windows ≤16 cannot fix the waist)

## Hypothesis (pre-registered)

Pure tc is pinned by the single most expensive contraction — the tree's
WAIST, a bipartition (A, B) of all tensors. TreeSA's O(1) rewrites move one
leaf at a time and cannot jump between distinct good bipartitions of the
same size; 033 proved no ≤16-leaf window reaches the waist; 052/047 kick
uniformly rather than at the waist. WAIST SURGERY injects global
information exactly there: given an SA-refined incumbent, (1) extract the
waist bipartition (the argmax-cost node's leaf-set A vs complement B);
(2) improve THAT CUT globally, ignoring the current tree — bounded
Fiduccia–Mattheyses passes on the tensor graph (gain = change in summed
log2 dims of straddling labels; balance constrained to |A| ± slack),
optionally seeded by several boundary-BFS alternatives; (3) if a strictly
cheaper cut of comparable balance exists, REBUILD: cold-anneal a subtree
for A' and for B' separately (each side is a much smaller problem; open
indices = the new cut + external labels), join them, splice, and accept iff
global tc drops; (4) iterate on the new waist (which may move) until no
improving cut is found or budget ends. Interleave with the 052-style
span-gated ratchet as the base engine.

Claim: beat ≥1 of the current records — reg3_1000, ksg, sycamore_53_20_0,
surfacecode_d21 (values as of scoring time; check the leaderboard rows
printed by the orchestrator in the brief) — by > 0.05, median-of-3.

Falsification (equally valuable): FM/BFS cut search finds NO cheaper
balanced cut at the waist of frontier-quality incumbents on any instance —
i.e., "the frontier's waist is already a globally minimal balanced cut" —
a new certificate-flavored statement tying the optimizer arc to the
certification arc. Report per-iteration: waist cost, best alternative cut
found, accepted/rejected, tc trajectory.

## Constraints

- Contract `attempt <graph.json> <budget_ms> <out.json>`; eager atomic
  writes; single thread; no per-instance constants; relabeling-invariant;
  pure tc; LINEAR schedules for all anneals.
- The cut improvement must be genuinely global (operate on the tensor
  graph, not the tree); the rebuild must be exact about open labels
  (outside-occurrence counting) so the scorer agrees.
- Implementation in `omeco/examples/attempt.rs`; port engines from
  .worktrees/attempt-047 / attempt-052; library helpers as needed.

## Implementation (as built)

Single self-contained `omeco/examples/attempt.rs` (no library changes). Pipeline:

1. **SIMPLIFY** (ported from 039): rank-non-increasing fusion + splice-back.
2. **SEED**: deterministic + Boltzmann greedy portfolio (written immediately).
3. **BASE ENGINE**: the 052 span-gated basin-hopping ratchet — a warm coarse
   kick on a clone, then a cold span-gated cooling ladder (`S_top=ceil(n/30)`
   down to 2). Adopt iff the clone beats the incumbent (monotone `best_tc`).
4. **WAIST SURGERY** (the hypothesis), interleaved every outer iteration:
   - `extract_waist`: argmax-cost node's leaf set A (bipartition A|B). Node cost
     recomputed from `out_dims` via the exact `tcscrw` formula.
   - `fm_refine`: bounded Fiduccia–Mattheyses on the reduced tensor hypergraph.
     Gain = real reduction in summed log2 dims of straddling (non-output)
     labels; output labels are constant (always in the top-node output) and
     excluded from gain. Balance |A| ± 18%. Seeded from the current cut plus 4
     boundary-BFS alternatives (highest-degree start + 3 RNG starts).
   - If a strictly cheaper comparable-balance cut is found, `rebuild`: promote
     the cut to the tree TOP and cold-anneal a subtree for EACH side separately
     (open labels via outside-occurrence counting, so the top-node output =
     straddle ∪ iy exactly matches the scorer), join at the root, splice, and
     accept iff global tc drops. FIXED-work sub-anneals (3 V-cycles) — see
     "variance" below.
   - `separator_cliqueness`: Tamaki-style near-clique note logged per cut.

Eager atomic writes (tmp+rename), rate-limited + a final forced flush; single
large-stack worker (deep trees); fixed RNG seed; every knob a function of `n`.

## Results (90 s budget, local, scorer.py-verified)

**Three robust median-of-3 record beats + one single-run beat**, all driven by
accepted REBUILDs (waist surgery is the mechanism; the base ratchet alone
plateaus far higher, e.g. surfacecode ~49.6, ksg ~50):

| instance          | record    | my runs (tc)                    | median   | Δ vs record |
|-------------------|-----------|---------------------------------|----------|-------------|
| **surfacecode_d21** | 47.82378 | 47.466, 47.381, 47.408        | 47.408   | **−0.416 ✓**|
| **ksg**             | 37.08693 | 36.922, 36.385, 36.436        | 36.436   | **−0.651 ✓**|
| **rqc_97_m24**      | 106.46849| 105.924, 106.177, 106.251     | 106.177  | **−0.292 ✓**|
| nqueens_28 (2nd)  | 121.36787 | 121.002, 122.322, 125.000     | 122.322  | +0.954 (single-run 121.002 beats; too high-variance for a median) |
| sycamore_53_20_0  | 59.91142  | 60.112                          | —        | +0.201 miss |
| reg3_1000         | 131.15364 | 133.067                         | —        | +1.913 miss |

## Per-iteration surgery traces (verdict evidence)

`surfacecode_d21` iter 1–8 (first run, old time-boxed sub-anneal, 6 accepts):
```
iter=1 WAIST cost=48 best_alt=40 gap=8   -> REBUILD ACCEPT 49.090 (was 49.597)
iter=2 WAIST cost=45 best_alt=33 gap=12  -> REBUILD ACCEPT 48.797 (was 49.090)
iter=4 WAIST cost=45 best_alt=33 gap=12  -> REBUILD ACCEPT 48.703
iter=5 WAIST cost=45 best_alt=39 gap=6   -> REBUILD ACCEPT 48.182
iter=6 WAIST cost=45 best_alt=37 gap=8   -> REBUILD ACCEPT 47.391
iter=7 WAIST cost=43 best_alt=39 gap=4   -> REBUILD ACCEPT 47.381
```
`ksg` (gap 9–11 bits every call; most rebuilds rejected, a few accepted ratchet
the incumbent below record):
```
iter=1 WAIST cost=40 best_alt=30 gap=10  -> REBUILD ACCEPT 40.470 (was 42.049)
iter=2 WAIST cost=37 best_alt=26 gap=11  -> REBUILD reject 41.190 (was 37.963)
iter=5 WAIST cost=36 best_alt=26 gap=10  -> REBUILD reject 37.587 (was 37.075)
... final 36.436  (6 accepts across the run)
```

## Verdict (honest)

**The pre-registered hypothesis is SUPPORTED on structured/separable networks
and its FALSIFICATION side is itself disproved.**

1. **The frontier's waist is NEVER a globally minimal balanced cut.** Across all
   6 instances and every one of the hundreds of surgery calls, FM found a
   strictly cheaper comparable-balance cut than the SA-refined waist
   (`waist_min_hits=0`, `cheaper_cuts==calls` on every run). Gaps are large:
   3–12 bits on surfacecode, 9–11 bits on ksg. So the pre-registered
   certificate-flavored statement — "the frontier's waist is a globally minimal
   balanced cut" — is FALSE here; NO `WAIST-MIN` line was ever emitted. The
   waist cut is always improvable; the interesting content is whether improving
   it lowers *global* tc.

2. **Cut improvement lowers global tc on separable networks → new records.**
   Promoting the cheaper cut to the tree top and re-annealing both halves beats
   the incumbent whenever the network has genuine balanced separators:
   surfacecode_d21 (planar-ish, −0.42), ksg (−0.65), rqc_97_m24 (−0.29), and a
   single-run nqueens_28 (−0.37). These are 3 confirmed median beats over
   engines (047/039/ref) that had held these rows.

3. **On expanders it helps but cannot overcome intrinsic hardness.** On reg3
   (3-regular expander) FM still finds cheaper cuts and rebuilds ratchet the tc
   down hard (181→133), but the uniform high treewidth caps the result above the
   record — a cheaper top cut merely trades against more expensive internal
   nodes. sycamore's *simplified* network (n 3369→381) is likewise too dense for
   a decisive separator (misses by 0.20).

4. **The improved cuts are NOT clique separators** (`clique_frac` 0.01–0.05):
   they are genuine sparse balanced cuts, not Tamaki-style safe separators. The
   win comes from sparsity, not near-cliqueness.

5. **Variance note.** The first design time-boxed each side's sub-anneal, which
   desynced the RNG stream under wall-clock pressure and gave a 0.8-bit spread on
   surfacecode (one of three runs missed). Switching to FIXED-work sub-anneals (3
   V-cycles) made the improvement trajectory reproducible, collapsed the spread
   to ~0.08 bit (all 3 runs beat), AND — by making each rebuild cheaper — allowed
   far more surgery iterations, which is what pushed ksg from 38.85 to 36.44.
   nqueens keeps high variance (huge intermediates ⇒ only ~3 surgery calls fit),
   so its beat is single-run only.

**Bottom line:** waist surgery — a global FM cut improvement of the dominant
contraction, followed by a two-sided rebuild — is a real, transferable mechanism
that set 3 new records the pure-SA frontier could not reach, and it converts the
pre-registered falsification into a positive structural fact (the frontier waist
is always a strictly-improvable, non-clique balanced cut).
