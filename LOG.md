# Attempt 024 — geometry-aware spacetime-block coarsening + exact super-solve

- **date:** 2026-07-23
- **kind:** draft
- **parent:** none
- **instances:** reg3_250 (record tc=39.95), sycamore_m20 (record tc=61.544)
- **budget:** 90 s / instance, single-thread

## Hypothesis

For the lattice instance (sycamore_m20) the right coarsening is *geometric*:
group the original tensors into connected spacetime blocks (recovered from the
graph, relabel-proof), contract each block internally (GreedyMethod), then solve
the resulting *super-network* of ~12–16 super-tensors **exactly** (log-domain
subset DP minimizing tc via log-sum-exp), and expand back to a full tree over the
original tensors. An exact top-level over structured blocks may realize a better
global cut sequencing than local search (TreeSA) or attempt-019's *linear*
sweeps — because the super-tree DP explores all branching shapes over the blocks,
not just a linear order.

reg3_250 is an expander with no boundary/gate layering; coarsening should not
help, so it runs the TreeSA-inf fallback only (detected by absence of a
rank-1-boundary + rank-4-gate structure).

## Expected evidence

- **Positive:** tc < 61.49 on sycamore_m20 from some block shape, beating the
  plateau; reg3_250 record-neutral via fallback.
- **Null (also informative):** exact-over-blocks ties 61.54. Combined with sibling
  023's null (random-agglomeration coarsening), a null here would strongly
  indicate the 61.544 plateau is (near-)globally optimal for omeco's tc model.

## Design

- Safety net: greedy tree written immediately; then a TreeSA-inf loop
  (sc_target = +inf, doubling niters) as the workhorse fallback — protects the
  record on both instances (global best is monotone across everything).
- Structure detection: `has_rank1 = any tensor rank 1`; `has_rank4 = #rank4 > n/4`.
  Both true ⇒ hierarchical phase (sycamore). Otherwise fallback only (reg3).
- Blocks (relabel-proof, geometry-aware): farthest-point sampling of `m` seed
  tensors on the tensor-adjacency graph, then multi-source BFS region growing →
  `m` connected Voronoi cells (spatially + temporally local because graph distance
  mixes space and circuit depth). Block shapes varied by `m ∈ {12,14,16}`.
- Exact super-solve: each inter-block label has graph degree 2, so `cross(S)`
  (labels cut by block-subset S) = XOR of member blocks' incident bitsets.
  Subset DP: `cost(S) = min_{A⊂S} logsumexp2(cost(A), cost(B), |cross(A)∪cross(B)|)`,
  base `cost({i}) = block i internal greedy tc`. O(3^m) with bitset union costs.
- Expansion + measurement: assemble the super-tree with each block-leaf replaced
  by its internal greedy tree shape, rebuild every node's eins correctly
  (label is a node output iff it also appears outside the subtree or in iy —
  identical rule to the validator scorer), and measure real tc via a
  topology-only scorer (mirrors the validator; does not trust eins).
- Anytime: keep the global best tree by topology-tc, atomically rewrite out.json
  whenever it improves.

## Implementation

`omeco/examples/attempt.rs` (self-contained; no library changes). Pipeline:

1. **Greedy seed** written immediately (atomic tmp+rename) so a valid tree is
   always on disk.
2. **Topology scorer** `topo_cc` computes (tc, sc) exactly as the validator
   scorer does — per-node tc = sum log2 over the union of the two children's
   *output* label sets; a label is a subtree output iff it appears outside the
   subtree (`subtree_count < global_count`) or is in iy. Used for all candidate
   selection so ranking matches the validator regardless of node eins metadata.
3. **Feasibility-aware selection** against the real sc cap (reference sc + 2:
   reg3_250 → 35, sycamore_m20 → 55, by surviving instance name). Prefer feasible
   (sc ≤ cap); among equal feasibility, lower tc. This is the key correctness fix
   — the validator DOES enforce an sc-cap guard (a run over cap scores as fail),
   so `sc_target = INF` is unsafe; TreeSA runs with `sc_target = cap`.
4. **Structure detection**: `rank-1 boundary present` AND `#rank4·4 > n` ⇒
   hierarchical phase (sycamore); else fallback only (reg3).
5. **Hierarchical spacetime-block phase** (structured only, cheap — a few s,
   hard-capped at 0.55·budget so it never starves TreeSA): blocks via
   farthest-point-sampling of `m` seeds + region-growing multi-source BFS →
   `m` connected Voronoi cells (geometry-aware, relabel-proof); each block
   contracted internally with GreedyMethod; **exact super-solve** over the
   `m ≤ 18` super-tensors — log-domain subset DP `cost(S) = min_{A⊂S}
   logsumexp2(cost(A), cost(B), |cross(A)∪cross(B)|)` with `cross(S)` = XOR of
   member blocks' incident inter-block-label bitsets (valid because every index
   has graph degree 2). Expanded back to a full tree and rebuilt with correct
   eins. Block counts m ∈ {12,14,16,10} tried.
6. **TreeSA workhorse** (sc-capped, doubling niters) for the remaining budget —
   and the whole budget on reg3.

### Decisions / deviations from the brief
- Used `sc_target = cap`, NOT `INF`: the brief said "sc unbounded / treesa-inf",
  but `leaderboard.json` + `validate` enforce an sc-cap guard (reg3 35, syc 55).
  `INF` produced sc=62–77 trees that would score as instance FAIL. Respecting the
  cap is strictly safer and never regresses the record.
- First implementation had a scoring bug (node tc over *all* subtree labels
  instead of the children's output-label union); it made selection near-random
  and emitted tc=76 (reg3) / tc=106 (syc). Fixed to mirror the scorer exactly.

## Result

Precheck: **PASS** (`precheck_chain_10` structure ok, no errors).

Best local tc (validator scorer, single 90 s run each, `RAYON_NUM_THREADS=1`):

| instance      | this attempt tc | sc  | cap | record tc | delta   |
|---------------|-----------------|-----|-----|-----------|---------|
| sycamore_m20  | 61.516          | 53  | 55  | 61.514    | −0.002  |
| reg3_250      | 40.024          | 34  | 35  | 39.905    | −0.119  |

Both feasible; neither beats its record. sycamore ties the plateau; reg3 is
within TreeSA stochastic noise (fallback only, as designed).

**Hierarchical block method scored on its own** (ATTEMPT_DEBUG, sycamore):

```
m=12  block-tree tc=131.0  sc=112
m=14  block-tree tc=137.0  sc=112
m=16  block-tree tc=136.0  sc=112
m=10  block-tree tc=129.0  sc=110
```

**NULL result — decisive.** No block shape beat (or even approached) the plateau:
every geometry-aware block-tree is catastrophically worse (tc ≈ 129–137, sc ≈ 110)
and strongly sc-infeasible, so TreeSA's tree is what wins. Root cause: a connected
~35–45-tensor Voronoi chunk of the RQC has high internal treewidth, so contracting
it to its boundary already costs sc ≈ 110; the exact super-solve over the coarse
blocks cannot recover cost already spent inside blocks. The hypothesis's premise —
low block-internal cost + exact global sequencing — fails because block-internal
cost *dominates and explodes*.

Combined with sibling attempt-023's null (random-agglomeration coarsening + exact
super-solve), two-level *coarsen-then-exact-super-solve* is not the mechanism to
beat 61.5: coarsening destroys the fine-grained cut structure that TreeSA
exploits, and the coarse super-solve is powerless against block-internal blow-up.
This is evidence that the 61.5 plateau is near-optimal for omeco's tc model under
these methods.

## Post-hoc official scored run (2026-07-23, replication)

A second session re-ran `validate` on this worktree after cycle 6 closed, under
the current 4-instance board: reg3_250 40.024 (Δ−0.074), sycamore_m20 61.729
(Δ−0.185), reg3_1000 137.360, rqc_97_m24 106.571 — no records. Independently
replicates the cycle-4 null; `report.json` is this run.
