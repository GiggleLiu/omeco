# attempt-057: scalable quotient-graph VE seed

Parent: attempt-038 ("hyperedge-aware variable-elimination seeding").

## Hypothesis

attempt-038's VE seed WINS on the dense hypergraph `dbn_13` (44 labels) but its
order computation does not scale: the min-cost score recomputes the log2-weighted
union of a label's whole holder group on every heap push (O(degree x live-set)),
so at ~4k+ labels the heap "freezes" and on the UAI relational instances
(10k-13k labels, 30k-70k tensors) VE contributes nothing — every method sits at
tc~202 while the known treewidth is ~100.

Hypothesis: replace the *order computation* with a scalable AMD-style elimination
ordering (quotient graph / element absorption, weighted min-degree, lazy heap),
keep attempt-038's topology-build + annealing untouched, and the seed will reach
the ~width-101 elimination at scale while not regressing dbn/linkage.

## What changed vs attempt-038

1. **New `HyperGraph::amd_order`** — a scalable elimination-order generator over
   the label primal graph using a QUOTIENT GRAPH with element absorption:
   - Original tensors ARE the initial elements (each tensor's label set is a
     clique); a variable's neighborhood is the union of its incident elements'
     boundaries. Eliminating a variable creates ONE new element whose boundary is
     that neighborhood and absorbs the variable's old elements. Fill edges are
     never materialized (only per-element boundary lists, O(width) not O(width^2)).
   - WEIGHTED min-degree score = summed log2 dims of the created clique boundary
     (reduces to neighbor counts for binary labels; handles non-binary cards via
     `log2` weights — verified on linkage_15).
   - Lazy binary heap keyed `(Reverse(weighted-degree), Reverse(id))`: a variable
     is rescored only when popped; a stale pop (recomputed degree != key) is
     recomputed and re-pushed. Neighbors of an eliminated variable are recomputed
     and pushed fresh. Deterministic tie-break by interned label id.
   - Budget guard: aborts (returns `None`) if it cannot finish within
     `min(10% budget, 10s)`; caller then falls through to greedy.
   - Returns `(order, width)` where width = max weighted clique (~ seed tc).
   Helpers `wdeg` / `boundary_of` do the O(scanned-size) neighborhood scan with a
   timestamp `mark` array for dedup, pruning absorbed elements in passing.

2. **New `HyperGraph::build_topo_from_order`** — replays a precomputed order
   through attempt-038's UNCHANGED holder-group merge machinery (`merge_group`,
   dropped-label logic, remaining-merge) to produce the `TopoTree`, then the
   existing `build_nested` (exact outside-occurrence counting) yields the
   `NestedEinsum`. So the measured tc equals the scored tc, same as attempt-038.

3. **Pipeline reorder (necessary, see Surprise):** the AMD seed is computed and
   EMITTED FIRST (eager valid answer at any scale), and the library greedy is
   GATED to `<= 8000` tensors (or when AMD aborts). On large instances greedy is
   skipped, `tc_greedy` stays +inf, so the router treats the hyperedge VE seed as
   fitting and hands off to the warm (per-sweep interruptible) anneal — correct
   for the large hyperedge instances, which are exactly where greedy is skipped.
   The dynamic attempt-038 VE (min-degree + min-cost) still RACES on small label
   graphs (`<= 3000` labels): it is the dbn winner and does not freeze there.

Everything else (simplification/interning front-end, eager atomic emission, warm
anneal, treesa_doubling fallback, CLI contract) is unchanged.

## Surprise

The stated freeze (the VE min-cost *rescore*) is real, but the ACTUAL scaling
wall on the relational instances turned out to be attempt-038's UPFRONT library
greedy: `optimize_code(GreedyMethod::default())` costs **~84 s at 30,400 tensors**
(measured: `t_build=15ms`, `t_greedy_opt=83845ms`). Because attempt-038 seeds
with greedy FIRST, that 84 s blocked the whole budget before VE ever ran — and at
the 60 s relational_2 budget it would never finish (no output at all). The AMD
order itself is ~**10 ms** for all 10,200 eliminations at 30k tensors. Fix: emit
the fast AMD seed first, gate the slow greedy. This is a pipeline change slightly
beyond "just the order computation", but it is required to hit the goal and the
60 s budget.

Deviations from spec:
- **Supervariable (hashing) detection NOT implemented.** Not needed: plain
  weighted min-degree with element absorption already reaches width 101 (=
  optimal treewidth; Tamaki bag 101) in ~10 ms at 30k tensors. Left as a possible
  future refinement.
- **min-fill NOT added to AMD.** Weighted min-degree already hits the optimal
  width; the dynamic min-cost VE still races on small graphs for quality.

## Measured smoke tests (this Mac, release)

| instance (budget)   | tensors | labels | seed time | ve_width | seed tc  | final tc | sc  | note |
|---------------------|---------|--------|-----------|----------|----------|----------|-----|------|
| relational_4 (120s) | 30,400  | 10,200 | 72 ms     | 101.00   | 108.9687 | 108.9687 | 100 | greedy skipped; target tc<=~110 (opt tw 100) |
| relational_2 (60s)  | 38,750  | 13,000 | 17 ms     | 3.00     | 17.4395  | 17.4395  | 3   | greedy skipped; target tc<=17.6 |
| DBN_13 (60s)        | 572     | 44     | ~1 ms     | 23.00    | 28.7944  | 28.7530  | 23  | REGRESSION bar <=29.5; attempt-038 ~29.0 -> improved |
| linkage_15 (60s)    | 2,304   | 1,152  | ~1 ms     | -        | -        | 30.4277  | 23  | non-binary cards OK; ~31, no crash |

- relational_4/2 seeds are emitted in tens of milliseconds (vs 84 s before), so
  the guard (<10% budget or <10s) is satisfied with huge margin; extrapolates
  comfortably to the 70k-tensor huawei sizes.
- Deterministic: two relational_4 runs gave identical tc 108.96866679319521.
- All builds: `cargo build --release`, `cargo clippy` (no warnings),
  `cargo fmt --check` all clean.
