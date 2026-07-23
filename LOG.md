# Attempt 023 — Hierarchical coarsen + exact super-network contraction tree

- **attempt:** 023
- **date:** 2026-07-23
- **kind:** draft
- **parent:** none

## Hypothesis

Hierarchical exactness escapes the local-search plateau. Coarsen the tensor
network into m ≈ 12–16 connected super-tensors (each cluster contracted
internally by a good heuristic order), solve the SUPER-network's contraction
tree EXACTLY via subset dynamic programming over the m super-nodes, then
expand by splicing cluster subtrees into the exact super-tree leaves. The
exact top-level topology can realize global structure that no sequence of
local rotations (TreeSA rules) reaches from heuristic seeds. Iterate over
different random coarsenings while budget remains, keeping the global best.

Both target instances are **pure size-2 graphs** (every index degree-2,
iy empty): reg3_250 is a 3-regular expander (hard — no small balanced cuts),
sycamore_m20 is a structured RQC proxy (may have exploitable community
structure). The validator enforces an **sc-cap** (reg3_250 ≤ 35,
sycamore_m20 ≤ 55), so the exact DP must minimize tc **subject to** every
intermediate staying under the cap.

## Expected evidence

- **Positive signal:** confirmed tc < 39.90 on reg3_250 or < 61.49 on
  sycamore_m20 (any gain > 0.05 over the plateau records tc=39.905 / 61.514).
- **Clean null (also decisive):** exact-top-level ties the plateau across many
  coarsenings — evidence FOR the plateau being globally near-optimal, and
  (for the expander) that no balanced coarsening yields cap-feasible cuts.

## Method

1. Safety net: greedy tree written immediately; a TreeSA doubling loop
   (ntrials=1, sc_target=cap, growing niters) as the quality floor. Anytime
   atomic writes, keep best feasible tc.
2. Coarsening: heavy-edge agglomeration on the tensor graph (merge the pair of
   clusters sharing the largest weighted boundary until m clusters remain),
   random tie-breaks per iteration for diversity.
3. Internal orders: GreedyMethod on each cluster's sub-einsum (output = its
   boundary label set) → cluster subtree, leaves remapped to original ids.
4. Exact super-solve: subset DP over the m super-nodes minimizing total FLOPs
   in log2 domain (log-sum-exp), pruning any subset whose output tensor
   exceeds the sc-cap. Because logsumexp is monotone in each argument and the
   cluster-internal costs are fixed once clustering is chosen, minimizing the
   super-level total flops exactly minimizes the combined tc for that
   clustering. m ≤ 16 keeps 3^m enumeration affordable single-threaded.
5. Expand + measure with omeco::contraction_complexity (same convention as the
   validator's scorer); keep global best across {fallback, hierarchical × many
   coarsenings}.

## Implementation / decisions

Single self-contained `omeco/examples/attempt.rs` (no library changes).

- **Instances are pure size-2 graphs** (verified: every index degree-2, iy
  empty). So sc(node) = #boundary edges, tc(node) = #edges in the union of the
  two children's boundaries. The whole problem is recursive graph partitioning.
- **Coarsening:** heavy-edge agglomeration — repeatedly merge the alive cluster
  pair with the largest weighted shared boundary (reservoir random tie-break),
  down to a random target m ∈ [10,16] per round for diversity.
- **Internal orders:** GreedyMethod per cluster (sub-einsum output = cluster
  boundary). Because greedy is not sc-aware and blew the cap on nearly every
  cluster (see below), a short `TreeSA(sc_target=cap, niters=30)` fallback runs
  whenever the greedy cluster subtree exceeds the cap.
- **Exact super-solve:** subset DP over the m super-nodes. dp[S] = min
  log2(total super-flops) via log-sum-exp over partitions (submask enumeration
  restricted to the lowbit half → each unordered split once, 3^m total). Every
  subset whose output (super-cut) exceeds the cap is pruned (dp=∞), so **all
  super-level intermediates are guaranteed ≤ cap by construction.** Node tc uses
  the identity tc = |out(L) ∪ out(R)| computed from precomputed per-subset
  out-bitsets. Correctness of minimizing only the super-level flops: log-sum-exp
  is monotone in each argument and the cluster-internal costs are fixed once the
  clustering is chosen, so it exactly minimizes the combined tc for that
  clustering. m capped at 16 (3^16 ≈ 43M, sub-second single-thread).
- **Splice + measure:** reconstruct the exact super-tree, splice remapped
  cluster subtrees into its leaves, measure with `contraction_complexity`
  (same convention as the validator scorer).
- **Budget:** hierarchical rounds for the first 40% of budget, then a
  `TreeSA(sc_target=cap)` doubling loop as the anytime quality floor for the
  rest. Greedy tree written immediately; `Best` prefers feasible (sc≤cap) then
  lower tc, atomic anytime writes. Debug counters gated behind `OMECO_DEBUG`.

## Results — clean NULL (decisive)

Exact top-level hierarchical **never beats the plateau** on either instance.

| instance | record tc | best feasible hierarchical tc | final emitted tc (floor) |
|---|---|---|---|
| reg3_250 (cap 35) | 39.905 | **40.30** (23/40 rounds feasible) | 39.931 |
| sycamore_m20 (cap 55) | 61.514 | **63.67** (2/19 rounds feasible) | 61.654 |

Two-stage finding:
1. With sc-unaware greedy internals, **0 of ~160–175 coarsenings produced a
   cap-feasible tree** — the exact super-level is always ≤ cap by construction,
   so the blow-up is purely cluster-internal. This alone shows heavy-edge
   clusters are hard subproblems on these graphs.
2. With sc-aware internal orders, feasible hierarchical trees appear but their
   tc is consistently **~0.4–2.3 above the TreeSA plateau**, never below.

Interpretation: the exact top-level minimizes tc *for a fixed coarsening*, but
the coarsening imposes a globally suboptimal recursive structure, and the
sc-cap forces feasible coarsenings toward unbalanced/high-tc cuts. On an
expander (reg3_250) there is no cap-feasible balanced cut for the exactness to
exploit; on the structured RQC proxy (sycamore_m20) most coarsenings are
outright infeasible. This is decisive evidence FOR the plateau being globally
near-optimal on these two instances — the untried "hierarchical exactness"
mechanism class does not escape it.

The emitted trees are the TreeSA floor (≈ plateau) since hierarchical is always
worse; the attempt is not expected to set a record.

## Precheck result

PASS. `validate --precheck` → `status=scored`, `precheck_chain_10: pass`,
exit 0, no errors (`/tmp/pc_023.json`). Build: `cargo build --release
--offline --example attempt -p omeco` clean.
