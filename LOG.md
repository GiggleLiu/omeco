# Attempt 026

- attempt: 026
- date: 2026-07-23
- kind: draft
- parent: none

## Hypothesis

The sum-form isoperimetric-profile bound certifies most of the gap between the
width (max-form) bound and the frontier `tc`. Concretely, for a binary
contraction tree over a closed dim-2 network, every internal node v with
leaf-set S_v costs `2^{cost_v} ≥ 2^{|∂S_v|}`, so
`tc ≥ log2 Σ_v 2^{|∂S_v|} ≥ log2 F(n)` where `F(n) = min over binary trees of
Σ_v 2^{b(|S_v|)}` and `b(k)` is the isoperimetric profile — a SUM over nodes,
expected to beat the single-term max-form cap (max balanced `b(k)`).

## Expected evidence

LB(tc) numbers for both instances in certified (spectral) and
high-confidence (empirical) variants; the gap-decomposition table
(max-form LB | sum-form spectral LB | sum-form empirical LB | frontier tc);
scrupulous certified-vs-empirical labeling.

## What was done

- **Theorem 1 formalized and proved** (`PROFILE_BOUND.md §1`): the per-node
  identity `cost_v = |∂S_v| + |E(A_v,B_v)| ≥ |∂S_v|` (network closed ⇒
  labels(A)=∂A; ∂A∪∂B = ∂S ⊍ E(A,B)); the global chain
  `tc ≥ log2 Σ 2^{|∂S_v|} ≥ log2 Σ 2^{b(|S_v|)} ≥ log2 F(n)`; the DP
  `f(k)=2^{b(k)}+min_j[f(j)+f(k−j)]` as the exact shape-minimization; and the
  relaxation lemma (any `b_lo ≤ b` stays a valid LB). DP verified against
  brute force (n=5,6,7) and closed forms (cycle, all-zero).
- **Certified spectral profile**: `b_spec(k)=λ₂ k(n−k)/n ≤ b(k)`, λ₂ exact from
  the integer Laplacian; inequality re-verified on 2000 random subsets/graph
  (0 violations). λ₂ = 0.203598 (reg3), 0.046181 (sycamore).
- **High-confidence empirical profile**: Fiedler sweep + min-boundary region
  growing (560 seeds) + harvested frontier-tree cuts + FM swap refinement,
  window-min for conservatism. Every cut double-count verified genuine.
- Ran `treesa_tuned` 90 s on each instance; parsed the emitted tree; computed
  the per-node `|∂S_v|`, `cost_v`, `E(A,B)` multisets and the tree's own
  sum-form value.

## Result — hypothesis NOT supported (informative negative)

Gap table (log2 bits; CERT = certified, HC = high-confidence):

| instance | max-form spec (CERT) | max-form emp (HC) | sum-form spec (CERT) | sum-form emp (HC) | frontier sc | frontier tc |
|---|---|---|---|---|---|---|
| reg3_250     | 11.36 | 30 | 13.14 | 30.81 | 34 | 39.95 |
| sycamore_m20 |  5.76 | 47 |  9.84 | 47.17 | 53 | 61.54 |

The sum-form DP bound exceeds the balanced-separator max-form by **< 1 bit**
(30.81 vs 30; 47.17 vs 47) and lies **below** the frontier — for sycamore even
below the known certified carving bound of 53. **Mechanism** (measured): with
the min-over-all-subsets profile the DP picks a maximally skewed tree with a
*single* balanced node, so `log2 Σ 2^{b} ≈ max term + O(1)` (largest term = 57%
of the sum for reg3, 89% for sycamore). The sum-form collapses to the max-form.

The gap is recovered only by the sum-form on the frontier tree's **own nested
(balanced) profile**: 37.34 (reg3) and 59.24 (sycamore), i.e. 93% / 96% of the
tree's `tc`, residual = the Σ 2^{|E(A,B)|} slack (2.68 / 2.37 bits). That is a
per-tree LB, not a bound on the optimum.

## Takeaway for re-planning

Theorem 1 is a correct, cheap, certified LB but too loose to matter
(dominated by one term). A publishable bound needs to forbid tree skew:
a **balance-constrained profile DP** (splits with j ≥ αk) or a carving-/
branch-width argument — the direction that already gives sycamore's 53.

Deliverables: `PROFILE_BOUND.md`, `scripts/*.py`, `data/results.json`,
`data/{reg3_250,sycamore_m20}_profile.csv`, `data/profiles.png`.

## Outcome (measurement attempt — no scored run by design)
- Theorem 1 proven and validated; unconstrained sum-form bound collapses
  to max-form (<1 bit above balanced-separator on both instances).
- Fix identified: balance-constrained (dyadic-window) profile DP.
- Deliverables: PROFILE_BOUND.md, scripts/, data/ in this worktree.
