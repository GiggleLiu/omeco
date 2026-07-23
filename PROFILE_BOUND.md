# The isoperimetric-profile (sum-form) lower bound on contraction cost `tc`

Attempt-026 · 2026-07-23 · THEORY + MEASUREMENT

Instances: `reg3_250` (3-regular, n=250 tensors, m=372 indices, dim 2) and
`sycamore_m20` (n=561 tensors, m=963 indices, dim 2). Both are **closed**
tensor networks (empty output, every index on exactly two tensors), hence
ordinary graphs G=(V,E) with |V|=n, |E|=m. All bond dimensions are 2, so a
contraction's log2-cost equals the number of distinct indices it ranges over,
and a vertex subset's boundary |∂S| (edges leaving S) equals, in bits, the
width of the intermediate tensor for the leaf-set S.

---

## 1. Theorem 1 (statement)

Let T be a binary contraction tree whose leaves are the tensors V(G). Each
internal node v has a **leaf-set** S_v ⊆ V (the tensors under v); its two
children carry leaf-sets A_v, B_v with A_v ⊍ B_v = S_v. Write
∂S = { edges of G with exactly one endpoint in S } and
E(A,B) = { edges with one endpoint in A and one in B }.

**(i) Per-node identity / inequality.**
> cost_v = | labels(A_v) ∪ labels(B_v) | = |∂S_v| + |E(A_v,B_v)| ≥ |∂S_v|.

**(ii) Global bound.** With the **isoperimetric profile**
b(k) = min_{|S|=k} |∂S| and
F(n) = min over binary trees on n leaves of Σ_v 2^{b(|S_v|)},

> tc(T) = log2 Σ_v 2^{cost_v} ≥ log2 Σ_v 2^{|∂S_v|} ≥ log2 Σ_v 2^{b(|S_v|)} ≥ log2 F(n).

Since this holds for every T, **tc_opt ≥ log2 F(n) =: LB(tc).** F(n) is computed
by the DP
> f(1) = 0,  f(k) = 2^{b(k)} + min_{1 ≤ j ≤ ⌊k/2⌋} [ f(j) + f(k−j) ],  F(n) = f(n).

**(iii) Relaxation.** Replacing b(k) by any lower bound b_lo(k) ≤ b(k) keeps
log2 F_lo(n) a valid (weaker) lower bound. Replacing b(k) by any value keeps
the DP a valid *definition*; only b_lo ≤ b guarantees certification.

### Proof

**(i)** The network is closed with empty output, so the tensor produced by
contracting the subtree A has exactly the indices shared between A and its
complement — i.e. labels(A) = ∂A, and |labels(A)| = |∂A| bits because every
bond has dimension 2. For disjoint A, B with S = A ∪ B, classify each edge by
its endpoints:
* both endpoints in S ⇒ one in A, one in B (A,B partition S) ⇒ edge ∈ E(A,B),
  and it lies in ∂A ∩ ∂B;
* exactly one endpoint in S ⇒ edge ∈ ∂S, and it lies in exactly one of ∂A, ∂B;
* no endpoint in S ⇒ in neither.

Hence ∂A ∪ ∂B = ∂S ⊍ E(A,B) (a disjoint union), so
|labels(A) ∪ labels(B)| = |∂A ∪ ∂B| = |∂S| + |E(A,B)| ≥ |∂S|. The contraction
at v ranges over all indices of A and B, so cost_v = |∂A ∪ ∂B|. ∎
*(This identity cost_v = |∂S_v| + |E(A_v,B_v)| is checked numerically on every
node of both frontier trees — 249 and 560 nodes — in `scripts/tree_profile.py`.)*

**(ii)** The first inequality is termwise (2^{cost_v} ≥ 2^{|∂S_v|} by (i)) under
the increasing map log2 Σ 2^{(·)}. The second is termwise from b(|S_v|) ≤ |∂S_v|
(definition of the minimum). For the third, the multiset of internal-node sizes
{|S_v|} of any binary tree obeys exactly the DP recursion (a size-k node splits
into sizes j and k−j, 1 ≤ j ≤ k−1, both ≥ 1; leaves cost 0). Therefore
Σ_v 2^{b(|S_v|)} is one feasible value of the shape-minimization and is
≥ F(n) = its minimum; the DP computes that minimum by the standard integer
partition recursion. ∎

**(iii)** b_lo ≤ b termwise ⇒ f_lo(k) ≤ f(k) by induction on k ⇒
log2 F_lo(n) ≤ log2 F(n) ≤ tc_opt. ∎

**DP validation.** `scripts/test_sanity.py` checks the DP against brute-force
minimization over *all* binary-tree shapes for n = 5,6,7 (random profiles),
and against closed forms for the cycle C_n (b(k)=2, F = 4(n−2)+1) and the
all-zero profile (F = n−1). All pass.

---

## 2. Computing b(k): two instantiations

b(k) is NP-hard (min-boundary at fixed size / min-bisection at k=n/2). We use
two lower/estimate families.

### 2a. Certified spectral profile b_spec(k)  — RIGOROUS

For the graph Laplacian L with algebraic connectivity λ₂ (second-smallest
eigenvalue), and any S with |S| = k,
> |∂S| = 1_S^T L 1_S = (1_S − (k/n)1)^T L (1_S − (k/n)1) ≥ λ₂ · ‖1_S − (k/n)1‖² = λ₂ · k(n−k)/n,

because 1_S − (k/n)1 ⟂ 1 (the λ₁-eigenvector) and the Rayleigh quotient on 1^⊥
is ≥ λ₂. Thus **b_spec(k) = λ₂·k(n−k)/n ≤ b(k)** is certified. λ₂ is computed
as an exact eigenvalue of the exact integer Laplacian by dense symmetric
`numpy.linalg.eigh` (n ≤ 561, fully converged; λ₁ = 0 to 1e-15, graph
connected). The inequality is empirically re-verified on 2000 random subsets
per graph with **0 violations** (`test_sanity.py::test_spectral_inequality`).

| instance | n | m | λ₁ (≈0) | **λ₂ (certified)** |
|---|---|---|---|---|
| reg3_250 | 250 | 372 | −3.4e-17 | **0.203598** |
| sycamore_m20 | 561 | 963 | 8.7e-16 | **0.046181** |

### 2b. Empirical structural profile b_emp(k)  — HIGH-CONFIDENCE, NOT CERTIFIED

Heuristic cuts are *achievable*, so any cut found gives b_emp(k) ≥ b_true(k)
(an upper bound on the true minimum). We drive b_emp down toward b_true with a
union of sources, so the estimate leans toward the true profile:
1. **Fiedler-vector sweep** (prefix sets ordered by the λ₂ eigenvector);
2. **Min-boundary region growing** from 560 random + low-degree seeds
   (greedy: add the frontier vertex minimizing Δ|∂S| = deg(v) − 2·|N(v)∩S|);
3. **Harvested frontier-tree cuts** — every internal node of the tuned TreeSA
   tree is a genuine nested cut; its boundary is a real achievable value at its
   size (these supply the good *balanced* cuts);
4. **Size-preserving steepest-descent swap refinement** (FM-style) seeded from
   3 and from grid growths.

We take the elementwise minimum, symmetrize b(k)=b(n−k), and apply a
**conservative window-min** (b_emp(k) ← min_{|j−k|≤2} b_emp(j)) so the profile
leans low (toward b_true) rather than high. Every cut used is verified genuine
by an independent double-count (edge-iteration vs. adjacency-matvec
x^T(D−A)x) in `scripts/verify_cuts.py`; all agree. Profiles are in
`data/{reg3_250,sycamore_m20}_profile.csv` and `data/profiles.png`.

Because b_emp is an *upper* bound on b_true, the resulting DP value is our
**best estimate of the true value of the bound LB(tc)**, labeled
high-confidence but **not certified** (a proof would require optimal cuts).

---

## 3. Results

### 3a. λ₂, bound values

| quantity | reg3_250 | sycamore_m20 |
|---|---|---|
| **CERTIFIED** spectral sum-form LB(tc) | **13.139** | **9.842** |
| **CERTIFIED** spectral max-form (balanced-sep) | 11.356 | 5.757 |
| **HIGH-CONF** empirical sum-form LB(tc) | **30.811** | **47.170** |
| **HIGH-CONF** empirical max-form (balanced-sep) | 30 (k=84) | 47 (k=206) |
| empirical profile peak max_k b_emp(k) | 34 | 53 |
| frontier tree tc / sc (this run's tuned tree) | 40.024 / 37 | 61.613 / 57 |
| **sum-form on the frontier tree's OWN nested profile** | **37.343** | **59.242** |
| frontier tc (target) / frontier sc (target) | 39.95 / 34 | 61.544 / 53 |

"balanced-sep" max-form = min_{n/3 ≤ k ≤ 2n/3} b(k): every binary tree has an
internal node with size in [n/3, 2n/3], so sc ≥ this. It is the strongest
*guaranteed* single-term (max-form) bound derivable from the profile.

### 3b. The gap table (key table)

log2-scale bits. CERT = certified rigorous; HC = high-confidence structural.

| instance | max-form spectral (CERT) | max-form empirical (HC) | **sum-form spectral (CERT)** | **sum-form empirical (HC)** | frontier sc | frontier tc |
|---|---|---|---|---|---|---|
| reg3_250     | 11.36 |  30 | **13.14** | **30.81** | 34 | 39.95 |
| sycamore_m20 |  5.76 |  47 |  **9.84** | **47.17** | 53 | 61.54 |

Reference max-form lower bounds from prior work: reg3 = 34 is the *achieved* sc
(an upper bound on optimal sc, not a proven LB); sycamore = **53 is a proven
carving-width lower bound** (attempt-022, "no max-form bound can exceed 53",
frontier width-optimal), which is *stronger* than the balanced-separator 47.

---

## 4. Interpretation — the bound does NOT close the gap (and why)

The measurement contradicts the working hypothesis. The sum-form DP bound, with
the true (min-over-all-subsets) isoperimetric profile, **exceeds the
balanced-separator max-form by less than one bit** (30.81 vs 30; 47.17 vs 47)
and sits **below** both the frontier sc and the frontier tc — and, for
sycamore, even below the known carving-width bound of 53.

**Mechanism (measured, `scripts/diagnose.py`).** The DP is free to choose a
*maximally skewed* tree. It routes the single unavoidable balanced node to the
globally-cheapest balanced cut and makes every other node tiny:
* reg3: the DP-optimal shape has **exactly 1** node in the balanced window
  (b=30); the single largest term is **57%** of Σ 2^{b}.
* sycamore: **1** balanced node (b=47); the largest term is **89%** of the sum.

So log2 Σ 2^{b} ≈ (top term) + O(1): the *sum*-form collapses to the *max*-form.
The min-over-subsets profile decouples the cut at each size from the tree, and a
skewed spine can pretend to realize the global optimum at every size at once.

**What a real tree pays.** A tc-optimal tree is essentially *balanced*, which
forces a whole plateau of wide nodes it cannot avoid:
* reg3: **23** nodes within 3 of the peak (34); 3 at exactly 34.
* sycamore: **82** nodes within 3 of the peak; **72 at exactly 53**.

Evaluating the sum-form on the frontier tree's **own nested boundary profile**
(a valid lower bound on *that* tree's tc) therefore gives the large values
**37.34** (reg3) and **59.24** (sycamore) — explaining **93.3%** and **96.2%**
of the tree's tc. The residual is exactly the Σ 2^{|E(A,B)|} contraction slack
from (i): tc_tree − sumform_own = 40.02−37.34 = **2.68** bits (reg3) and
61.61−59.24 = **2.37** bits (sycamore).

**Conclusion.** Theorem 1 is correct and rigorously proven. But as an
*optimizer-independent* lower bound it is weak: the isoperimetric-profile DP is
dominated by its single largest term because the relaxation permits tree skew,
so it barely improves on the max-form and does not certify the gap to the
frontier. The gap *is* explained by the sum-form only when the profile is the
**constrained (nested, balanced) profile a good tree actually realizes** — which
is a per-tree quantity, not a bound on the optimum. Closing the gap rigorously
requires a bound that forbids skew, e.g. a *balance-constrained* profile DP
(minimize over trees whose splits satisfy j ≥ αk) or a carving-/branch-width
argument like the one that yields sycamore's certified 53.

---

## 5. Reproduce

```
scripts/graphlib.py           # graph loading, Laplacian/λ₂, boundary
scripts/dp.py                 # Theorem-1 DP (float + exact big-int)
scripts/empirical_profile.py  # b_emp: fiedler sweep + region grow + FM + harvest
scripts/tree_profile.py       # parse TreeSA tree; per-node |∂S_v|, cost_v, E(A,B)
scripts/run_all.py            # driver -> data/results.json, *_profile.csv
scripts/diagnose.py           # DP-shape & tree-profile diagnostics (§4)
scripts/verify_cuts.py        # independent double-count of every cut
scripts/plot_profiles.py      # data/profiles.png
scripts/test_sanity.py        # DP vs brute force; spectral inequality (0 viol.)

# frontier trees (90 s tuned TreeSA each), already in data/:
research/validator/bin/treesa_tuned <target>.json 90000 data/<name>_tree.json

python3 scripts/test_sanity.py     # all sanity checks
python3 scripts/run_all.py         # bounds + gap table
python3 scripts/verify_cuts.py     # cut verification
python3 scripts/diagnose.py        # mechanism diagnostics
```

Data files: `data/results.json` (all numbers, profiles, tree node multisets),
`data/reg3_250_profile.csv`, `data/sycamore_m20_profile.csv`
(columns: k, b_spec, b_emp_raw, b_emp_windowmin), `data/profiles.png`.

**Certified vs high-confidence — scorecard.**
* CERTIFIED (rigorous): Theorem 1 and its proof; the per-node identity;
  λ₂ values and b_spec(k) ≤ b(k); the spectral sum-form LB(tc) =
  **13.14 / 9.84**; the DP correctness (brute-force + closed-form checked).
* HIGH-CONFIDENCE (not certified): the empirical profile b_emp(k) and the
  empirical sum-form LB(tc) = **30.81 / 47.17** (assumes heuristic cuts are
  near-optimal; window-min applied for conservatism; all cuts double-count
  verified as genuine, so these are honest *estimates of the bound's value*,
  bracketed below by the certified 13.14/9.84).
