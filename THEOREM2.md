# Theorem 2 — the balance-constrained (dyadic-window) profile lower bound

Attempt-031 · 2026-07-23 · THEORY + MEASUREMENT · parent: attempt-026

This is the fix proposed by attempt-026 for the collapse of Theorem 1: forbid
tree skew by walking the **larger-child descent path** and charging one node per
**dyadic size window**. We formalize it, prove it rigorously, compute it (both a
certified spectral variant and a high-confidence empirical variant) on
`reg3_250` and `sycamore_m20`, and add a Theorem-2 row to the gap table.

**Headline result (measured): the bound does *not* exceed the max-form cap.**
The dyadic-window LB collapses (as a log-sum-exp) to its single largest term,
which is the window-min of the widest window — and that minimum sits at the
window's *off-center* end, **below** the balanced separator. The stronger
path-DP variant recovers exactly the balanced separator (+<1 bit), i.e. it
reproduces attempt-026's result and confirms its diagnosis. Full numbers in §6.

---

## 0. Setup and conventions (same as attempt-026)

Both instances are **closed** dim-2 tensor networks (empty output; every index
on exactly two tensors), hence ordinary graphs `G=(V,E)`, `|V|=n`. A contraction
tree `T` is a binary tree whose leaves are the tensors `V`. An internal node `v`
has **leaf-set** `S_v ⊆ V` (the tensors under `v`), split by its two children
into `A_v ⊍ B_v = S_v`. `∂S = {edges with exactly one endpoint in S}`. Because
all bonds have dimension 2 and the network is closed, the intermediate tensor at
`v` has exactly the indices `∂S_v`, and the log2-cost of the contraction at `v`
is

> `cost_v = |labels(A_v) ∪ labels(B_v)| = |∂S_v| + |E(A_v,B_v)| ≥ |∂S_v|`   (Thm-1 (i))

so `tc(T) = log2 Σ_v 2^{cost_v} ≥ log2 Σ_v 2^{|∂S_v|}`. The **isoperimetric
profile** is `b(k) = min_{|S|=k} |∂S|`, and `|∂S_v| ≥ b(|S_v|)`. All of this is
attempt-026, restated. Theorem 2 controls the *sum over a specific subset of
nodes* (the descent path), replacing Theorem-1's unconstrained DP over all
tree shapes — which was free to hide all cost in one balanced node.

---

## 1. Theorem 2 (statement)

Let `T` be any binary contraction tree over `n` leaves. Define the **larger-child
descent path** `v_0, v_1, v_2, …`: `v_0 = root`; from `v_i` (internal) descend to
the child with the larger leaf-set (ties arbitrary); stop at a leaf. Let
`s_i = |S_{v_i}|`, so `s_0 = n`.

For `j = 0, 1, …, ⌊log₂ n⌋−1` define the **dyadic window**
`W_j = ( n/2^{j+1}, n/2^j ]` (integer sizes `k` with `⌊n/2^{j+1}⌋ < k ≤ ⌊n/2^j⌋`),
and `b_min(W_j) = min_{k ∈ W_j} b(k)`.

**Theorem 2.** With `J = ⌊log₂ n⌋`,

> `tc(T) ≥ log₂ Σ_{j=1}^{J-1} 2^{b_min(W_j)}`  for every `T`, hence
> `tc_opt ≥ log₂ Σ_{j=1}^{J-1} 2^{b_min(W_j)} =: LB_dyadic`.

Replacing `b` by any lower bound `b_lo ≤ b` (e.g. the certified spectral
`b_spec`) keeps `LB_dyadic` a valid lower bound.

The sum starts at `j = 1`: window `W_0 = (n/2, n]` is guaranteed only by the
**root**, whose boundary is `0` under the closed-network convention (`∂V = ∅`),
so it contributes nothing usable — see the proof's Step 5.

---

## 2. Proof

### Step 1 — halving of the descent path.
An internal node `v_i` of leaf-set size `s_i` splits into `A ⊍ B` with
`|A|+|B| = s_i`, both `≥ 1`. The larger child has size
`max(|A|,|B|) ≥ ⌈s_i/2⌉`, and the smaller has size `≥ 1`, so
`⌈s_i/2⌉ ≤ s_{i+1} ≤ s_i − 1`. In particular `s_{i+1} ≥ s_i/2` and the sequence
is **strictly decreasing**, from `s_0 = n` down to `1` (a leaf). Hence all path
nodes have distinct sizes, so they are **distinct nodes** of `T`. ∎(Step 1)

### Step 2 — window-coverage lemma.
Fix `j` with `1 ≤ j ≤ J−1`, so `n ≥ 2^{j+1}` and the threshold
`τ_j := n/2^{j+1} ≥ 1`. The path sizes start at `s_0 = n > τ_j` and end at `1 ≤ τ_j`,
strictly decreasing, so there is a unique consecutive pair with
`s_i > τ_j ≥ s_{i+1}`. By halving (Step 1), `s_i ≤ 2 s_{i+1} ≤ 2 τ_j = n/2^j`.
Therefore

> `n/2^{j+1} < s_i ≤ n/2^j`,  i.e. `s_i ∈ W_j`.

So the path visits at least one node whose size lies in `W_j`. ∎(Step 2)

*Edge cases.* The largest usable `j` is `J−1 = ⌊log₂ n⌋−1`; then
`τ_{J-1} = n/2^{⌊log₂ n⌋} ∈ [1,2)`, so `W_{J-1}` contains only size-2 nodes
(reg3: `W_6 = (1.95, 3.91] = {2,3}`; sycamore: `W_8 = (1.10, 2.19] = {2}`). For
`j = J` we would need `τ_J = n/2^{J+1} < 1`, and the crossing `s_i > τ_J ≥ s_{i+1}`
may fail to place `s_i` inside `(n/2^{J+1}, n/2^J]` (the terminal leaf sits below),
so `j = J` is **not** guaranteed. The guaranteed windows are exactly
`j = 1, …, J−1`. Every `s_i ∈ [2, n]` is a genuine internal node.

### Step 3 — distinctness across windows.
The windows `W_1, …, W_{J−1}` are pairwise disjoint intervals. The node selected
for window `W_j` (Step 2) has size in `W_j`, so nodes selected for different `j`
have sizes in disjoint ranges ⇒ they are **distinct internal nodes** of `T`.
∎(Step 3)

### Step 4 — charging.
Each selected node `v_{i(j)}` (size in `W_j`) is a genuine internal node, so by
Thm-1 (i), `2^{cost_{v}} ≥ 2^{|∂S_v|} ≥ 2^{b(|S_v|)} ≥ 2^{b_min(W_j)}`. Since
`Σ_v 2^{cost_v}` sums over **all** internal nodes and the selected nodes are
distinct (Steps 1,3),

> `Σ_v 2^{cost_v} ≥ Σ_{j=1}^{J-1} 2^{cost_{v_{i(j)}}} ≥ Σ_{j=1}^{J-1} 2^{b_min(W_j)}`.

Taking `log₂`: `tc(T) ≥ log₂ Σ_{j=1}^{J-1} 2^{b_min(W_j)}`. ∎(Step 4)

### Step 5 — the closed-network / root convention (why `j=0` is dropped).
`W_0 = (n/2, n]` contains `s_0 = n` (the root), and the crossing argument of
Step 2 for `j=0` (`τ_0 = n/2`) selects the *last* path node with size `> n/2`.
If the root split is balanced (`s_1 = ⌈n/2⌉ ≤ n/2` when `n` is even), that last
node is the **root itself**, whose boundary is `|∂V| = 0` (closed network). Then
the guaranteed `W_0` node contributes only `2^0 = 1`, not `2^{b_min(W_0)}`. So
`j=0` yields no usable term and is excluded. (Including it would add the honest
but useless `+1`; the reported `LB_dyadic` omits it.) ∎(Step 5)

### Relaxation.
For any `b_lo ≤ b` termwise, `b_min,lo(W_j) ≤ b_min(W_j)`, so
`log₂ Σ_j 2^{b_min,lo(W_j)} ≤ LB_dyadic ≤ tc_opt`: the spectral variant is a
valid (weaker) certified lower bound. ∎

**Validation.** `scripts/test_theorem2.py` checks Steps 1–3 on 3000 random
binary trees (`n = 4..400`): strict decrease, `s_{i+1} ≥ ⌈s_i/2⌉`, every window
`j=1..J−1` hit by a distinct node — **0 failures**.

---

## 3. The strengthening over descent sequences (path-DP)

Theorem 2 charges *one* node per window and lower-bounds it by the window
minimum. That double relaxation is loose. The tight object minimizes over the
descent sequence directly. Any tree's larger-child path is a sequence
`n = s_0 > s_1 > … ` with `s_{i+1} ∈ [⌈s_i/2⌉, s_i − 1]`; conversely we only need
a *lower* bound, so we minimize over **all** such sequences:

> `P(1) = 0;  P(s) = 2^{b(s)} + min_{⌈s/2⌉ ≤ s' ≤ s−1} P(s')`   (s ≥ 2)
> `LB_pathDP = log₂ P(n)`.

**Proposition.** `LB_dyadic ≤ LB_pathDP ≤ tc_opt`.
*Proof.* For any tree `T`, its larger-child path sizes form a feasible sequence,
so `Σ_{path} 2^{b(s_i)} ≥ P(n)` (minimum over feasible sequences), and
`tc(T) ≥ log₂ Σ_{path} 2^{|∂S|} ≥ log₂ Σ_{path} 2^{b(s_i)} ≥ log₂ P(n)`. The
first inequality (`LB_dyadic ≤ LB_pathDP`) holds because the dyadic sum uses one
window-min per window while the path visits ≥1 node in each window and pays
`b(s_i) ≥ b_min(W_j)` at *each* — a subset of the path DP's terms. ∎

`test_theorem2.py::test_pathdp_vs_bruteforce` confirms `P(n)` equals the exact
minimum over *all* binary-tree shapes of the larger-child path sum for
`n = 4..9` (0 discrepancies), and `::test_pathdp_is_lower_bound` confirms
`dyadic ≤ pathDP ≤ (every random tree's path sum)` on 2000 random profiles.

**Note on monotonicity (as requested).** When `b` is non-decreasing on `[1, n/2]`
and symmetric `b(k)=b(n−k)` (the expander regime), the sequence minimizer takes
one size per dyadic window at its cheap end, so `LB_pathDP` and `LB_dyadic`
coincide up to the terms forced by the un-skippable **balanced band**
`[n/3, 2n/3]` (which no factor-≤2 step can jump over): every feasible sequence
has a size in `[n/3,2n/3]`, so `LB_pathDP ≥ b_min([n/3,2n/3])` = the
balanced-separator max-form bound. This is why path-DP never drops below the
max-form, whereas the raw dyadic bound can (its widest window `W_1=(n/4,n/2]`
dips below the band). Measured in §6.

---

## 4. The two profile variants

### 4a. Certified spectral `b_spec` (RIGOROUS, from attempt-026)
`b_spec(k) = λ₂·k(n−k)/n ≤ b(k)`, with `λ₂` the exact algebraic connectivity of
the integer Laplacian (`reg3` 0.203598, `sycamore` 0.046181). Certified in 026
(Rayleigh quotient on `1^⊥`; re-verified on 2000 random subsets, 0 violations).

### 4b. High-confidence empirical `b_emp` (NOT certified)
Per-window minimum-boundary search (`scripts/window_search.py`): fiedler-sweep
prefixes + multi-seed min-boundary region growing to sampled window sizes +
size-preserving FM swap refinement + harvested frontier-tree nested cuts, taking
the elementwise best and symmetrizing `b(k)=b(n−k)`. Because heuristic cuts are
**achievable**, each is an *upper* bound on `b(k)`; the window-min of these is
therefore an **upper** estimate of `b_min(W_j)`. Using it as `b_min(W_j)` gives a
high-confidence *estimate of the bound's value* that assumes the cuts are near
optimal — **not** a certified lower bound on `tc`. Every cut is verified genuine
by **three** independent boundary counts (edge iteration, adjacency matvec
`xᵀ(D−A)x`, and a third direct recount); all agree (§7).

---

## 5. Per-window tables

`b_emp_min` is the best (smallest) boundary found at any searched size in the
window (HC); `b_spec_min` is the certified spectral minimum over the window.

### reg3_250 (n=250, J=⌊log₂250⌋=7, windows j=1..6)

| j | window `W_j` | k-range | b_emp_min (HC) | @k | cert size / boundary | b_spec_min (CERT) | @k |
|---|---|---|---|---|---|---|---|
| 1 | (125, 250] | 63..125 | **27** | 65 | 65 / 27 ✓ | 9.594 | 63 |
| 2 | (62.5, 125] | 32..62 | 17 | 33 | 33 / 17 ✓ | 5.681 | 32 |
| 3 | (31.25, 62.5] | 16..31 | 11 | 17 | 17 / 11 ✓ | 3.049 | 16 |
| 4 | (15.6, 31.25] | 8..15 | 8 | 8 | 8 / 8 ✓ | 1.577 | 8 |
| 5 | (7.8, 15.6] | 4..7 | 4 | 4 | 4 / 4 ✓ | 0.801 | 4 |
| 6 | (3.9, 7.8] | 2..3 | 2 | 2 | 2 / 2 ✓ | 0.404 | 2 |

`LB_dyadic`: CERT **9.712**, HC **27.001**. (Windows W1..W6 map to k∈[63,125],
etc.; the k-range column is `⌊n/2^{j+1}⌋+1 .. ⌊n/2^j⌋`.)

### sycamore_m20 (n=561, J=⌊log₂561⌋=9, windows j=1..8)

| j | window `W_j` | k-range | b_emp_min (HC) | @k | cert size / boundary | b_spec_min (CERT) | @k |
|---|---|---|---|---|---|---|---|
| 1 | (280.5, 561] | 141..280 | **40** | 141 | 141 / 40 ✓✓✓ | 4.875 | 141 |
| 2 | (140.25, 280.5] | 71..140 | 27 | 72 | 72 / 27 ✓ | 2.864 | 71 |
| 3 | (70.1, 140.25] | 36..70 | 16 | 36 | 36 / 16 ✓ | 1.556 | 36 |
| 4 | (35.06, 70.1] | 18..35 | 9 | 19 | 19 / 9 ✓ | 0.805 | 18 |
| 5 | (17.5, 35.06] | 9..17 | 6 | 9 | 9 / 6 ✓ | 0.409 | 9 |
| 6 | (8.77, 17.5] | 5..8 | 3 | 5 | 5 / 3 ✓ | 0.229 | 5 |
| 7 | (4.38, 8.77] | 3..4 | 2 | 3 | 3 / 2 ✓ | 0.138 | 3 |
| 8 | (2.19, 4.38] | 2..2 | 2 | 2 | 2 / 2 ✓ | 0.092 | 2 |

`LB_dyadic`: CERT **5.523**, HC **40.000**.

The decisive entry is **W_1**: the profile peaks at ~53 near the center
(`k≈280`), but the window minimum lands at its *left* end `k=141`, where a
genuine boundary-**40** cut exists (triple-verified, §7). The window-min is 40,
not 53 — so the dyadic bound cannot reach 53. "Temporal slabs have boundary 53
over a wide size range," but slabs are **not the minimum** cuts at those sizes;
compact spacetime sets (here, the complement of a size-420 region) cut only 40.

---

## 6. Results and the gap table

All values log2-scale bits. CERT = certified rigorous; HC = high-confidence
(assumes near-optimal cuts). "cap" = the max-form ceiling established by prior
work (sycamore **53** = proven carving-width LB, attempt-022; reg3 **30** =
best balanced-cut value, attempt-026).

| instance | maxform-CERT | maxform-HC | **dyadic-CERT** | **dyadic-HC** | **pathDP-CERT** | **pathDP-HC** | cap | frontier sc | frontier tc |
|---|---|---|---|---|---|---|---|---|---|
| reg3_250     | 11.36 | 30 | **9.71** | **27.00** | **12.24** | **30.32** | 30 | 34 | 39.95 |
| sycamore_m20 |  5.76 | 47 | **5.52** | **40.00** |  **6.94** | **47.01** | 53 | 53 | 61.54 |

**Did the max-form cap get exceeded? NO.**
- **Dyadic-window HC**: 27.00 (reg3) and 40.00 (sycamore) — **below** both caps,
  and even below attempt-026's sum-form (30.81 / 47.17). The dyadic "fix" is
  *weaker* than 026's own DP, because the window-min discards the balanced node.
- **Path-DP HC**: 30.32 (reg3) and 47.01 (sycamore). On reg3 it lands 0.32 bit
  above the value 30 — but 30 is itself only the balanced-cut value, and 0.32 is
  exactly the log-sum-exp slack over the balanced separator (30.00), not a
  structural gain. On **sycamore it is 47.01, well below the proven cap 53**.

**Sanity vs frontier (required check).** Every bound is `≤` the frontier tc, as
it must be: reg3 all `≤ 39.95`, sycamore all `≤ 61.54` (`scripts/run_theorem2.py`
asserts `bound ≤ frontier_tc`, passes). The frontier tree's own larger-child
path already sums to 37.34 (reg3) / 59.24 (sycamore) — a *per-tree* lower bound
far above every optimizer-independent bound here, exactly as in 026.

### Why it fails — structural

1. **Log-sum-exp collapse (same mechanism as 026).**
   `log₂ Σ_j 2^{b_min(W_j)} = max_j b_min(W_j) + log₂(Σ_j 2^{b_min(W_j) − max}) ≤
   max_j b_min(W_j) + log₂(J−1)`. With `J−1 ≤ ⌊log₂ n⌋`, the *entire* multi-window
   sum can add at most `log₂ log₂ n ≈ 2.6` (reg3) / `3.0` (sycamore) bits over its
   largest term. A sum of windows cannot beat its max by more than ~3 bits.

2. **The largest window-min is off-center, below the balanced separator.**
   `max_j b_min(W_j) = b_min(W_1)`, and `W_1 = (n/4, n/2]` reaches down to `n/4`,
   where cheaper cuts exist than in the balanced band `[n/3, 2n/3]`. So
   `b_min(W_1) ≤ b_min([n/3,2n/3]) =` balanced separator `≤` cap. Concretely
   `b_min(W_1) = 40 < 47 = ` sycamore bisection `< 53 = ` carving cap;
   `27 < 30` for reg3. Adding ≤3 bits of log-sum-exp cannot cross the gap.

3. **Path-DP repairs (2) but not (1).** Because no factor-≤2 step can jump over
   `[n/3, 2n/3]`, path-DP is forced to pay `b_min([n/3,2n/3])` = the balanced
   separator, recovering the max-form (47 sycamore / 30 reg3), then adds only the
   log-sum-exp slack (<1 bit). It equals attempt-026's sum-form DP by
   construction and inherits its ceiling: **`≈` balanced separator, not the cap.**

4. **Why the cap itself is unreachable this way.** Sycamore's `53` is a
   *carving-width* bound (attempt-022): every contraction tree has a node of
   width `≥ 53` because the spacetime bisection structure forbids all levels
   being as cheap as the single best balanced cut (`47`). Any bound built from
   the *isoperimetric profile alone* sees only `min_S |∂S|` at each size and thus
   caps at the bisection width `47`; it is blind to the `47 → 53` gap, which lives
   in the *joint* structure of nested cuts, not in the per-size minima.

---

## 7. Cut certificates

Every `b_emp_min(W_j)` in §5 is realized by an explicit, verified vertex set
(full sets in `data/window_search.json`, key `windows[j].cert_vertices`). Each
was boundary-counted three independent ways — edge iteration, adjacency matvec
`xᵀ(D−A)x`, and a direct third recount — all agreeing. The **linchpin** is
sycamore `W_1`:

> a size-**141** vertex set with boundary exactly **40** (triple-count agrees:
> 40 = 40 = 40). This proves `b(141) ≤ 40 < 53`, so no profile-based bound using
> window `W_1 = [141,280]` can charge more than 40 there.

(This set is the complement of a compact size-420 spacetime region; the cut
crosses 40 worldlines, fewer than a full temporal slab's 53.)

`reg3` `W_1`: a size-65 set, boundary 27 (`< 30` balanced). Same conclusion.

---

## 8. Reproduce

```
scripts/graphlib.py          # 026: graph load, exact λ₂, boundary  (reused)
scripts/dp.py                # 026: Theorem-1 DP                     (reused)
scripts/empirical_profile.py # 026: region-grow / fiedler / FM       (reused)
scripts/tree_profile.py      # 026: parse frontier tree              (reused)
scripts/window_search.py     # NEW: per-window min-cut search + 3× cut verify
scripts/run_theorem2.py      # NEW: dyadic-window LB, path-DP, gap table
scripts/test_theorem2.py     # NEW: window-coverage/halving/path-DP validation
scripts/explore.py           # NEW: quick dyadic bound from 026 profiles

python3 scripts/window_search.py   # -> data/window_search.json (certificates)
python3 scripts/run_theorem2.py    # -> data/theorem2_results.json + gap table
python3 scripts/test_theorem2.py   # all sanity checks pass
```

Data: `data/window_search.json` (per-window minima + explicit cut vertex sets),
`data/theorem2_results.json` (all bound values), `data/*_profile.csv`,
`data/{reg3,sycamore}_tree.json` (frontier trees, from 026).

---

## 9. Scorecard (certified vs high-confidence)

- **CERTIFIED (rigorous):** Theorem 2 and its proof (Steps 1–5); the
  window-coverage/halving lemmas (validated on 3000 random trees); path-DP as a
  valid lower bound and `= min over tree shapes` (brute-forced `n=4..9`); the
  spectral `b_spec ≤ b`; the certified bounds **dyadic 9.71 / 5.52** and
  **pathDP 12.24 / 6.94** (reg3 / sycamore).
- **HIGH-CONFIDENCE (not certified):** the empirical window minima and hence
  **dyadic 27.00 / 40.00** and **pathDP 30.32 / 47.01** — these *assume* the
  triple-verified cuts are near-optimal; each cut is a genuine achievable set, so
  the true bound is `≤` these values (searching harder can only *lower* them),
  which only strengthens the negative conclusion.

**Verdict.** Theorem 2 is a correct, cheap, optimizer-independent lower bound,
but — like Theorem 1 — it collapses to the max-form (log-sum-exp ≈ max) and does
**not** exceed the max-form cap: 40.0 (HC) on sycamore vs the proven 53, and
30.3 (HC) on reg3 vs 30. Closing the `bisection(47) → carving(53)` gap requires a
bound over the *joint* structure of nested cuts (carving/branch-width), not the
per-size isoperimetric profile, which is provably blind to it.
