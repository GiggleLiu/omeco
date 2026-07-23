# Attempt 031

- attempt: 031
- date: 2026-07-23
- kind: improve
- parent: 026

## Hypothesis

The dyadic-window (balance-constrained) profile bound is the first bound to
**certifiably exceed the max-form cap** (53 on sycamore_m20, ~30 on reg3_250).
Concretely: walk the larger-child descent path of any binary contraction tree;
it visits ≥1 distinct internal node in each dyadic size window
`W_j = (n/2^{j+1}, n/2^j]`, so `tc ≥ log₂ Σ_j 2^{b_min(W_j)}`. Because this sums
one term per window instead of collapsing to a single balanced node (Theorem 1's
failure), it was expected to clear the cap.

## Expected evidence

LB(tc) > 53 on sycamore (HC, several windows near ~53); reg3 ~30.1–31; an
updated gap table with a Theorem-2 row; certified (spectral) and high-confidence
(empirical) variants; explicit cut certificates.

## What was done

- **Theorem 2 formalized and proved rigorously** (`THEOREM2.md §1–2`): larger-
  child descent with halving `⌈s_i/2⌉ ≤ s_{i+1} ≤ s_i−1`; window-coverage lemma
  (each `W_j`, `j=1..⌊log₂n⌋−1`, hit by a distinct node; edge cases down to
  size-2 windows stated); distinctness across disjoint windows; the closed-
  network convention handled (root `|∂V|=0` ⇒ drop `j=0`). Validated on 3000
  random trees (0 failures).
- **Path-DP strengthening** (`§3`): `P(s)=2^{b(s)}+min_{⌈s/2⌉≤s'<s}P(s')`, proven
  a valid LB with `dyadic ≤ pathDP ≤ tc_opt`, and shown `= min over ALL tree
  shapes` by brute force (`n=4..9`, 0 discrepancies).
- **Both variants, both instances**: certified spectral `b_spec=λ₂k(n−k)/n`
  (026, re-used) and a dedicated per-window min-cut search (`window_search.py`);
  every cut triple-verified (edge count / matvec `xᵀ(D−A)x` / third recount).

## Result — hypothesis REFUTED (informative negative)

Gap table (log2 bits; CERT = certified, HC = high-confidence):

| instance | maxform-CERT | maxform-HC | dyadic-CERT | dyadic-HC | pathDP-CERT | pathDP-HC | cap | frontier sc | frontier tc |
|---|---|---|---|---|---|---|---|---|---|
| reg3_250     | 11.36 | 30 |  9.71 | 27.00 | 12.24 | 30.32 | 30 | 34 | 39.95 |
| sycamore_m20 |  5.76 | 47 |  5.52 | 40.00 |  6.94 | 47.01 | 53 | 53 | 61.54 |

**The cap was NOT exceeded.** Dyadic-window HC = 27.00 (reg3) / **40.00**
(sycamore) — below both caps, and even below 026's sum-form (30.81 / 47.17): the
"fix" is *weaker* than the parent, because taking the window minimum discards the
balanced node. Path-DP HC = 30.32 / **47.01** — it recovers the balanced
separator (the un-skippable band `[n/3,2n/3]`) plus <1 bit of log-sum-exp slack,
reproducing 026 and staying below sycamore's proven 53.

**Two structural reasons** (`THEOREM2.md §6`):
1. Log-sum-exp collapse — `log₂ Σ_j 2^{b_min(W_j)} ≤ max_j b_min(W_j) + log₂(J−1)`,
   so the multi-window sum adds ≤ ~3 bits over its largest term.
2. `max_j b_min(W_j) = b_min(W_1)` and `W_1=(n/4,n/2]` dips to `n/4`, where cheaper
   cuts exist than in the balanced band; e.g. a **triple-verified size-141
   sycamore set with boundary 40** proves `b(141) ≤ 40 < 53`. Temporal slabs
   (53) are not the minimum cuts at those sizes; compact spacetime boxes cut 40.

The profile-only approach is provably capped at the **bisection width** (47
sycamore / 30 reg3); the `47→53` gap lives in the joint structure of nested cuts
(carving/branch-width, attempt-022's 53), which the per-size isoperimetric
profile cannot see. Every bound `≤` frontier tc, as required (asserted, passes).

## Takeaway for re-planning

Every profile-based relaxation (Thm-1 sum-form, Thm-2 dyadic-window, path-DP)
collapses to the max-form / bisection width and cannot reach the carving-width
cap. The gap to the frontier is a *nested-cut* phenomenon. To beat the cap one
must bound the joint cost of the whole nested cut family — i.e. a carving-/
branch-width or spacetime-volume argument (the direction that already yields
sycamore's certified 53), not any function of the per-size profile `b(k)`.

## Outcome (measurement attempt — no scored run by design)

- Theorem 2 proven and validated; dyadic-window and path-DP bounds computed
  (certified + HC) for both instances.
- Max-form cap NOT exceeded (sycamore HC 40.0/47.0 vs 53; reg3 27.0/30.3 vs 30).
- Root cause: log-sum-exp collapse + off-center window minima; profile bounds
  cap at bisection width, blind to the bisection→carving gap.
- Deliverables: `THEOREM2.md`, `scripts/{window_search,run_theorem2,test_theorem2,explore}.py`
  (+ reused 026 scripts), `data/{window_search,theorem2_results}.json`,
  cut certificates in `data/window_search.json`.
