# Figure plan — caption-summaries (Phase 1)

Data provenance: every number is from the autoresearch audit trail
(`docs/discussion/cycle-01..06`, `.worktrees/attempt-{021,022,026,027,031}`,
`research/validator/leaderboard.json`). Regenerate with `figures/make_figures.py`.

1. **fig1_certification_ladder** — For both benchmark instances, every lower
   bound we can certify or structurally argue is placed on the same log₂-cost
   axis as the achieved optimizer frontier: on sycamore_m20 the optimum is
   localized to [53, 61.544] and the remaining gap ≈ log₂ n is a counting
   residual no width-type bound can reach. *(Flagship; main result 1.)*
2. **fig2_frontier_convergence** — Ten in-house mechanisms and two independent
   toolchains (omeco TreeSA, cotengra SA) land within ~0.5 bits of one frontier
   at a matched 90 s single-thread budget, while hyper-optimization without SA
   refinement lags by 1.7–2.4 bits and the mis-tuned default lags by 4–10.
   *(Main result 2.)*
3. **fig3_sc_target_masquerade** — Eight novel search mechanisms appeared to
   beat the reference by up to 9.7 bits (≈800×) until one reference parameter
   (`sc_target`) was re-tuned, after which the tuned reference explains the
   entire gain. *(Methodology main result; motivates the omeco library fix.)*
4. **fig4_profiles** — The measured isoperimetric profile b(k), the certified
   spectral bound, and the frontier tree's own (|S|,|∂S|) nodes show why
   profile-only bounds cap at the bisection width (30/47) below the width
   floor (34/53): the b(141) ≤ 40 certificate shows off-center cuts dip below
   temporal slabs. *(Theory: why Theorems 1–2 collapse; feeds the
   impossibility observation.)*
5. **fig5_profile_conservation** — Reducing a tree's peak node cost by one bit
   fattens the near-peak shelf (7→19 and 5→17 nodes) at equal or better tc:
   flops behave as conserved under profile reshaping, so no profile-aware
   search has slack to exploit. *(Supports near-optimality of the frontier.)*
6. **fig6_budget_scaling** — The frontier moves by < 0.1 bits from 90 s to
   900 s on both instances: the plateau is structural, not budget-limited.

Candidate figure 7 (not yet drafted): methodology timeline of the six cycles
with the three instrument-caught failure modes (illusory gain ×2, invisible
harness constraint) — needed only if the methodology story line is chosen.
