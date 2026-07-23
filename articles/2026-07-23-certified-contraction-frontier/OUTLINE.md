# Telegram outline — certification-led, target: Quantum

Contract (user-selected 2026-07-23): certification story; figures 1,2,4,5 lead,
3,6 support; methodology as one section, not the spine.

- §1 Introduction
  - beat 1: contraction order = cost of TN simulation of quantum circuits; optimizers abound
  - beat 2: prior work compares optimizers to each other; nobody certifies distance-to-optimal per instance
  - beat 3: this paper: reproducible fixed-budget frontier + certified/structural LBs bracketing it + impossibility result for profile-only bounds + a default-parameter trap that mimics algorithmic gains
  - beat 4: pointers — Fig 1 (bracket, MAIN RESULT 1), Fig 2 (convergence, MAIN RESULT 2), Thm 2 + Fig 4/5 (impossibility + conservation, MAIN RESULT 3)
- §2 Setup: closed dim-2 TN = graph; tc/sc in bits; per-node identity Lemma 1 (cost = |∂S| + |E(A,B)|); instances reg3_250 / sycamore_m20; 90 s single-thread protocol, per-run relabeling, validator recomputes all costs
- §3 The frontier (Results I): Fig 2 — 10 mechanisms + 2 toolchains converge to 39.95/61.544; Fig 6 — budget-flat 90→900 s; Fig 3 — the sc_target masquerade (honest-baseline lesson + library fix); sentence naming MAIN RESULT 2
- §4 Bracketing the optimum (Theory + Results II): Thm 1 sum-form profile bound + spectral instantiation (certified 13.14/9.84); treewidth minor certificates (18/22); balanced-cut lemma + explicit 53-wire temporal cut → optimum ∈ [53, 61.544], width-optimality of the frontier; Fig 1 (MAIN RESULT 1), Fig 4
- §5 Why the residual resists certification (Results III): Thm 2 dyadic-window bound (proven, collapses); impossibility observation — no function of b(k) exceeds bisection width; Fig 5 profile conservation; residual 8.5 ≈ log₂ n = near-max multiplicity; MAIN RESULT 3
- §6 Practical consequences: sc_target default ∞/cap guidance; omeco fix; when hyper-opt vs SA refinement matters
- §7 Methods: autoresearch protocol, validator design, audit trail (31 worktrees, 6 cycle reports), certificates + reproduce commands
- §8 Conclusions: certification as the missing axis of optimizer benchmarking; open problem: sum-form bounds beyond the profile (nested/carving-aware)

Main-result sentences: §3 (frontier), §4 (bracket), §5 (impossibility). One each, tagged explicitly.
