# Telegram outline — story A' (question-led; executed 2026-07-25)

Target: Quantum, regular article. User-selected story: "When can tuned
simulated annealing be beaten at tensor-network contraction ordering?"
Supersedes the certification-led outline (v0.1, preserved in git history).
This file reflects the structure as drafted in main.tex.

Figures (as numbered in the source):
fig1 sc_target cliff · fig2 frontier convergence · fig3 bounds ladder ·
fig4 isoperimetric profiles · fig5 profile conservation ·
fig7 methodology timeline · fig8 record board (flagship) ·
fig9 simplification · fig10 waist surgery · fig11 same-machine campaign
(budget scaling / distributions / family trend) · fig12 inference two
regimes · roadmap TikZ (7 stages). Taxonomy is Table 3 (was figure F7).

- **Title**: When can tuned simulated annealing be beaten at
  tensor-network contraction ordering?
- **Abstract**: question; four answers (tuning first; certified never on
  converged instances; input-injection mechanisms at scale; two-regime
  inference frontier); same-machine external validation; 56-attempt loop.
- **§1 Introduction**: ordering matters across domains; comparisons are
  relative; the question; the four-part answer; reader shortcut =
  fig1 + fig8 + Table 2.
- **§2 Setup**: cost model (tc log-sum-exp, sc); Table 1 instances with
  provenance (2 synthetic certification + 8 benchmark imports + 10 UAI
  in §8); protocol (90 s single-thread, independent rescoring, per-run
  relabeling, confirm-twice worse-of-two records, median-of-3 at scale);
  same-machine re-measurement campaign (dedicated 2-core server: 305
  runs + Julia ladder + inference batch).
- **§3 Tuning before beating [MR1]**: fig1 cliff; 20x/800x/2100x; the
  published Julia TreeSA row (71.0) exhibits the same cliff; tuned
  reference = the paper's fairness spine.
- **§4 The certified null (where SA cannot be beaten)**: §4.1 thirteen
  mechanisms + cotengra tie one frontier (fig2); §4.2 width-optimality,
  interval [53, 61.544] [MR2], Theorems 1-2, profile-cap impossibility
  (Obs. 1, figs 3-5).
- **§5 Beating SA I — simplification [MR3]**: fig9; 88.7% Sycamore
  shrink; A/B 61.3 vs 76.0; record march 60.605→59.911; zero effect on
  the regular-graph control (why the proxy hid it).
- **§6 Beating SA II — waist surgery [MR4]**: fig10; mechanism; 835
  calls, waist never locally minimal, gaps 3-12 bits, cuts sparse;
  ratchet-only ablation (49.6→47.38, 50.0→36.8); fixed-work variance
  0.8→0.08; three records incl. the only expander fall.
- **§7 Record board + same-machine validation**: fig8 board; fig11
  campaign (budget scaling: Sycamore gap never closes, surface code 10x
  time-to-quality, reg3_1000 crossover at 900 s — honest negative;
  distributions: surface-code fully separated; family trend: gap grows
  with d); Table 2 external baselines (beat Julia ladder at 10x budget
  by 1.5-4.6 bits; published suite by 6.8/4.9/2.0; dbn goes to
  treewidth — honest; nqueens excluded as unstable).
- **§8 Inference: a frontier with two regimes [MR5]**: fig12; UAI-2014
  export faithful to TensorInference's construction; defaults 12-30 bits
  off frontier; elimination wins dense DBN (a038 within 0.7), annealing
  wins linkage (KaHyPar segfaults ×4, min-fill 6-15 bits off);
  elimination-probe-first portfolio.
- **§9 Composites, anytime, VE seeds**: 047 three records in one run;
  TTF records 16.1 s / 0.2 s; VE seed mechanism + its 4086-label
  boundary.
- **§10 Taxonomy of failure [MR6]**: Table 3; four families with
  measured causes; the central negative lesson (control vs input).
- **§11 Methodology**: 56 attempts / 10 cycles; three instrument-caught
  failure modes; harness evolution (lanes, CPU accounting, medians,
  anytime clocking); fig7 timeline.
- **§12 Roadmap (steered autonomous research)**: 7-stage TikZ; steering
  prompts verbatim; measurement-forced pivots; inference directive as
  stage 7.
- **§13 Conclusions**: the answer as a characterization; deliverables
  (omeco: tuned defaults, simplify pass, surgery, warm-start API, 2 bug
  fixes; upstream PR); open problems (carving bounds, rqc/nqueens,
  heterogeneous dims, slicing).
- **Back matter**: Code/data availability; Use of AI systems
  (JOSS-style disclosure); acknowledgments.
- **Main-result tags**: MR1 §3, MR2 §4.2, MR3 §5, MR4 §6, MR5 §8,
  MR6 §10.

Pending data hooks: cotengra rows on the new instances (batch running on
the measurement server) — add a cotengra column/sentence to §7 when
scored.
