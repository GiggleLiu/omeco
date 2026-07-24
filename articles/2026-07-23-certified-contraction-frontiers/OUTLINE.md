# Telegram outline — story A' (question-led, 2026-07-24 restructure)

Target: Quantum, regular article. User-selected story: "When can tuned
simulated annealing be beaten at tensor-network contraction ordering?"
Supersedes the certification-led outline (v0.1, preserved in git history).

Figures: F1 sc_target cliff (kept), F2 record board (NEW, flagship),
F3 simplification attribution (NEW), F4 waist surgery (NEW: always-
improvable scatter + tc trajectories), F5 certification ladder (kept,
was fig3), F6 frontier convergence + window/local-optimality (kept, was
fig2, extended), F7 mechanism map (NEW, table-figure of 20 falsified +
5 winning mechanisms with causal one-liners), F8 methodology timeline
(kept, was fig7, extended to 56 attempts).

- **Title**: When can tuned simulated annealing be beaten at
  tensor-network contraction ordering? (working)
- **Abstract**: question; tuning result (35x/2100x); certification on
  converged instances; two winning mechanisms with record board; the
  waist inversion; negative-mechanism taxonomy; library deliverables.
- **§1 Introduction**: contraction ordering matters; heuristics compared
  without knowing when the incumbent is beatable; the question; three
  answers (never on structureless/converged instances — certified;
  by input transformation and by global cut moves on structured
  instances; the boundary is characterizable). Forward pointers F1/F2.
- **§2 Setup**: einsum, trees, tc/sc; instance set incl. real benchmark
  imports (provenance); protocol: 90 s single-thread, median-of-3 at
  scale, harness-clocked anytime, records = confirm-twice worse-of-two
  vs quiet-machine tuned-TreeSA references. [table of instances]
- **§3 Tuning before beating (MAIN RESULT 1)**: F1 cliff; the sc_target
  rule; 35x/2100x; all subsequent comparisons are against the TUNED
  reference — the paper's fairness spine.
- **§4 Where SA cannot be beaten**: F5 ladder + F6 convergence;
  certified interval [53, 61.5] on the proxy; window-exact optimality;
  profile-bound impossibility; 13-mechanism convergence. Establishes
  the null hypothesis the next sections break.
- **§5 Beating SA I — structural simplification (MAIN RESULT 2)**: F3;
  shrink table; matched-budget A/B (61.3 vs 76.0); the record march
  60.61 -> 59.91 (three independent replications); why the proxy hid it
  (013 vs 039); generality (surface code, nqueens).
- **§6 Beating SA II — waist surgery (MAIN RESULT 3)**: F4; the
  mechanism; the inversion (waist never globally minimal; hundreds of
  calls, zero exceptions; sparse not near-clique); records on
  separable instances (ksg 36.96, surfacecode 47.40, reg3_1000 131.07)
  vs the expander/dense boundary; fixed-work variance lever.
- **§7 The composite and the anytime axis**: routing + racing (047's
  three records in one run; probe-leader-never-wins); TTF records
  (16.1 s reg3_250, 0.2 s dbn); VE seeds for hyperedge instances.
- **§8 What does not work — a mechanism taxonomy (MAIN RESULT 4)**: F7;
  the four failure families (construction proxies, search-control
  globality, exhaustive enumeration, population policy) each with the
  measured root cause; the outer-product DP counterexample; why
  negatives with causal chains are the field's missing map.
- **§9 Methodology**: F8; harness (independent rescoring, relabeling,
  lanes, medians, early-abort, harness-clocked TTF); the instrument
  catches (illusory gains, scorer bias, contention distortion,
  confirmation coin-flips); 56-attempt audit trail.
- **§10 Conclusions**: the question answered as a characterization;
  library deliverables (simplifier, surgery pass, warm-start API, two
  bug fixes; upstream to OMEinsumContractionOrders.jl); open: rqc/
  nqueens bounds, heterogeneous dims, real-workload slicing.
- **Main-result tags**: §3 (tuning cliff), §5 (simplification records),
  §6 (waist inversion + records), §8 (taxonomy).
