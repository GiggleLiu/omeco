# Telegram outline — story A (certified frontiers)

Target: Quantum, regular article. Figures: F1–F7 (drafts in figures/).

- **Title**: Certified frontiers for tensor-network contraction ordering
- **Abstract**: one sentence per section; three tagged main results.
- **§1 Introduction**: contraction ordering matters (RQC simulation, many-body);
  heuristics compared without certified references; three questions: how much
  do defaults cost, where is the optimizer frontier, how far from optimal.
  Forward pointers to F1/F2/F3.
- **§2 Setup**: einsum graphs, contraction trees, tc/sc (log2, log-sum-exp);
  instances (reg3_250, sycamore proxy — construction stated honestly);
  evaluation protocol: 90 s single-thread wall-clock, independent rescoring
  from topology, per-run relabeling, confirmed records. [no figure]
- **§3 The memory-target effect (MAIN RESULT 1)**: F1 sweep; the rule
  ("aspirational low sc_target is harmful; unbounded is safe"); ablation
  evidence (018); one-directionality; library fix. tc drop 4.3/9.7 log2.
- **§4 The optimizer frontier (MAIN RESULT 2a)**: F2; 13 mechanisms, two
  toolchains, one basin; cotengra hyper vs SA modes; budget-independence
  (part of F6a); what this does and does not claim.
- **§5 Certified bounds (MAIN RESULTS 2b + 3)**: F3 ladder; width-optimality
  of the frontier (carving cut 53, explicit temporal cut); interval
  [53, 61.544]; Theorem 1 (sum-form) + collapse; Theorem 2 (dyadic window) +
  collapse; impossibility: profile bounds ≤ bisection width (F4, the
  boundary-40 certificate); profile conservation at the frontier (F5).
- **§6 Scale regimes**: F6; converged vs non-converged; variance dominates
  single runs at n ≥ 1000; honest statement of what scale claims need.
- **§7 Methodology**: F7; hardened-gate loop; three caught failure modes
  (schedule exploit, mis-tuned baseline, scorer bias); negative-control +
  reference-ratchet pattern; 31 attempts audit trail.
- **§8 Conclusions**: the three results restated technically; implications
  (defaults in shipped libraries; certification as the missing benchmark
  practice; carving-width as the right bound type); future work (carving
  LB machinery, variance-aware scale protocols, real Sycamore circuits,
  heterogeneous dims).
- **Main-result sentences**: §3 F1 tagged first; §5 interval tagged second;
  §5 impossibility tagged third.
