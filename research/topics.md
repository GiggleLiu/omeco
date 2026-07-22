# Autoresearch Topics

## Beat existing contraction-order optimizers

**Slug:** `beat-existing-optimizers`

**Problem statement.** OMECO's current optimizers (GreedyMethod, TreeSA,
TreeSASlicer) mirror the Julia OMEinsumContractionOrders baselines. Find any
contraction-order search algorithm that wins against these baselines on
standard benchmark graphs: either a strictly better contraction complexity
(tc, log2 FLOPs) at the same optimizer wall-clock budget, or the same tc
reached in less optimizer time. The attempt space is open — richer simulated
annealing move sets, min-cut/partition-based tree builders, improved greedy
cost functions, hybrid seeds, or novel designs — as long as the result is a
valid contraction tree scored under the fixed protocol.

**Why autoresearch fits** (suitability scores 1–5):
- **Checkable: 5** — tc/sc/rwc are exact log2 quantities recomputed by the
  validator from the emitted contraction tree; no human judgment.
- **Cheap: 5** — one attempt evaluates in ~2–4 minutes on the shipped
  benchmark suite on local hardware.
- **Headroom: 5** — move sets × schedules × cut strategies × cost functions
  form a design space supporting hundreds of genuinely distinct attempts.
- **Publishable: 4** — beating TreeSA/cotengra-class optimizers on standard
  graphs (chains, grids, 3-regular) is a recognized open bar; recent venues
  (SEA 2024, arXiv 2025) still publish incremental wins here.

**Key references:**
- Optimizing Tensor Network Partitioning using Simulated Annealing —
  arXiv:2507.20667
- Hyper-optimized tensor network contraction (cotengra) — arXiv:2002.01935;
  SA refinement mode per arXiv:2108.05665
- Computing Tree Decompositions with FlowCutter: PACE 2017 Submission —
  arXiv:1709.08949
- Improved Cut Strategy for Tensor Network Contraction Orders —
  DOI:10.4230/LIPIcs.SEA.2024.27
- Optimizing Tensor Contraction Paths: A Greedy Algorithm Approach With
  Improved Cost Functions — arXiv:2405.09644
- Positive-instance driven dynamic programming for treewidth —
  arXiv:1704.05286 (exact baselines for small instances)

### Metrics

- **Δtc@budget** (primary): For each graph `g` in `benchmarks/graphs/*.json`,
  run the candidate under wall-clock budgets `T(g)`, `T(g)/4`, `T(g)/16`,
  where `T(g)` is the measured runtime of baseline TreeSA (default params,
  `RAYON_NUM_THREADS=1`, measured once and cached by the validator). Score =
  mean over graphs and budgets of `tc_baseline(g) − tc_candidate(g)` (log2).
  Positive = better; finding equal tc faster also scores positive via the
  smaller budgets. Computation: release-build run over the suite, ~2–4 min
  per attempt. Gaming risks: fake reported tc (caught by valid-tree guard);
  budget overrun (validator kills the process at the limit).
- **valid-tree** (guard): Attempt emits its contraction tree as JSON
  (`writejson` format); the validator recomputes tc/sc/rwc from that tree
  using baseline omeco complexity code and checks it is a valid binary
  contraction over exactly the input tensors. Reported numbers are never
  trusted; invalid or mismatched → attempt fails. Cost: milliseconds.
- **held-out generalization** (guard): 5 fresh graphs (random 3-regular +
  grids, seeds generated at validation time, never visible to attempts) are
  scored identically; require `Δtc_heldout ≥ Δtc_public − 1.0`. Catches
  hard-coded orders / overfitting to shipped benchmarks. Cost: ~1 min.
- **sc-cap** (guard): `sc_candidate(g) ≤ sc_baseline(g) + 2` for every graph.
  Catches trading memory blow-up for flops. Cost: free (from valid-tree
  recomputation).
- **resource-cap** (guard): `RAYON_NUM_THREADS=1`, no network access,
  wall clock per scored run ≤ STATE.md `time_limit_seconds`. Catches wins by
  parallelism or external solvers rather than algorithmic improvement.
