# NOTES — tensor-network contraction-order optimization

Built by /survey on 2026-07-25 (strategies: landscape mapping, cross-method,
historical lineage, cross-vocabulary). Serves the "Global cut surgery"
manuscript (articles/2026-07-23-certified-contraction-frontiers).

## Field landscape

**Three independent roots, one problem.** The ordering problem was proved
hard three separate times in three fields: optimal join nesting in
relational databases [@ibaraki_1984_optimal], generalized matrix-chain /
sum-of-products loop optimization in parallel compilers
[@lam_1997_optimizing] (with the quantum-chemistry compiler tradition of
[@hirata_2003_tensor]), and — for our exact objective, log-sum time
complexity distinguished from operation count — [@xu_2023_np]. The
complexity backbone for tensor networks is the treewidth correspondence
[@liu_2022_verifying and Markov–Shi], formalized for contraction trees by
[@ogorman_2019_parameterization] and connected to carving decompositions
by [@dudek_2019_efficient].

**The practical-optimizer canon.** Exact search: netcon
[@pfeifer_2013_faster]; imported treewidth solvers
[@dumitrescu_2018_benchmarking; @tamaki_2017_positive;
@tamaki_2022_heuristic; @strasser_2017_computing; @gogate_2004_complete].
Ecosystem front-ends: opt_einsum [@smith_2018_opt], cotengra's
hyper-optimized partitioning portfolio [@gray_2020_hyper], cuTensorNet's
simplify→partition→slice pathfinder [@bayraktar_2023_cuquantum],
OMEinsumContractionOrders.jl [@liu_2026_omeinsumcontractionorders],
EinExprs [@ramrez_2024_einexprs]. Annealing refiners: the TreeSA lineage
originates in [@kalachev_2021_multi]; greedy cost-function refinement in
[@orgler_2024_optimizing]; cut-based divide and conquer in
[@staudt_2024_improved]. The Sycamore ordering race (2017–2022) drove
most of this: [@boixo_2017_simulation; @chen_2018_classical;
@villalonga_2018_flexible; @arute_2019_quantum;
@pednault_2019_leveraging; @huang_2020_classical; @pan_2021_simulating;
@liu_2021_closing; @pan_2021_solving].

**2024–2026 trends directly adjacent to this project.**
(a) Refinement layered on top of constructions: [@guerrero_2026_bond]
shows cotengra-hyper output leaves a bond-dimension-growing gap that NNI
(local tree-move) refinement closes — independent validation that the
construction+refinement frontier is where gains live; our surgery differs
by repairing the dominant cut *globally* rather than by local moves.
(b) Learned/automated heuristic discovery: RL+GNN ordering
[@meirom_2022_optimizing], genetic/SA benchmarking
[@schindler_2020_algorithms], and LLM-evolved ordering heuristics
[@hoppe_2026_algorithmic] — the latter is direct prior art for our
steered-autoresearch methodology. (c) SA moving to partitioning
decisions [@anon_2025_optimizing (Geiger et al. 2507.20667)].
(d) Portfolio behavior exists implicitly (TensorOrder2's solver race
[@dudek_2020_parallel], opt_einsum auto) but no paper treats algorithm
selection for ordering as its subject.

**Cross-vocabulary canon (for the "same problem elsewhere" paragraph).**
Join ordering: [@selinger_1979_access; @steinbrunn_1997_heuristic;
@leis_2015_how] and learned optimizers [@marcus_2019_neo;
@marcus_2020_bao; @yan_2023_join]; bushy-tree DP over connected subgraphs
(Moerkotte–Neumann DPccp — no DOI, VLDB 2006) ≅ exact contraction-tree
DP; [@stoian_2022_optimal] already imports IKKBZ join ordering into
tensor networks. Sparse elimination: [@george_1973_nested;
@lipton_1977_generalized; @george_1989_evolution;
@amestoy_1996_approximate; @karypis_1998_fast; @yannakakis_1981_computing].
Graphical models: [@dechter_1999_bucket; @bodlaender_2010_treewidth].
Partitioning refinement: [@fiduccia_1982_linear] (the FM pass surgery
uses), [@karypis_1997_multilevel], survey [@ccatalyurek_2022_more].
Objective spectrum note: databases minimize a sum, treewidth a max; the
tensor-network log-sum sits between them.

## Key open problems

- No native SAT/ILP encoding of contraction-tree ordering exists; all
  exact routes go through graph reductions (treewidth, carving, linear
  ordering, join ordering). Gap acknowledged by multiple reports.
- Algorithm selection / portfolio design for ordering has no dedicated
  study, despite implicit portfolios everywhere.
- Sum-form lower bounds beyond max-form (carving/branch-width type) —
  our own impossibility result marks profile-based bounds as exhausted.
- Interaction of slicing with refinement-layer optimizers (both ours and
  Guerrero's NNI) is unexplored.
- Heterogeneous bond dimensions: the weighted ordering problem is
  largely open.

## Key bottlenecks

- Evaluation fragmentation: cost models (ops vs log-sum flops vs
  max-size), budgets, and hardware differ across papers; the Einsum
  Benchmark (NeurIPS 2024, not yet in bib) and OMEinsumContractionOrders
  benchmark suite are the consolidation attempts.
- Variance at scale: single-run comparisons dominate the literature;
  distributional reporting (as in our campaign) is rare.
- Refinement methods inherit the incumbent's blind spots: local tree
  moves cannot cross between bipartitions (our frozen-waist measurement;
  Guerrero's gap) — the structural cause is now measured but not yet
  theoretically characterized.
