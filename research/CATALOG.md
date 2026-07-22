# Catalog — beat-existing-optimizers

Every algorithm/software relevant to the topic. Status: `reproduced` (we ran
it), `pinned` (source-locked, not yet run), `paper-only` (no public code).

| Algorithm / software | Source | Status | Notes |
|---|---|---|---|
| omeco GreedyMethod / TreeSA / TreeSASlicer (baselines) | this repo, commit `6c83996f41` | reproduced | `RAYON_NUM_THREADS=1 cargo run --release --example benchmark -p omeco` on 2026-07-22: 9 graphs, e.g. reg3_250 greedy tc=66.61 (5.4 ms), TreeSA tc=47.02 sc=33.0 (1708 ms/run). Full output in `research/database/baselines.json`. |
| OMEinsumContractionOrders.jl TreeSA / GreedyMethod | github.com/TensorBFS/OMEinsumContractionOrders.jl @ `3397c24c` (local `~/.julia/dev`, 2026-07-10) | reproduced | Smoke test 2026-07-22: 4-matrix chain, `optimize_code(..., TreeSA(ntrials=1, niters=10))` → `tc=4.0 sc=2.0` (correct). Alignment reference per project CLAUDE.md. |
| cotengra HyperOptimizer (KaHyPar partitioning, Boltzmann greedy, SA refinement) [gray_2020_hyper] | github.com/jcmgray/cotengra @ `8ff0e9d3` | pinned | Not installed locally; install via pip when a head-to-head vs an attempt is needed. |
| FlowCutter (max-flow bisection tree decomposition) [strasser_2017_computing] | github.com/kit-algo/flow-cutter-pace17 @ `7f94541b` | pinned | C++, PACE 2017 heuristic-track submission; builds with make. |
| Tamaki PID-BT exact treewidth (meiji-e) [tamaki_2017_positive] | github.com/TCS-Meiji/PACE2017-TrackA @ `7278390f` | pinned | Java; exact optimal-width reference for small line graphs. |
| Tamaki heuristic-exact treewidth (UB/LB bracketing) [tamaki_2022_heuristic] | paper (SEA 2022), code github.com/twalgor/tw (unpinned) | paper-only | Pin + build only if exact lower bounds become load-bearing for validating headroom. |
| Kalachev–Panteleev–Yung SA tree optimizer + slicing [kalachev_2021_multi] | arXiv:2108.05665 | paper-only | Algorithmic ancestor of TreeSA; behavior reproduced in practice by both omeco and Julia TreeSA (rows above). |
| SA partitioning with subtree-shift moves [anon_2025_optimizing] | arXiv:2507.20667 | paper-only | No public code found; subtree-shift move is implementable from the paper. |
| Staudt et al. improved cut strategy [staudt_2024_improved] | SEA 2024, DOI 10.4230/LIPIcs.SEA.2024.27 | paper-only | Artifact availability unverified; the three fixes (tree/partition balance decoupling, free node, cost weights) are implementable from the paper. |
| Orgler–Blacher greedy cost-function portfolio [orgler_2024_optimizing] | arXiv:2405.09644 | paper-only | Cost-function formulas fully specified in the paper. |
| netcon exact contraction-sequence search [pfeifer_2013_faster] | arXiv:1304.6112 ancillary (MATLAB) | paper-only | Exact reference for ≲20–40 tensors; would need a small Rust re-implementation if made load-bearing. |
| Ibrahim et al. MLA order-then-tree DP [ibrahim_2022_constructing] | arXiv:2209.02895 | paper-only | Linear-order-to-optimal-tree DP is implementable from the recurrence in the paper. |
| Markov–Shi treewidth↔contraction theory [markov_2005_simulating] | arXiv:quant-ph/0511069 | paper-only | Theory only (cc(G) = tw(line graph)); no code applicable. |
| PACE-solver-to-contraction-order pipeline (QuickBB, freetdi, meiji-e comparison) [dumitrescu_2018_benchmarking] | arXiv:1807.04599 | paper-only | Conversion recipe (tree decomposition → elimination order → contraction order) documented in INSIGHTS.md. |
