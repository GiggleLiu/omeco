# Changelog

All notable changes to omeco are documented here. This file is the single
authoritative changelog; the mdBook appendix page includes it verbatim.
omeco adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

**Breaking:** `treesa::anneal_surgery_rounds` is removed; use
`anneal_refine_rounds(seed, code, sizes, config, rounds, &RoundsOptions::default())`
for identical behavior.

- Added `treesa::RoundsOptions` and `anneal_refine_rounds`, including an
  exactly work-matched cold-only control (`surgery: false`) and an incumbent
  ratchet that makes the loop monotone in its round count.
- Added opt-in `treesa::RoundsSchedule` (`RoundsOptions::schedule`). The
  default `Cold` variant is the historical span-gated fine-tuning pass;
  `BandReheatThenFront { switch_fraction }` reheats the waist cost band and then
  descends a continuous log-span freeze-out front, with the switch clamped
  between two band epochs and a fixed fraction of the planned sweeps.
- Waist-surgery side rebuilds now initialize from the restricted incumbent
  topology (`warm-restricted`), which the surgery ablation campaign showed
  strictly better than the historical greedy seed; `SurgeryScope::Local` is
  opt-in and rebuilds only a bounded ancestor around a deep waist. Defaults
  are warm-restricted + `Root`.
- `RoundsReport::fine_tune_sweeps_total` exposes deterministic fine-tuning work,
  and `optimize_treesa_seeded` supports matched optimizer repetitions. The
  resumable `surgery_ablation` example reports quality against both node visits
  and wall time with a Markdown summarizer; campaign artifacts are gitignored.
- Moved the companion paper's benchmark manifests, canonical instances,
  provenance gate, semantic verifiers, runners, generated artifacts, and CI
  checks to the `contraction-order-frontiers` repository. OMECO retains its
  reusable waist-surgery, `RoundTrace`, and opt-in waist-trace APIs and tests.

## 0.2.7 — 2026-07-29

**Default behavior change:** `TreeSA` now runs the structural
simplification front-end by default (`preprocess: true`) — output trees
may differ from 0.2.6 (same contraction result, typically equal or better
cost). Restore the old behavior with `preprocess: false`.

- New `TreeSA.surgery_iters` (default `0`/off): number of interleaved
  anneal–surgery rounds (Algorithm 1 of the companion paper) run on the
  reduced-network tree. Each round is one waist-surgery iteration followed by
  a cold span-gated fine-tuning pass with an incumbent ratchet. After
  splice-back, the candidate is guarded against the rounds-off baseline; the
  standalone reduced-network loop is monotone in its round count. Fully
  deterministic — rounds are counted, never timed — and auto-skipped for
  `DecompositionType::Path` configs, which the loop cannot preserve. `TreeSA`
  has no wall-clock knob; the low-level `waist_surgery::refine`/
  `refine_capped` APIs are unchanged and keep their `Duration` budget for
  power users.
- New `TreeSA.surgery_probability` (default `0.0`/off): mixes waist surgery
  directly into TreeSA. At each sweep, the configured probability replaces
  the local sweep with one waist-guided leaf prune-and-regraft move, accepted
  at the current inverse temperature. The rule never restarts cooling, invokes
  a timer, or launches a nested anneal; Rust and Python expose the same setting.
- New `treesa::anneal_surgery_rounds` + `RoundsReport`: the same interleaved
  loop as a standalone function, with separate per-round surgery candidate,
  raw fine-tuning endpoint, and retained-incumbent traces.
- Python: `TreeSA(preprocess=..., surgery_iters=..., surgery_probability=...)`,
  `simplify_then_optimize`, `waist_refine`, `SimplifyReport`, `WaistReport`.
- Fixed Python `TreeSA()` to delegate its default β schedule to Rust
  `TreeSA::default()` (`0..300`); the previous duplicate used `1..=300` and
  could shift benchmark quality by several bits.
- Committed paper-benchmark artifact under `benchmarks/paper/` (manifest,
  instances, deterministic runner, checker), plus a CI job that re-derives the
  small `ci` set on every PR and fails on a single changed field. A dedicated
  `figure2b` set checks the paper mechanism and renders a static SVG.
- New opt-in waist-call trace: `RoundTrace.waist`
  (`waist_surgery::WaistCallTrace`) records each round's exactly rescored
  incumbent and best FM waist cuts. Off in ordinary runs and behavior-neutral —
  no change to the proposal rule, RNG, acceptance, or `WaistReport`. A
  `waist_trace` benchmark set (five deterministic relabelings each of
  `surfacecode_d21` and `ksg`, 128 rounds each) regenerates the paper's dense
  mechanism evidence via `make paper-waist-trace`.
- `RoundTrace` gains `score_before`/`score_retained`: the configured
  multi-objective TreeSA score of the retained incumbent, which is the
  quantity the rounds ratchet actually guarantees is monotone (`tc_retained`
  can rise when another weighted term improves). Paper-benchmark artifacts
  emit the fields when the manifest sets `trace_scores: true` (implied by
  `trace_cuts`), and the `figure2b`/`waist_trace` verifiers now check the
  configured-score ratchet instead of raw `tc`.
- `Label` now requires `Ord` (all built-in label types already qualify).
- The Julia behavioral-alignment rule is retired; JSON interop is unchanged.

## 0.2.5 — 2026-06-30

### Added
- Comprehensive mdBook documentation following tropical-gemm standard
- Pretty printing for Python `NestedEinsum` with ASCII tree visualization
- PyTorch integration guide and examples
- GPU optimization guide with `rw_weight` configuration
- Slicing strategy guide for memory-constrained environments
- Troubleshooting guide with common issues and solutions
- API reference for Python and Rust APIs
- Performance benchmarks comparing Rust vs Julia
- Algorithm comparison guide (Greedy vs TreeSA)

### Changed
- Migrated documentation from scattered markdown files to structured mdBook
- Improved Python bindings with better `__str__` and `__repr__` methods

### Deprecated
- Legacy `docs/score_function_guide.md` (migrated to mdBook)

## 0.2.1 — 2024-01

### Fixed
- **Issue #6**: Hyperedge index preservation in contraction operations ([PR #7](https://github.com/GiggleLiu/omeco/pull/7))
  - Fixed `contract_tree!` macro to correctly preserve tensor indices during contraction
  - Added regression tests to verify hyperedge handling
  - Ensures contraction order matches input tensor order specified in `ixs`

### Added
- Test suite for hyperedge index preservation
- CI improvements for better test coverage

## 0.2.0 — 2024-01

### Added
- TreeSA (Tree-based Simulated Annealing) optimizer
- `TreeSA.fast()` preset for quick high-quality optimization
- Slicing support with `TreeSASlicer` for memory reduction
- `ScoreFunction` for configurable optimization objectives
- `contraction_complexity` and `sliced_complexity` functions
- Python bindings via PyO3
- `optimize_code` generic function accepting optimizer instances
- Read-write complexity (rwc) metric for GPU optimization

### Changed
- Improved API ergonomics with preset methods
- Better default parameters for optimizers

### Performance
- 1.4-1.5x faster than Julia OMEinsumContractionOrders.jl on benchmarks
- Efficient TreeSA implementation with better exploration

## 0.1.0 — 2023

### Added
- Initial release
- GreedyMethod optimizer
- Basic contraction order optimization
- Support for tensor networks with arbitrary indices
- Complexity calculation (time and space)
- Rust core library
- Basic documentation

### Features
- Greedy algorithm with configurable parameters
- Stochastic variants for improved solutions
- Efficient index handling with generic types
- HashMap-based dimension tracking
