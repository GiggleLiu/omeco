# Interleaved anneal–surgery rounds + reproducible paper benchmark

> Historical design record. OMECO no longer owns the benchmark workflow below;
> see the [Unreleased changelog](../../../CHANGELOG.md#unreleased).

Date: 2026-07-29. Approved by the maintainer in-session ("implement this plan
directly"). Builds on the usability spec
(`2026-07-29-usability-post-alignment-design.md`) and stacks on branch
`jg/usability` (PR #27, unreleased 0.2.7).

## Goal

1. Upgrade `TreeSA::surgery_iters` from a surgery-only post-pass to the
   paper's Algorithm 1: interleaved rounds of one waist-surgery step followed
   by a warm-started outer anneal — fully deterministic (no wall clock
   anywhere in the default path).
2. Turn the paper benchmark into a committed, self-verifying artifact:
   a deterministic runner over the 13 tracked paper instances
   (`research/benchmark/targets/*.json`), committed expected outputs, a CI
   job that re-runs a small subset and fails on drift.

The paper repo's "released artifact" table (PR C) consumes the committed
JSON later and is out of scope for this spec.

## Key insight

The paper's Algorithm 1 is deterministic in spec — `T ← SurgeryStep(T);
T ← Anneal(T, C_outer)` where `C_outer` is a *schedule*. Only the campaign
implementation chopped the outer anneal into wall-clock chunks. Aligning the
library with the spec (not the campaign binaries) makes every benchmark arm
deterministic, which removes reference hosts, repetitions, and error bars
from the benchmark entirely.

## 1. Interleaved rounds (library)

### New unit: `anneal_surgery_rounds` in `omeco/src/treesa.rs`

```rust
/// Per-round trace of the interleaved anneal–surgery loop (Algorithm 1 of
/// the paper). Scores are `config.score.evaluate(tc, sc, rwc)` of the
/// trajectory tree at the END of each round.
pub struct RoundsReport {
    pub rounds_run: u64,
    /// Round index (0-based) whose end-of-round tree was best; u64::MAX if
    /// the seed itself was never improved.
    pub best_round: u64,
    /// Trajectory score at the end of each round (len == rounds_run).
    pub round_scores: Vec<f64>,
    /// Sum of surgery_calls across all rounds' SurgeryStep invocations.
    pub surgery_calls_total: u64,
}

/// Run `rounds` rounds of: one waist-surgery step
/// (`refine_capped(_, _, _, Duration::MAX, 1)`) then one full warm-started
/// anneal pass over `config.betas` with `config.niters` sweeps per beta.
/// The trajectory chains (round r+1 starts from round r's annealed tree,
/// even if worse — annealing must be able to escape); the RETURNED tree is
/// the best tree seen anywhere along the trajectory, including the seed, so
/// the result is never worse than the seed and monotone-or-equal in
/// `rounds`. Deterministic: round r's anneal RNG is
/// `SmallRng::seed_from_u64(0xA55E + r)`; no wall clock is consulted
/// (`refine_capped` gets `Duration::MAX`).
pub fn anneal_surgery_rounds<L: Label>(
    seed: &NestedEinsum<L>,
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    rounds: u64,
) -> (NestedEinsum<L>, RoundsReport)
```

Implementation shape per round:

1. `let (t_surg, wr) = refine_capped(&trajectory, code, size_dict, Duration::MAX, 1);`
   accumulate `wr.surgery_calls` into `surgery_calls_total`.
2. Warm anneal: `prepare_warm_anneal(code, size_dict, &t_surg)`; if `None`
   (bare-leaf seed), stop early with `rounds_run` = rounds completed so far.
   Otherwise run the private `optimize_tree_sa(ctx.tree, &ctx.log2_sizes,
   &config.betas, config.niters, &config.score, DecompositionType::Tree,
   &mut rng, ctx.nedge)` and convert back with `warm_exprtree_to_nested`.
3. Score the annealed tree via `contraction_complexity` +
   `config.score.evaluate`; push to `round_scores`; update best-seen
   (strict `<` so ties keep the earlier tree); trajectory ← annealed tree.

Scoring of the seed happens once before the loop to initialize best-seen.

### Wiring in `optimize_treesa`

Replace the current tail:

```rust
if config.surgery_iters > 0 && config.decomposition_type != DecompositionType::Path {
    return Some(anneal_surgery_rounds(&tree, code, size_dict, config, config.surgery_iters).0);
}
Some(tree)
```

- `surgery_iters` is now "number of Algorithm-1 rounds"; each round costs
  roughly one full anneal of the network. Docs on the field, on
  `optimize_treesa`, `with_surgery_iters`, the mdBook default-pipeline page,
  the CHANGELOG 0.2.7 entry, and the Python docstring all state the new
  semantics. 0.2.7 is unreleased, so this is not a breaking change.
- Path decompositions skip surgery entirely (surgery rebuilds are not
  path-preserving), documented on the field like the preprocess auto-skip.
- `waist_surgery::refine` / `refine_capped` keep their current meaning
  (surgery-steps-only) as the low-level API.

### Tests (TDD; all deterministic)

1. Report shape: `rounds_run == rounds`, `round_scores.len() == rounds` on a
   benchmark graph; `best_round < rounds || best_round == u64::MAX`.
2. Determinism: two identical calls yield identical trees (compare via JSON
   serialization) and identical `round_scores`.
3. Never-worse + monotone: score(result, k=3) <= score(result, k=1) <=
   score(seed) on `benchmarks/graphs/grid_6x6.json` (or petersen).
4. Bare-leaf seed: single-tensor code returns the seed, `rounds_run == 0`.
5. Wrapper: `TreeSA::default().with_surgery_iters(2)` on grid_4x4 is
   deterministic across two runs and never worse than `surgery_iters(0)`.
6. Path guard: `TreeSA::path().with_surgery_iters(2)` result still satisfies
   the path-decomposition property.
7. Existing PR #27 tests written for the old surgery-only semantics
   (`test_surgery_iters_deterministic_and_never_worse`,
   `test_surgery_off_is_reproducible`, and any others that reference
   `surgery_iters` behavior) are UPDATED deliberately to the new semantics —
   this is maintainer-authorized (this spec) and they are this-session
   tests, not ALIGNED-WITH-JULIA heritage tests.

## 2. Paper benchmark artifact (`benchmarks/paper/`)

### Runner: `omeco/examples/paper_bench.rs`

CLI: `cargo run --release --example paper_bench -p omeco -- --manifest
benchmarks/paper/manifest.json --set ci --out benchmarks/paper/expected/ci.json`
(arg parsing by hand from `std::env::args`; no new dependencies).

- Reads instances via the existing JSON reader (`omeco::json`) from
  `research/benchmark/targets/` and `benchmarks/graphs/` (manifest gives
  relative paths from repo root; the binary resolves them against
  `--repo-root`, default `.`).
- Arms (all deterministic):
  - `greedy`: `GreedyMethod::default()`
  - `treesa`: `TreeSA::default()` (preprocess on, surgery off)
  - `treesa_rounds_k`: `TreeSA::default().with_surgery_iters(k)`; also
    records the per-round curve from `RoundsReport::round_scores` by calling
    the pipeline pieces directly (`optimize_treesa` for the seed path is
    fine — the runner calls `anneal_surgery_rounds` itself on the
    `surgery_iters == 0` result so it can emit the curve).
  - `treewidth`: only on instances the manifest marks `treewidth: true`
    (small/structured ones where it completes).
- Output JSON (serde, `BTreeMap`-ordered, floats rounded to 6 decimals
  before serialization to absorb cross-platform libm ulp noise):

```json
{
  "format": 1,
  "set": "ci",
  "results": [
    {"instance": "dbn_13", "arm": "treesa_rounds_4", "tc": 18.53, "sc": 10.0,
     "rwc": 14.2, "curve": [{"round": 0, "score": 19.1}, ...]}
  ]
}
```

  No timestamps, no hostnames, no omeco version inside the file (the git
  history carries provenance) — the file must be byte-stable.

### Manifest: `benchmarks/paper/manifest.json`

Two sets:

- `ci`: small, < ~5 min on a laptop — `dbn_13`, `qft_27`, `surfacecode_d9`,
  `reg3_250`, plus `petersen` from `benchmarks/graphs`. Arms: greedy,
  treesa (ntrials reduced via manifest override `"ntrials": 4`),
  treesa_rounds_2, treewidth where feasible (`dbn_13`, `petersen`).
- `full`: all 13 paper instances. Arms: greedy, treesa, treesa_rounds_8,
  treewidth on the manifest-marked feasible subset. Runtime target: under
  ~1 h on a laptop; if an instance/arm exceeds that in practice, the
  manifest may reduce its rounds — with the reduction visible in the
  manifest, never silent.

Manifest schema: `{"sets": {"ci": {"instances": [{"path": "...", "name":
"...", "treewidth": false}], "arms": {"treesa": {"ntrials": 4}, "rounds": 2}}}}`
— exact schema is the implementer's choice, but overrides must be explicit
per set, and the runner must fail loudly on unknown keys (no silent typos).

### Expected outputs + verification

- `benchmarks/paper/expected/ci.json` and `expected/full.json`, generated
  locally, committed.
- `benchmarks/paper/check.py` (stdlib-only Python): compares a fresh run
  against the committed file — bitwise equality preferred, tolerance
  `1e-9` on floats as the pass criterion (absorbs cross-platform libm
  differences), any structural difference (missing/extra results) is a hard
  fail. Exit code drives CI.
- Make targets:
  - `make paper-bench` — regenerate `expected/full.json` (the "I changed the
    algorithm, update the artifact" path).
  - `make paper-bench-check` — run the `ci` set to a temp file and verify
    against `expected/ci.json` via `check.py`.
- CI: new job `Paper bench` in `.github/workflows/test.yml` (ubuntu-only):
  build release, run the `ci` set, `check.py` against the committed file.
  This makes any future change that moves a paper number a visible CI
  failure instead of silent drift.
- `benchmarks/paper/README.md`: what this is (every number regenerates from
  the released library), how to run both sets, the determinism contract
  (bit-exact per machine, 1e-9 across platforms), and the calibration
  results from the final task.

### Calibration (empirical, part of this branch)

Run the `full` set once; record in `benchmarks/paper/README.md` a short
table comparing `treesa_rounds_8` tc/sc against the campaign's frozen
results (from the paper repo's `data/huawei_campaign.json`) for
`surfacecode_d13`, `surfacecode_d21`, `ksg`. No gate on matching campaign
records (those were multi-hour huawei runs); the deliverable is the honest
table plus one sentence in the README on how rounds trade off against the
campaign's wall-clock budgets.

## 3. Non-goals / unchanged

- `waist_surgery::refine`/`refine_capped` semantics (low-level, unchanged).
- Python API surface (no new functions; `surgery_iters` docstring updated).
- `GreedyMethod`, `Treewidth`, `TreeSASlicer` defaults.
- The paper repo (PR C happens after this branch merges to master).
- No new Rust dependencies.

## 4. Versioning / rollout

- Folded into the unreleased 0.2.7 (branch stacks on `jg/usability`;
  single PR based on `jg/usability`).
- CHANGELOG 0.2.7 entry amended: `surgery_iters` = interleaved Algorithm-1
  rounds; paper-bench artifact + CI gate added.
- Coverage stays above 95% (example binaries are not counted; keep the
  runner's logic in the example, library additions fully tested).
