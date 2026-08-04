# Interleaved Rounds + Paper Bench Implementation Plan

> Historical implementation record. OMECO no longer owns the benchmark
> workflow below; see the [Unreleased changelog](../../../CHANGELOG.md#unreleased).

**Goal:** Upgrade `TreeSA::surgery_iters` to the paper's interleaved Algorithm-1 rounds (surgery step + warm-started outer anneal, fully deterministic), and add a committed, CI-verified paper-benchmark artifact.

**Architecture:** New public `anneal_surgery_rounds` in `treesa.rs` composing existing pieces (`refine_capped` max_iters=1, `prepare_warm_anneal`, private `optimize_tree_sa`, `warm_exprtree_to_nested`); `optimize_treesa` delegates to it. A new example binary `paper_bench` runs deterministic arms over tracked instances against a manifest; expected JSON is committed and a CI job re-verifies a small set.

**Tech Stack:** Rust (no new dependencies), stdlib-only Python for the checker, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-29-interleaved-rounds-paper-bench-design.md` (in this branch).

## Global Constraints

- Branch `jg/interleaved-rounds`, stacked on `jg/usability` (base commit 9f85215). Worktree: `/private/tmp/claude-501/-Users-liujinguo-rcode-omeco/ec429751-cbe2-44e2-a096-1ead2502c7b2/scratchpad/interleaved`. **The shell cwd resets between commands — prefix EVERY command with `cd <worktree> &&`.**
- No wall-clock reads anywhere in the new default path: `refine_capped` always gets `std::time::Duration::MAX`; RNG seeds are fixed constants.
- Anneal RNG seed for round `r` is exactly `SmallRng::seed_from_u64(0xA55E + r)`.
- No panics/unwraps in production code (tests may unwrap). All public items get doc comments with examples. Clippy clean under `-D warnings`; run `make check-all` before each commit.
- Coverage must stay ≥ 95% (tarpaulin, `--packages omeco`; example binaries are excluded — keep untested logic in the example, tested logic in the library).
- Benchmark output JSON must be byte-stable: floats rounded to 6 decimals before serialization, keys ordered (BTreeMap / ordered struct fields), no timestamps/hostnames/versions in the file.
- Updating this-session tests from PR #27 that encode the OLD surgery-only semantics is authorized by the spec; ALIGNED-WITH-JULIA heritage tests must not change.
- Existing low-level APIs (`waist_surgery::refine`, `refine_capped`) keep their current semantics.

---

### Task 1: `anneal_surgery_rounds` + `RoundsReport`

**Files:**
- Modify: `omeco/src/treesa.rs` (new items after `warm_exprtree_to_nested`, ~line 956; tests in the existing `mod tests`)

**Interfaces:**
- Consumes: `refine_capped` (already imported in treesa.rs), `prepare_warm_anneal`, `optimize_tree_sa` (private, same module), `warm_exprtree_to_nested`, `crate::contraction_complexity`.
- Produces (used by Tasks 2 and 3):
  ```rust
  pub struct RoundsReport { pub rounds_run: u64, pub best_round: u64, pub round_scores: Vec<f64>, pub surgery_calls_total: u64 }
  pub fn anneal_surgery_rounds<L: Label>(seed: &NestedEinsum<L>, code: &EinCode<L>, size_dict: &HashMap<L, usize>, config: &TreeSA, rounds: u64) -> (NestedEinsum<L>, RoundsReport)
  ```

- [ ] **Step 1: Write failing tests** in `mod tests` of `treesa.rs` (reuse the existing `load_benchmark_graph` helper at ~line 1905):

```rust
#[test]
fn test_rounds_report_shape_and_determinism() {
    let (code, sizes) = load_benchmark_graph("petersen");
    let config = TreeSA::fast();
    let seed = optimize_treesa(&code, &sizes, &TreeSA { surgery_iters: 0, ..config.clone() }).unwrap();
    let (t1, r1) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 3);
    let (t2, r2) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 3);
    assert_eq!(r1.rounds_run, 3);
    assert_eq!(r1.round_scores.len(), 3);
    assert!(r1.best_round == u64::MAX || r1.best_round < 3);
    // Determinism: identical trees and traces
    assert_eq!(
        crate::json::to_json_string(&t1).unwrap(),
        crate::json::to_json_string(&t2).unwrap()
    );
    assert_eq!(r1.round_scores, r2.round_scores);
    assert_eq!(r1.surgery_calls_total, r2.surgery_calls_total);
}

#[test]
fn test_rounds_never_worse_and_monotone() {
    let (code, sizes) = load_benchmark_graph("grid_6x6");
    let config = TreeSA::fast();
    let seed = optimize_treesa(&code, &sizes, &TreeSA { surgery_iters: 0, ..config.clone() }).unwrap();
    let score_of = |t: &NestedEinsum<usize>| {
        let cc = crate::contraction_complexity(t, &sizes, &code.ixs);
        config.score.evaluate(cc.tc, cc.sc, cc.rwc)
    };
    let s0 = score_of(&seed);
    let (t1, _) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 1);
    let (t3, _) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 3);
    assert!(score_of(&t1) <= s0);
    assert!(score_of(&t3) <= score_of(&t1));
}

#[test]
fn test_rounds_bare_leaf_seed() {
    let code: EinCode<usize> = EinCode::new(vec![vec![0, 1]], vec![0, 1]);
    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
    let seed = NestedEinsum::leaf(0);
    let (t, r) = anneal_surgery_rounds(&seed, &code, &sizes, &TreeSA::fast(), 2);
    assert_eq!(r.rounds_run, 0);
    assert!(r.round_scores.is_empty());
    assert_eq!(t.leaf_count(), 1);
}
```

(If `to_json_string` needs a different call shape for `NestedEinsum`, wrap via the same path the existing determinism test `test_surgery_off_is_reproducible` uses — copy its comparison method.)

- [ ] **Step 2: Run tests, verify they fail** with "cannot find function `anneal_surgery_rounds`": `cd <worktree> && cargo test -p omeco anneal_surgery -- --nocapture` (compile error is the expected RED here).

- [ ] **Step 3: Implement** (doc comments per the spec, with a doctest example on `anneal_surgery_rounds` using a small chain code):

```rust
pub struct RoundsReport {
    pub rounds_run: u64,
    pub best_round: u64,
    pub round_scores: Vec<f64>,
    pub surgery_calls_total: u64,
}

pub fn anneal_surgery_rounds<L: Label>(
    seed: &NestedEinsum<L>,
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    rounds: u64,
) -> (NestedEinsum<L>, RoundsReport) {
    use rand::SeedableRng;
    let score_of = |t: &NestedEinsum<L>| {
        let cc = crate::contraction_complexity(t, size_dict, &code.ixs);
        config.score.evaluate(cc.tc, cc.sc, cc.rwc)
    };
    let mut best = seed.clone();
    let mut best_score = score_of(&best);
    let mut best_round = u64::MAX;
    let mut trajectory = seed.clone();
    let mut report = RoundsReport {
        rounds_run: 0,
        best_round: u64::MAX,
        round_scores: Vec::new(),
        surgery_calls_total: 0,
    };
    for r in 0..rounds {
        let (t_surg, wr) =
            refine_capped(&trajectory, code, size_dict, std::time::Duration::MAX, 1);
        report.surgery_calls_total += wr.surgery_calls;
        let Some(ctx) = prepare_warm_anneal(code, size_dict, &t_surg) else {
            break;
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0xA55E + r);
        let annealed = optimize_tree_sa(
            ctx.tree,
            &ctx.log2_sizes,
            &config.betas,
            config.niters,
            &config.score,
            DecompositionType::Tree,
            &mut rng,
            ctx.nedge,
        );
        let cand = warm_exprtree_to_nested(&annealed, code, &ctx.labels);
        let surg_score = score_of(&t_surg);
        if surg_score < best_score {
            best_score = surg_score;
            best = t_surg;
            best_round = r;
        }
        let cand_score = score_of(&cand);
        if cand_score < best_score {
            best_score = cand_score;
            best = cand.clone();
            best_round = r;
        }
        report.round_scores.push(cand_score);
        report.rounds_run = r + 1;
        trajectory = cand;
    }
    report.best_round = best_round;
    (best, report)
}
```

Doc comments must state: trajectory chains (round r+1 anneals from round r's result even if worse), returned tree is best-seen including the seed (never-worse, monotone-or-equal in `rounds`), determinism contract (fixed seeds, no wall clock), and per-round cost ≈ one full anneal of the network.

- [ ] **Step 4: Run tests, verify pass**: `cd <worktree> && cargo test -p omeco` (full suite — nothing else may regress).
- [ ] **Step 5: `make check-all`, commit**: `git add omeco/src/treesa.rs && git commit -m "TreeSA: anneal_surgery_rounds — deterministic interleaved Algorithm-1 loop"`.

---

### Task 2: Wire `surgery_iters` to rounds; Path guard; migrate old-semantics tests; docs

**Files:**
- Modify: `omeco/src/treesa.rs` (`optimize_treesa` tail ~line 768; `surgery_iters` field docs ~line 55; `with_surgery_iters` docs; existing tests that encode old semantics)
- Modify: `docs/src/algorithms/default-pipeline.md` (surgery section)
- Modify: `CHANGELOG.md` (0.2.7 entry)
- Modify: `omeco-python/src/lib.rs` (surgery_iters docstring text only — no signature change)
- Modify: `omeco-python/tests/` (any test asserting old surgery semantics)

**Interfaces:**
- Consumes: `anneal_surgery_rounds` from Task 1.
- Produces: `optimize_treesa` with `surgery_iters > 0` ⇒ Algorithm-1 rounds; Path decomposition skips surgery entirely.

- [ ] **Step 1: Write failing tests** (in `treesa.rs` `mod tests`):

```rust
#[test]
fn test_surgery_iters_runs_interleaved_rounds() {
    // The wrapper with surgery_iters=k must equal seed-then-rounds composed by hand.
    let (code, sizes) = load_benchmark_graph("grid_4x4");
    let base = TreeSA::fast();
    let seed = optimize_treesa(&code, &sizes, &TreeSA { surgery_iters: 0, ..base.clone() }).unwrap();
    let (by_hand, _) = anneal_surgery_rounds(&seed, &code, &sizes, &base, 2);
    let wrapped =
        optimize_treesa(&code, &sizes, &TreeSA { surgery_iters: 2, ..base }).unwrap();
    assert_eq!(
        crate::json::to_json_string(&wrapped).unwrap(),
        crate::json::to_json_string(&by_hand).unwrap()
    );
}

#[test]
fn test_path_decomposition_skips_surgery() {
    let (code, sizes) = load_benchmark_graph("chain_10");
    let cfg = TreeSA::path().with_surgery_iters(2);
    let with_surgery = optimize_treesa(&code, &sizes, &cfg).unwrap();
    let without = optimize_treesa(&code, &sizes, &TreeSA::path()).unwrap();
    assert_eq!(
        crate::json::to_json_string(&with_surgery).unwrap(),
        crate::json::to_json_string(&without).unwrap()
    );
}
```

(If an existing path-property assertion helper exists from `test_path_decomposition_holds_with_preprocess_true`, additionally assert the path property on `with_surgery` with it.)

- [ ] **Step 2: Verify RED**: `test_surgery_iters_runs_interleaved_rounds` fails (wrapper currently calls `refine_capped(…, surgery_iters)`); `test_path_decomposition_skips_surgery` may fail or pass — record which.

- [ ] **Step 3: Implement wrapper change** — replace the tail of `optimize_treesa`:

```rust
    if config.surgery_iters > 0 && config.decomposition_type != DecompositionType::Path {
        return Some(
            anneal_surgery_rounds(&tree, code, size_dict, config, config.surgery_iters).0,
        );
    }
    Some(tree)
```

- [ ] **Step 4: Migrate old-semantics tests.** Grep `surgery_iters` across `omeco/src/` and `omeco-python/`. Tests from PR #27 that assert the surgery-only behavior (at minimum `test_surgery_iters_deterministic_and_never_worse`; check `test_surgery_off_is_reproducible`, `test_treesa_pipeline_defaults`, Python tests) are updated to the rounds semantics: determinism and never-worse assertions stay, any assertion tying the result to `refine_capped(…, k)` output is replaced by the Task-1/Task-2 equivalences. Keep test names describing behavior. Do NOT touch ALIGNED-WITH-JULIA tests.
- [ ] **Step 5: Update docs.**
  - `surgery_iters` field + `with_surgery_iters` + `optimize_treesa` rustdoc: rounds semantics, cost ≈ one anneal per round, Path auto-skip, determinism.
  - `docs/src/algorithms/default-pipeline.md`: surgery section now describes the interleaved loop (SurgeryStep → outer anneal, best-seen returned); keep the existing attribution stats (76% surface-code d21, 31% king graph) tied to the paper's campaign, and state each round ≈ one full anneal.
  - `CHANGELOG.md` 0.2.7: amend the surgery bullet — `surgery_iters` now counts interleaved anneal–surgery rounds (Algorithm 1 of the paper), deterministic; low-level `waist_surgery::refine`/`refine_capped` unchanged.
  - `omeco-python/src/lib.rs`: surgery_iters docstring says "number of interleaved anneal–surgery rounds (each ≈ one full anneal); 0 disables".
- [ ] **Step 6: Verify GREEN + full gates**: `cd <worktree> && make check-all && make python-dev && make python-test`.
- [ ] **Step 7: Commit**: `git add -A && git commit -m "TreeSA: surgery_iters now runs interleaved anneal-surgery rounds"`.

---

### Task 3: `paper_bench` runner + manifest + checker

**Files:**
- Create: `omeco/examples/paper_bench.rs`
- Create: `benchmarks/paper/manifest.json`
- Create: `benchmarks/paper/check.py`
- Create: `benchmarks/paper/README.md` (structure + usage; calibration table left as an explicitly labeled "pending Task 5" section is NOT allowed — instead write the README fully except the calibration table, which Task 5 appends as a new section)

**Interfaces:**
- Consumes: `optimize_treesa`, `anneal_surgery_rounds` (Task 1), `GreedyMethod`, `Treewidth` public APIs, `contraction_complexity`.
- Produces: CLI contract used by Task 4/5 and CI:
  `cargo run --release --example paper_bench -p omeco -- --manifest benchmarks/paper/manifest.json --set ci --out <file> [--repo-root <dir>]`
  and `python3 benchmarks/paper/check.py <fresh.json> <expected.json>` (exit 0 = pass; float tolerance 1e-9; structural mismatch = fail).

- [ ] **Step 1: Manifest.** Instance paths are repo-root-relative. Exact content:

```json
{
  "sets": {
    "ci": {
      "arms": {
        "greedy": {},
        "treesa": {"ntrials": 4},
        "treesa_rounds": {"ntrials": 4, "rounds": 2}
      },
      "instances": [
        {"name": "petersen", "path": "benchmarks/graphs/petersen.json", "treewidth": true},
        {"name": "dbn_13", "path": "research/benchmark/targets/dbn_13.json", "treewidth": true},
        {"name": "qft_27", "path": "research/benchmark/targets/qft_27.json", "treewidth": false},
        {"name": "surfacecode_d9", "path": "research/benchmark/targets/surfacecode_d9.json", "treewidth": false},
        {"name": "reg3_250", "path": "research/benchmark/targets/reg3_250.json", "treewidth": false}
      ]
    },
    "full": {
      "arms": {
        "greedy": {},
        "treesa": {},
        "treesa_rounds": {"rounds": 8}
      },
      "instances": [
        {"name": "dbn_13", "path": "research/benchmark/targets/dbn_13.json", "treewidth": true},
        {"name": "qft_27", "path": "research/benchmark/targets/qft_27.json", "treewidth": false},
        {"name": "surfacecode_d9", "path": "research/benchmark/targets/surfacecode_d9.json", "treewidth": false},
        {"name": "surfacecode_d13", "path": "research/benchmark/targets/surfacecode_d13.json", "treewidth": false},
        {"name": "surfacecode_d17", "path": "research/benchmark/targets/surfacecode_d17.json", "treewidth": false},
        {"name": "surfacecode_d21", "path": "research/benchmark/targets/surfacecode_d21.json", "treewidth": false},
        {"name": "ksg", "path": "research/benchmark/targets/ksg.json", "treewidth": false},
        {"name": "nqueens_28", "path": "research/benchmark/targets/nqueens_28.json", "treewidth": false},
        {"name": "reg3_250", "path": "research/benchmark/targets/reg3_250.json", "treewidth": false},
        {"name": "reg3_1000", "path": "research/benchmark/targets/reg3_1000.json", "treewidth": false},
        {"name": "rqc_97_m24", "path": "research/benchmark/targets/rqc_97_m24.json", "treewidth": false},
        {"name": "sycamore_m20", "path": "research/benchmark/targets/sycamore_m20.json", "treewidth": false},
        {"name": "sycamore_53_20_0", "path": "research/benchmark/targets/sycamore_53_20_0.json", "treewidth": false}
      ]
    }
  }
}
```

  Arm-config keys allowed: `ntrials` (usize, overrides `TreeSA::default().ntrials`), `niters`, `rounds` (u64, `treesa_rounds` only). Runner must error (nonzero exit, message naming the key) on any unknown key in arms/instances/sets — no silent typos.

- [ ] **Step 2: Runner.** Hand-rolled arg parsing (`std::env::args`), no new deps. Loads the manifest, resolves instance paths against `--repo-root` (default `.`). Instance labels are `usize` (all files use integer labels). For each instance × arm:
  - `greedy` → `optimize_code(&code, &sizes, &GreedyMethod::default())`.
  - `treesa` → `TreeSA::default()` with manifest overrides applied, `surgery_iters: 0`.
  - `treesa_rounds` → seed from the `treesa` config result (recompute with the same overrides), then `anneal_surgery_rounds(&seed, …, rounds)`; emit `curve` = `[{round, score}]` from `round_scores` (6-decimal rounding), result tree = returned best.
  - `treewidth` (only if instance flag true) → `Treewidth` default optimizer.
  - Every arm records `tc`, `sc`, `rwc` from `contraction_complexity`, rounded via `(x * 1e6).round() / 1e6` before insertion.
  - Output: `{"format": 1, "set": <set>, "results": [...]}` with `results` sorted by `(instance, arm)`; serialize with `serde_json::to_string_pretty` over ordered structs (field order fixed by struct definition; no HashMap anywhere in the output path). Write with a trailing newline.
  - The example must not `panic!`/`unwrap` on user errors (bad path, bad manifest): print to stderr and `std::process::exit(2)`. (`unwrap` on programmatic invariants is acceptable in an example binary but prefer `expect` with a message.)
- [ ] **Step 3: Checker `check.py`** (stdlib only, Python 3.8+): loads two JSON files; fails (exit 1, message) if `format`/`set` differ, if the `(instance, arm)` key sets differ, or if any numeric leaf differs by more than 1e-9 (recursive compare; ints exact). Prints `OK: N results compared` on success.
- [ ] **Step 4: Smoke test**: `cd <worktree> && cargo run --release --example paper_bench -p omeco -- --manifest benchmarks/paper/manifest.json --set ci --out /tmp/ci1.json && cargo run --release --example paper_bench -p omeco -- --manifest benchmarks/paper/manifest.json --set ci --out /tmp/ci2.json && python3 benchmarks/paper/check.py /tmp/ci1.json /tmp/ci2.json` → must print OK (also proves determinism). `diff /tmp/ci1.json /tmp/ci2.json` must be empty (bit-stability on one machine). Record the ci-set wall time in the report; if > 8 min, reduce `ci` arm overrides (`ntrials`, `rounds`) in the manifest and note it.
- [ ] **Step 5: README** (`benchmarks/paper/README.md`): what the artifact is (every number regenerates from the released library — no reference host), how to run ci/full, the determinism contract (bit-exact per machine; 1e-9 across platforms due to libm), manifest override policy (reductions must be visible in the manifest, never silent), and a pointer to the paper repo for the frozen campaign data.
- [ ] **Step 6: `make check-all`** (the example must compile under clippy `-D warnings`), commit: `git add omeco/examples/paper_bench.rs benchmarks/paper && git commit -m "paper_bench: deterministic benchmark runner + manifest + checker"`.

---

### Task 4: Expected `ci` output, Make targets, CI job

**Files:**
- Create: `benchmarks/paper/expected/ci.json` (generated)
- Modify: `Makefile` (two targets)
- Modify: `.github/workflows/test.yml` (new job)

**Interfaces:**
- Consumes: Task 3's CLI + checker contracts.
- Produces: `make paper-bench` (regenerates `expected/full.json`), `make paper-bench-check` (ci-set verification), CI job `paper-bench`.

- [ ] **Step 1: Generate + commit expected ci**: `cd <worktree> && cargo run --release --example paper_bench -p omeco -- --manifest benchmarks/paper/manifest.json --set ci --out benchmarks/paper/expected/ci.json`.
- [ ] **Step 2: Make targets** (match existing Makefile style/tabs):

```makefile
paper-bench:  ## Regenerate the committed full paper benchmark artifact
	cargo run --release --example paper_bench -p omeco -- --manifest benchmarks/paper/manifest.json --set full --out benchmarks/paper/expected/full.json

paper-bench-check:  ## Re-run the ci set and verify against the committed artifact
	cargo run --release --example paper_bench -p omeco -- --manifest benchmarks/paper/manifest.json --set ci --out /tmp/paper_bench_ci.json
	python3 benchmarks/paper/check.py /tmp/paper_bench_ci.json benchmarks/paper/expected/ci.json
```

(If the Makefile documents targets via a help convention, follow it.)

- [ ] **Step 3: CI job** in `.github/workflows/test.yml`, following the file's existing checkout/toolchain/cache steps:

```yaml
  paper-bench:
    name: Paper bench
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Verify committed benchmark artifact (ci set)
        run: make paper-bench-check
```

- [ ] **Step 4: Verify locally**: `cd <worktree> && make paper-bench-check` → OK.
- [ ] **Step 5: Commit**: `git add benchmarks/paper/expected/ci.json Makefile .github/workflows/test.yml && git commit -m "paper-bench: committed ci artifact + make targets + CI verification job"`.

---

### Task 5: Full artifact + calibration table

**Files:**
- Create: `benchmarks/paper/expected/full.json` (generated; long run)
- Modify: `benchmarks/paper/README.md` (append `## Calibration vs the frozen campaign` section)

**Interfaces:**
- Consumes: `make paper-bench` (Task 4); campaign baselines from `/Users/liujinguo/rcode/contraction-order-frontiers/data/huawei_campaign.json` (read-only reference — do NOT modify the paper repo).

- [ ] **Step 1: Run the full set** (background, may take ~1 h): `cd <worktree> && make paper-bench`. If any instance × arm exceeds ~20 min alone, stop, reduce that arm in the manifest (`rounds` or `ntrials`) — the reduction stays visible in the manifest — and restart; note it in the README.
- [ ] **Step 2: Verify determinism of the artifact**: re-run only the smallest three full-set instances to a temp file with a temporary manifest copy is NOT needed — instead run `make paper-bench-check` (ci set) once more after the full run to confirm the build is unchanged; then `git add benchmarks/paper/expected/full.json`.
- [ ] **Step 3: Calibration section.** Read `treesa_rounds_8` tc/sc for `surfacecode_d13`, `surfacecode_d21`, `ksg` from `expected/full.json`; read the campaign's best tc/sc for the same instances from the paper repo's `data/huawei_campaign.json` (field names: inspect the file). Append a table: instance | campaign best tc (huawei, wall-clock budget) | treesa_rounds_8 tc (deterministic) | gap. One paragraph: rounds are schedule-budgeted, campaign was wall-clock-budgeted on huawei; the artifact's claim is bit-reproducibility, not record-matching; larger `rounds` closes the gap at proportional deterministic cost.
- [ ] **Step 4: Commit**: `git add benchmarks/paper/expected/full.json benchmarks/paper/README.md && git commit -m "paper-bench: full committed artifact + calibration vs frozen campaign"`.

---

## Self-review notes

- Type consistency: `anneal_surgery_rounds` signature identical in Tasks 1–3; `RoundsReport.round_scores: Vec<f64>` consumed as `curve` in Task 3.
- All test code is concrete; graph names (`petersen`, `grid_6x6`, `grid_4x4`, `chain_10`) exist in `benchmarks/graphs/`.
- Spec coverage: §1 → Tasks 1–2; §2 runner/manifest/checker → Task 3; expected/Make/CI → Task 4; calibration → Task 5; CHANGELOG/mdBook/Python docs → Task 2.
