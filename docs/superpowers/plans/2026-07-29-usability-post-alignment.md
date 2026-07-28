# Post-Alignment Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the paper pipeline (preprocess → TreeSA → optional waist surgery) the flags-on-TreeSA default path in Rust and Python, and retire the Julia behavioral-alignment rule.

**Architecture:** `TreeSA` gains `preprocess: bool` (default true) and `surgery_budget: f64` seconds (default 0.0). `optimize_treesa` wraps its existing trial loop (renamed `optimize_treesa_core`) with simplify/splice and an optional `refine` post-pass. Python mirrors the flags and additionally exposes `simplify_then_optimize` / `waist_refine` with report types. Docs and CLAUDE.md updated; patch version bump.

**Tech Stack:** Rust (edition 2021, MSRV 1.70, clippy -D warnings), PyO3/maturin via `make python-dev`, pytest, mdBook.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-usability-post-alignment-design.md` (same repo).
- Worktree: `/private/tmp/claude-501/-Users-liujinguo-rcode-omeco/ec429751-cbe2-44e2-a096-1ead2502c7b2/scratchpad/usability`, branch `jg/usability`, base = master `6b3d95c`.
- Defaults exactly: `preprocess: true`, `surgery_budget: 0.0`.
- No panics/unwraps in production code; doc comments with runnable examples on all public items.
- Gate before every commit that touches Rust: `make check-all` (fmt + clippy + tests). Python tasks: `make python-dev && make python-test`.
- JSON interop with Julia unchanged. `GreedyMethod`, `Treewidth`, `TreeSASlicer`, `ExhaustiveSearch` behavior unchanged.
- Do not modify `research/**` or `articles/**`.

---

### Task 1: `Ord` as a `Label` supertrait

`preprocess::simplify` requires `L: Ord` for determinism; `optimize_treesa` is generic over `L: Label` through the `CodeOptimizer` trait, whose signature cannot add bounds per-impl. Fold `Ord` into `Label`.

**Files:**
- Modify: `omeco/src/label.rs:22` (trait bound)
- Modify: `omeco/src/preprocess.rs` (drop now-redundant `+ Ord` bounds on `simplify`, `simplify_then_optimize`)

**Interfaces:**
- Produces: `pub trait Label: Clone + Eq + Ord + Hash + Debug + Send + Sync + 'static {}` — later tasks call `simplify::<L: Label>` without extra bounds.

- [ ] **Step 1: Change the trait**

In `omeco/src/label.rs` replace:

```rust
pub trait Label: Clone + Eq + Hash + Debug + Send + Sync + 'static {}
```

with:

```rust
pub trait Label: Clone + Eq + Ord + Hash + Debug + Send + Sync + 'static {}
```

Every provided impl (`char`, `u8`–`u64`, `usize`, `i8`–`i64`, `isize`, `String`, whatever the file lists) already satisfies `Ord`; leave the impls untouched. Update the trait's doc comment to mention `Ord` (needed by deterministic preprocessing).

- [ ] **Step 2: Drop redundant bounds**

In `omeco/src/preprocess.rs` change `pub fn simplify<L: Label + Ord>` → `pub fn simplify<L: Label>` and `pub fn simplify_then_optimize<L: Label + Ord, O: CodeOptimizer>` → `pub fn simplify_then_optimize<L: Label, O: CodeOptimizer>`. Update the doc sentence "requires `L: Ord` for that determinism" to say `Ord` comes with `Label`.

- [ ] **Step 3: Run the gate**

Run: `make check-all`
Expected: clean (this is a pure widening for all in-repo label types).

- [ ] **Step 4: Commit**

```bash
git add omeco/src/label.rs omeco/src/preprocess.rs
git commit -m "Label: require Ord (needed by deterministic preprocessing)"
```

---

### Task 2: TreeSA config fields

**Files:**
- Modify: `omeco/src/treesa.rs:18-33` (struct), `:44-57` (Default), `:59-96` (new/fast/path), builder-setter block below them
- Modify: `omeco/src/treesa.rs` tests that construct `TreeSA { ... }` literally (`test_trial_selection_ranks_by_emitted_tree_cost` around line 845 builds a full literal — give it `preprocess: false, surgery_budget: 0.0` so it keeps testing the bare trial loop)
- Test: inline `#[cfg(test)]` in `omeco/src/treesa.rs`

**Interfaces:**
- Produces: `TreeSA { pub preprocess: bool, pub surgery_budget: f64, /* existing fields */ }`; `TreeSA::with_preprocess(bool) -> Self`; `TreeSA::with_surgery_budget(f64) -> Self`. Task 3/4 read `config.preprocess` / `config.surgery_budget`; Task 6 sets both from Python.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_treesa_pipeline_defaults() {
    let config = TreeSA::default();
    assert!(config.preprocess);
    assert_eq!(config.surgery_budget, 0.0);
    let fast = TreeSA::fast();
    assert!(fast.preprocess);
    assert_eq!(fast.surgery_budget, 0.0);
    let tuned = TreeSA::default()
        .with_preprocess(false)
        .with_surgery_budget(30.0);
    assert!(!tuned.preprocess);
    assert_eq!(tuned.surgery_budget, 30.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p omeco --lib test_treesa_pipeline_defaults`
Expected: compile error — no field `preprocess`.

- [ ] **Step 3: Implement**

Add to the struct (with doc comments):

```rust
    /// Run the structural simplification front-end before annealing
    /// (simplify → optimize the reduced network → splice back). Deterministic
    /// and exactness-preserving; see [`crate::preprocess`].
    pub preprocess: bool,
    /// Wall-clock budget in seconds for the waist-surgery post-pass on the
    /// selected tree; `0.0` disables it. With a positive budget the result is
    /// never worse than without, but is not reproducible across machines.
    /// See [`crate::waist_surgery`].
    pub surgery_budget: f64,
```

`Default`: add `preprocess: true, surgery_budget: 0.0`. `TreeSA::new` builds a full literal — append the same two defaults there. `fast()` and `path()` use `..Default::default()`, so they inherit automatically. Add builder setters next to `with_ntrials`:

```rust
    /// Enable or disable the structural simplification front-end.
    pub fn with_preprocess(mut self, preprocess: bool) -> Self {
        self.preprocess = preprocess;
        self
    }

    /// Set the waist-surgery wall-clock budget in seconds (0.0 disables).
    pub fn with_surgery_budget(mut self, seconds: f64) -> Self {
        self.surgery_budget = seconds;
        self
    }
```

Fix the full-literal test construction in `test_trial_selection_ranks_by_emitted_tree_cost` by adding `preprocess: false, surgery_budget: 0.0,` to its `TreeSA { ... }`.

- [ ] **Step 4: Run the gate**

Run: `make check-all`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add omeco/src/treesa.rs
git commit -m "TreeSA: preprocess and surgery_budget config fields"
```

---

### Task 3: Preprocess stage in `optimize_treesa`

**Files:**
- Modify: `omeco/src/treesa.rs:658` region — rename the existing `pub fn optimize_treesa` body to a private `fn optimize_treesa_core` with the identical signature, and add a new `pub fn optimize_treesa` wrapper
- Test: inline tests in `omeco/src/treesa.rs`

**Interfaces:**
- Consumes: `crate::preprocess::{simplify, splice}` (`simplify(&EinCode<L>, &HashMap<L, usize>) -> Simplified<L>` with fields `code`, `subtrees`, `report`; `splice(&NestedEinsum<L>, &[NestedEinsum<L>]) -> NestedEinsum<L>`).
- Produces: `optimize_treesa` honoring `config.preprocess`; `optimize_treesa_core` for Task 4's test to call the bare loop.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn test_default_pipeline_preprocess_preserves_interfaces() {
    // Matrix chain: simplify collapses it; the spliced tree must keep all leaves.
    let code = EinCode::new(
        vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd'], vec!['d', 'e']],
        vec!['a', 'e'],
    );
    let sizes: HashMap<char, usize> =
        [('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)].into();
    let tree = optimize_treesa(&code, &sizes, &TreeSA::fast()).unwrap();
    assert_eq!(tree.leaf_count(), 4);
    let cc = crate::contraction_complexity(&tree, &sizes, &code.ixs);
    assert!(cc.tc.is_finite());
}

#[test]
fn test_default_pipeline_quality_on_benchmark_graphs() {
    // Spec §4: default (preprocess on) is within 0.5 bits of the bare loop on
    // the benchmark graphs; exact equality where simplify is a no-op (reg3).
    for (name, no_op) in [("grid_4x4", false), ("reg3_50", true)] {
        let graph_json = std::fs::read_to_string(format!("../benchmarks/graphs/{name}.json")).unwrap();
        let graph: serde_json::Value = serde_json::from_str(&graph_json).unwrap();
        let mut ixs: Vec<Vec<usize>> = Vec::new();
        for edge in graph["edge_list"].as_array().unwrap() {
            let a = edge.as_array().unwrap();
            ixs.push(vec![a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize]);
        }
        // line-graph convention used by the other benchmark tests: tensors are
        // vertices, labels are edge endpoints — mirror test_reg3_220_treesa's
        // construction in omeco/src/lib.rs if it differs from the above.
        let code = EinCode::new(ixs, vec![]);
        let sizes: HashMap<usize, usize> =
            code.unique_labels().into_iter().map(|l| (l, 2)).collect();
        let cfg = TreeSA::fast();
        let with_pre = optimize_treesa(&code, &sizes, &cfg).unwrap();
        let without = optimize_treesa(&code, &sizes, &cfg.clone().with_preprocess(false)).unwrap();
        let tc_pre = crate::contraction_complexity(&with_pre, &sizes, &code.ixs).tc;
        let tc_raw = crate::contraction_complexity(&without, &sizes, &code.ixs).tc;
        if no_op {
            assert!((tc_pre - tc_raw).abs() < 1e-9, "{name}: {tc_pre} vs {tc_raw}");
        } else {
            assert!(tc_pre <= tc_raw + 0.5, "{name}: {tc_pre} > {tc_raw} + 0.5");
        }
    }
}

#[test]
fn test_preprocess_off_matches_core_loop() {
    let code = EinCode::new(
        vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'a']],
        Vec::<char>::new(),
    );
    let sizes: HashMap<char, usize> = [('a', 4), ('b', 4), ('c', 4)].into();
    let config = TreeSA::fast().with_preprocess(false);
    let via_public = optimize_treesa(&code, &sizes, &config).unwrap();
    let via_core = optimize_treesa_core(&code, &sizes, &config).unwrap();
    assert_eq!(
        format!("{via_public:?}"), format!("{via_core:?}"),
        "preprocess=false must be byte-identical to the bare trial loop"
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p omeco --lib test_preprocess_off_matches_core_loop`
Expected: compile error — `optimize_treesa_core` not defined.

- [ ] **Step 3: Implement**

Rename the current function to `fn optimize_treesa_core<L: Label>(...)` (same body, same signature, not `pub`). Add:

```rust
/// Optimize an EinCode using TreeSA.
///
/// By default this runs the full pipeline: structural simplification
/// ([`crate::preprocess::simplify`]), the annealing trial loop on the reduced
/// network, and splice-back — controlled by [`TreeSA::preprocess`]. A positive
/// [`TreeSA::surgery_budget`] additionally refines the result with
/// [`crate::waist_surgery::refine`].
pub fn optimize_treesa<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
) -> Option<NestedEinsum<L>> {
    let tree = if config.preprocess {
        let simplified = simplify(code, size_dict);
        let reduced = optimize_treesa_core(&simplified.code, size_dict, config)?;
        splice(&reduced, &simplified.subtrees)
    } else {
        optimize_treesa_core(code, size_dict, config)?
    };
    Some(tree)
}
```

Move the existing doc comment (plus the pipeline paragraph above) onto the wrapper; add `use crate::preprocess::{simplify, splice};` to the imports. Keep the existing `# Example` doc test on the wrapper.

Note: `simplify` on a 0- or 1-tensor code returns it unchanged with the leaf subtree, and `optimize_treesa_core` already handles those sizes, so no new edge-case code.

- [ ] **Step 4: Run the gate**

Run: `make check-all`
Expected: clean. If any pre-existing treesa test asserting an exact tree fails under the new default, set `preprocess: false` is NOT the fix — first check whether the test's network is simplifiable; only where it is, update the expectation deliberately with a comment citing this plan, one test per commit hunk. (The tolerance-based `test_reg3_220_treesa` at `omeco/src/lib.rs:1296` asserts `sc <= 32` and is expected to keep passing.)

- [ ] **Step 5: Commit**

```bash
git add omeco/src/treesa.rs
git commit -m "TreeSA: preprocessing front-end on by default"
```

---

### Task 4: Surgery stage in `optimize_treesa`

**Files:**
- Modify: `omeco/src/treesa.rs` (wrapper from Task 3)
- Test: inline tests in `omeco/src/treesa.rs`

**Interfaces:**
- Consumes: `crate::waist_surgery::refine(&NestedEinsum<L>, &EinCode<L>, &HashMap<L, usize>, Duration) -> (NestedEinsum<L>, WaistReport)`.
- Produces: final `optimize_treesa` behavior for Task 6's Python bindings.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn test_surgery_budget_never_worse() {
    // 4x4 periodic grid — a frozen-waist-style instance where surgery acts.
    let code = grid_code(4, 4); // helper below
    let sizes: HashMap<usize, usize> =
        code.unique_labels().into_iter().map(|l| (l, 2)).collect();
    let base_cfg = TreeSA::fast();
    let base = optimize_treesa(&code, &sizes, &base_cfg).unwrap();
    let base_tc = crate::contraction_complexity(&base, &sizes, &code.ixs).tc;
    let cfg = TreeSA::fast().with_surgery_budget(2.0);
    let refined = optimize_treesa(&code, &sizes, &cfg).unwrap();
    let refined_tc = crate::contraction_complexity(&refined, &sizes, &code.ixs).tc;
    assert!(refined_tc <= base_tc + 1e-9, "{refined_tc} > {base_tc}");
    assert_eq!(refined.leaf_count(), code.num_tensors());
}

#[test]
fn test_surgery_off_is_reproducible() {
    let code = EinCode::new(
        vec![vec![0usize, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
        vec![],
    );
    let sizes: HashMap<usize, usize> = (0..4).map(|l| (l, 8)).collect();
    let cfg = TreeSA::fast(); // surgery_budget == 0.0
    let a = optimize_treesa(&code, &sizes, &cfg).unwrap();
    let b = optimize_treesa(&code, &sizes, &cfg).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}
```

`grid_code(rows, cols)` helper (put next to the tests): build a periodic 2D grid where every horizontal/vertical neighbor pair shares a fresh label id — 16 tensors of rank 4; copy the shape of `waist_surgery`'s own `grid` test helper (`omeco/src/waist_surgery.rs`, `mod tests`, fn `grid`) rather than inventing a new topology.

- [ ] **Step 2: Run tests to verify they fail**

The never-worse assertion can pass vacuously before the wiring exists, so
make the red observable: add this temporary assertion as the test's last
line, remove it in Step 3 once wiring lands:

```rust
    // RED marker: fails until optimize_treesa consumes surgery_budget.
    let calls_grep = std::fs::read_to_string("src/treesa.rs").unwrap();
    assert!(calls_grep.contains("waist_surgery::refine"),
        "optimize_treesa does not call refine yet");
```

Run: `cargo test -p omeco --lib test_surgery_budget_never_worse`
Expected: FAIL on the RED marker assertion.

- [ ] **Step 3: Implement**

In the Task-3 wrapper, before `Some(tree)`:

```rust
    if config.surgery_budget > 0.0 {
        let budget = std::time::Duration::from_secs_f64(config.surgery_budget);
        let (refined, _report) = crate::waist_surgery::refine(&tree, code, size_dict, budget);
        return Some(refined);
    }
```

- [ ] **Step 4: Run the gate**

Run: `make check-all`
Expected: clean; the two new tests pass.

- [ ] **Step 5: Commit**

```bash
git add omeco/src/treesa.rs
git commit -m "TreeSA: opt-in waist-surgery post-pass via surgery_budget"
```

---

### Task 5: Python bindings

**Files:**
- Modify: `omeco-python/src/lib.rs:546-548` (`PyTreeSA::new` signature), `:576-581` (`fast`), getters block, `:799` (`optimize_treesa` default construction), module registration at the bottom (`add_class`/`add_function` list)
- Test: `omeco-python/tests/test_pipeline.py` (new)

**Interfaces:**
- Consumes: Rust `TreeSA` fields from Task 2; `preprocess::simplify_then_optimize`, `waist_surgery::refine`, `SimplifyReport { n_original, n_reduced, shrink }`, `WaistReport { n_original, surgery_calls, cheaper_cuts, rebuild_attempts, rebuild_accepts, waist_min_hits }`.
- Produces (Python API): `TreeSA(ntrials=10, niters=50, betas=None, score=None, preprocess=True, surgery_budget=0.0)` with matching getters; `simplify_then_optimize(ixs, out, sizes, optimizer) -> (NestedEinsum, SimplifyReport)`; `waist_refine(tree, ixs, out, sizes, budget_secs) -> (NestedEinsum, WaistReport)`.

- [ ] **Step 1: Write the failing tests**

`omeco-python/tests/test_pipeline.py`:

```python
from omeco import (
    TreeSA, GreedyMethod, optimize_code, contraction_complexity,
    simplify_then_optimize, waist_refine,
)

CHAIN_IXS = [[0, 1], [1, 2], [2, 3], [3, 4]]
CHAIN_OUT = [0, 4]
CHAIN_SIZES = {i: 2 for i in range(5)}


def test_treesa_pipeline_defaults():
    opt = TreeSA()
    assert opt.preprocess is True
    assert opt.surgery_budget == 0.0
    opt2 = TreeSA(preprocess=False, surgery_budget=1.5)
    assert opt2.preprocess is False
    assert opt2.surgery_budget == 1.5


def test_treesa_default_keeps_all_leaves():
    tree = optimize_code(CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, TreeSA(ntrials=1, niters=5))
    assert tree.leaf_count() == 4


def test_simplify_then_optimize_reports_shrink():
    tree, report = simplify_then_optimize(CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, GreedyMethod())
    assert tree.leaf_count() == 4
    assert report.n_original == 4
    assert report.n_reduced <= report.n_original


def test_waist_refine_never_worse():
    seed = optimize_code(CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, GreedyMethod())
    seed_tc = contraction_complexity(seed, CHAIN_IXS, CHAIN_SIZES).tc
    refined, report = waist_refine(seed, CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, 0.5)
    tc = contraction_complexity(refined, CHAIN_IXS, CHAIN_SIZES).tc
    assert tc <= seed_tc + 1e-9
    assert report.surgery_calls >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make python-dev && make python-test`
Expected: `test_pipeline.py` fails — `TreeSA() got an unexpected keyword` / import errors for the two functions.

- [ ] **Step 3: Implement**

1. `PyTreeSA::new`: signature `(ntrials=10, niters=50, betas=None, score=None, preprocess=true, surgery_budget=0.0)`; set the two fields on `inner` (the constructor already uses a struct literal with `..Default::default()` — set explicitly from the arguments). Add `#[getter] preprocess() -> bool` and `#[getter] surgery_budget() -> f64`. `fast()` keeps defaults via `TreeSA::fast()`.
2. Update the `optimize_treesa` pyfunction's fallback `PyTreeSA::new(10, 50, None, None)` call to the new arity `PyTreeSA::new(10, 50, None, None, true, 0.0)`.
3. New pyclasses, following the `PyContractionComplexity` pattern in the same file (plain `#[pyclass]` with `#[getter]`s and `__repr__`):

```rust
#[pyclass(name = "SimplifyReport")]
#[derive(Clone)]
pub struct PySimplifyReport { inner: omeco::preprocess::SimplifyReport }
// getters: n_original -> usize, n_reduced -> usize, shrink -> f64
// __repr__: format!("SimplifyReport(n_original={}, n_reduced={}, shrink={:.3})", ...)

#[pyclass(name = "WaistReport")]
#[derive(Clone)]
pub struct PyWaistReport { inner: omeco::waist_surgery::WaistReport }
// getters for all six u64/usize counter fields; __repr__ listing them
```

4. New pyfunctions (mirror `optimize_treesa`'s i64 conversion style):

```rust
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn simplify_then_optimize(
    ixs: Vec<Vec<i64>>, out: Vec<i64>, sizes: HashMap<i64, usize>,
    optimizer: Option<PyGreedyMethod>,
) -> PyResult<(PyNestedEinsum, PySimplifyReport)> {
    let code = EinCode::new(ixs, out);
    let opt = optimizer.map(|o| o.inner).unwrap_or_default();
    omeco::preprocess::simplify_then_optimize(&code, &sizes, &opt)
        .map(|(t, r)| (PyNestedEinsum { inner: t }, PySimplifyReport { inner: r }))
        .ok_or_else(|| PyValueError::new_err("optimization failed"))
}

#[pyfunction]
fn waist_refine(
    tree: PyNestedEinsum, ixs: Vec<Vec<i64>>, out: Vec<i64>,
    sizes: HashMap<i64, usize>, budget_secs: f64,
) -> PyResult<(PyNestedEinsum, PyWaistReport)> {
    if !budget_secs.is_finite() || budget_secs < 0.0 {
        return Err(PyValueError::new_err("budget_secs must be a non-negative finite number"));
    }
    let code = EinCode::new(ixs, out);
    let (t, r) = omeco::waist_surgery::refine(
        &tree.inner, &code, &sizes, std::time::Duration::from_secs_f64(budget_secs),
    );
    Ok((PyNestedEinsum { inner: t }, PyWaistReport { inner: r }))
}
```

(`simplify_then_optimize` takes a `GreedyMethod` optimizer argument only — the generic `O: CodeOptimizer` cannot cross PyO3; Greedy is the documented inner optimizer for the reduced network, and TreeSA users get preprocessing via the flag instead. State this in the docstring.)

5. Register: `m.add_class::<PySimplifyReport>()?;`, `m.add_class::<PyWaistReport>()?;`, `m.add_function(wrap_pyfunction!(simplify_then_optimize, m)?)?;`, `m.add_function(wrap_pyfunction!(waist_refine, m)?)?;`. Add both names to `omeco-python`'s `__init__.py` re-export list if one exists (check `omeco-python/python/omeco/__init__.py` or equivalent; mirror how `optimize_treesa` is exported).

- [ ] **Step 4: Run the gates**

Run: `make python-dev && make python-test` then `make check-all`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add omeco-python
git commit -m "Python: TreeSA pipeline flags, simplify_then_optimize, waist_refine"
```

---

### Task 6: Retire the alignment rule (CLAUDE.md, README, test marker)

**Files:**
- Modify: `.claude/CLAUDE.md` (sections listed below)
- Modify: `README.md` (positioning line)
- Modify: `omeco/src/lib.rs:1296-1310` (marker comment only)

**Interfaces:** none (docs/policy).

Note: no GitHub workflow actually gates on Julia (verified — `grep -l julia
.github/workflows/*` matches nothing; only the Makefile has a Julia
benchmark target), so the spec's "demote the CI Julia gate" reduces to
deleting the CLAUDE.md sentence claiming that gate exists.

- [ ] **Step 1: Rewrite CLAUDE.md**

Replace the `## CRITICAL: Alignment with Julia OMEinsumContractionOrders` section (lines ~8-20) with:

```markdown
## Relationship to Julia OMEinsumContractionOrders

omeco originated as a port of
[OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl)
and keeps **JSON interop** (`writejson`/`readjson`) as a compatibility
contract. Behavior and defaults are independent: changes are judged on
omeco's own results and ergonomics, not parity with Julia. The Julia
implementation at `~/.julia/dev/OMEinsumContractionOrders/` remains a
useful reference when porting algorithms.
```

In the Testing section: delete the line "**Always compare results with Julia implementation**" and the "CRITICAL: Do NOT modify existing tests… ALIGNED WITH JULIA" sentence; replace with "Do not weaken existing regression tests to make a change pass; update expectations only with a comment explaining the behavior change." In CI Requirements: delete "**Benchmark tc values must match Julia within tolerance**". In Benchmarks/Expected Results and Debugging sections: keep the Julia comparison instructions but reword "should match Julia within" → "can be compared against Julia with" (informational).

- [ ] **Step 2: Update the test marker and README**

In `omeco/src/lib.rs` `test_reg3_220_treesa`, change the banner comment `ALIGNED WITH JULIA - DO NOT MODIFY WITHOUT CHECKING JULIA TESTS` to `REGRESSION (originally cross-checked against Julia test/treesa.jl)` — keep the assertion and provenance notes. In `README.md`, change "ported from" to "originated as a port of" in the opening description.

- [ ] **Step 3: Run the gate**

Run: `make check-all`
Expected: clean (comment/doc changes only).

- [ ] **Step 4: Commit**

```bash
git add .claude/CLAUDE.md README.md omeco/src/lib.rs
git commit -m "Retire the Julia behavioral-alignment rule (JSON interop stays)"
```

---

### Task 7: Documentation (README quickstart, mdBook page, rustdoc check)

**Files:**
- Modify: `README.md` (Python and Rust quickstart blocks, lines ~54-133)
- Create: `docs/src/algorithms/default-pipeline.md`
- Modify: `docs/src/SUMMARY.md` (add the page above `paper-algorithms.md`'s entry)

**Interfaces:** consumes the Task 2-5 API exactly as named there.

- [ ] **Step 1: README quickstart**

Lead the Python block with the default path and the budget knob:

```python
from omeco import optimize_code, contraction_complexity, TreeSA

ixs = [[0, 1], [1, 2], [2, 3]]
out = [0, 3]
sizes = {0: 100, 1: 200, 2: 50, 3: 100}

tree = optimize_code(ixs, out, sizes, TreeSA())        # full default pipeline
better = optimize_code(ixs, out, sizes, TreeSA(surgery_budget=30.0))  # spend time, get a better tree
print(contraction_complexity(tree, ixs, sizes))
```

Keep the existing Greedy/slicing content below it. Mirror the two-line story in the Rust block (`TreeSA::default()` / `.with_surgery_budget(30.0)`).

- [ ] **Step 2: mdBook page**

`docs/src/algorithms/default-pipeline.md` — sections: (1) what `TreeSA::default()` runs, with the stage list simplify → anneal trials → splice → optional surgery; (2) determinism: default output is seeded-deterministic; `surgery_budget > 0` trades reproducibility for quality (never worse, machine-dependent); (3) preprocessing guarantees — exactness yes, space-safety yes (each merged intermediate is no larger than its larger input, so sc is never pushed above the original network's floor), tc-optimality no (cite the module docs); (4) when surgery helps — frozen-waist instances (grids, surface-code-like circuits), near-no-op elsewhere; (5) escape hatches table: `preprocess: false`, `surgery_budget: 0.0`. Register in `SUMMARY.md`.

- [ ] **Step 3: Build the book and docs**

Run: `make build-book` and `cargo doc -p omeco --no-deps 2>&1 | grep -i warn; true`
Expected: book builds; no new rustdoc warnings.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/src
git commit -m "Docs: default-pipeline page and quickstart for the new flags"
```

---

### Task 8: Changelog and patch version bump

**Files:**
- Create: `CHANGELOG.md` (if absent; otherwise prepend)
- Version bump via `make bump-patch` (bumps 0.2.6 → 0.2.7 across Cargo/pyproject and commits)

**Interfaces:** none.

- [ ] **Step 1: Changelog entry**

```markdown
# Changelog

## 0.2.7 — 2026-07-29

**Default behavior change:** `TreeSA` now runs the structural
simplification front-end by default (`preprocess: true`) — output trees
may differ from 0.2.6 (same contraction result, typically equal or better
cost). Restore the old behavior with `preprocess: false`.

- New `TreeSA.surgery_budget` (seconds, default 0.0/off): opt-in
  waist-surgery post-pass; never worse than the unrefined tree, wall-clock
  dependent.
- Python: `TreeSA(preprocess=..., surgery_budget=...)`,
  `simplify_then_optimize`, `waist_refine`, `SimplifyReport`, `WaistReport`.
- `Label` now requires `Ord` (all built-in label types already qualify).
- The Julia behavioral-alignment rule is retired; JSON interop is unchanged.
```

Commit: `git add CHANGELOG.md && git commit -m "Changelog for 0.2.7"`

- [ ] **Step 2: Bump**

Run: `make bump-patch`
Expected: version 0.2.7 committed by the make target.

- [ ] **Step 3: Final gate**

Run: `make check-all && make python-dev && make python-test && make build-book`
Expected: everything green.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin jg/usability
gh pr create -R GiggleLiu/omeco --base master --title "TreeSA pipeline flags: preprocess default, opt-in surgery; retire Julia alignment" \
  --body "Implements docs/superpowers/specs/2026-07-29-usability-post-alignment-design.md. Default change: preprocess=true (changelog). surgery_budget opt-in. Python parity + simplify_then_optimize/waist_refine. Alignment rule retired, JSON interop kept. Patch bump 0.2.7."
```

Do not merge; the maintainer approves.
