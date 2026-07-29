# omeco post-alignment usability design

Date: 2026-07-29. Approved by the maintainer in-session.

## Goal

Make the paper's algorithms (structural preprocessing, waist surgery)
reachable from the default user path in both Rust and Python, and retire the
Julia behavioral-alignment constraint that previously prevented this.

## Decisions (user-confirmed)

1. **Rule scope — full freedom.** omeco is independent: behavior and defaults
   may diverge from OMEinsumContractionOrders.jl when it improves results or
   ergonomics. JSON `writejson`/`readjson` interop with Julia remains a
   compatibility contract (file format, not behavior).
2. **API shape — flags on TreeSA.** No new pipeline type; `TreeSA` gains the
   pipeline stages as config fields.
3. **Defaults.** `preprocess: true` (deterministic, exactness-preserving, big
   wins on circuit-like networks, near-no-op elsewhere — output trees differ
   from 0.2.x). `surgery_budget: 0.0` (off — consistent with previous
   behavior; surgery is wall-clock-dependent so it stays opt-in).

## 1. API

### Rust

```rust
pub struct TreeSA {
    // existing fields unchanged: betas, ntrials, niters, score,
    // decomposition_type, initializer
    /// Run the structural simplification front-end (simplify -> optimize the
    /// reduced network -> splice). Deterministic and exactness-preserving.
    pub preprocess: bool,        // default: true
    /// Wall-clock budget in seconds for the waist-surgery post-pass on the
    /// selected tree. 0.0 disables it. Results with surgery enabled are
    /// never worse than without, but are not reproducible across machines.
    pub surgery_budget: f64,     // default: 0.0
}
```

`optimize_treesa` becomes, in order:

1. If `preprocess`: `simplify(code)`, run the existing trial loop on the
   reduced code, `splice` back. Otherwise run the trial loop directly.
2. If `surgery_budget > 0.0`: `waist_surgery::refine(tree, code, sizes,
   Duration::from_secs_f64(surgery_budget))`, return the refined tree.

Quality is monotone across stages: splice preserves exactness; `refine`
independently rescores and accepts only strict global improvements.

Presets: `TreeSA::default()` = `{preprocess: true, surgery_budget: 0.0}`;
`TreeSA::fast()` keeps the same two values. Builder-style setters
(`with_preprocess`, `with_surgery_budget`) for non-Python callers.

Unchanged: `GreedyMethod`, `Treewidth`, `TreeSASlicer`, `ExhaustiveSearch`,
JSON I/O, the standalone `preprocess::*` and `waist_surgery::*` modules,
warm-start API.

### Python

- `TreeSA(..., preprocess=True, surgery_budget=0.0)` — same semantics;
  `optimize_treesa` picks up the defaults.
- New functions for manual composition:
  - `simplify_then_optimize(ixs, out, sizes, optimizer) -> (NestedEinsum,
    SimplifyReport)`
  - `waist_refine(tree, ixs, out, sizes, budget_secs) -> (NestedEinsum,
    WaistReport)`
- New pyclasses `SimplifyReport` (n_original, n_reduced, shrink) and
  `WaistReport` (existing counter fields), read-only.

## 2. Rule removal and migration

- **CLAUDE.md**: replace the "MUST stay aligned with Julia" section with:
  omeco is independent; the JSON format remains Julia-interoperable; check
  the Julia implementation for reference when porting algorithms, not for
  behavioral parity. Drop the "do not modify ALIGNED WITH JULIA tests" rule
  and the "benchmark tc must match Julia" CI requirement text.
- **Tests**: keep every ALIGNED-WITH-JULIA test as an ordinary regression
  test. Where `preprocess: true` changes an exact expected tree, update the
  expectation deliberately (one commit per test file, each reviewed against
  the new pipeline's contraction_complexity — never blanket-regenerated).
  Tests asserting tolerance-band tc against recorded Julia values stay
  as-is if they still pass (preprocess never worsens quality; investigate
  any that fail rather than relaxing them).
- **CI**: the Julia-comparison benchmark becomes a non-blocking
  informational job; it no longer gates merges.
- **README**: "ported from" -> "originated as a port of"; positioning text
  updated.

## 3. Documentation

- README quickstart: default path shown first
  (`optimize_code(ixs, out, sizes, TreeSA())` in Python, three lines), with
  the surgery flag shown as the "spend more time, get a better tree" knob.
- mdBook: new page "How omeco optimizes by default" — pipeline diagram
  (simplify -> anneal -> splice -> optional surgery), when surgery helps
  (frozen-waist instances; cites the paper), determinism note and the
  `surgery_budget=0` escape hatch, preprocessing guarantees (exactness, sc
  never harmed, no tc-optimality claim).
- rustdoc: examples on both new fields; `# Determinism` note on
  `surgery_budget`.

## 4. Testing

- Default pipeline quality: on each benchmark graph, `TreeSA::default()`
  tc <= plain trial-loop tc + 0.5 bits (empirical bound — preprocessing
  carries no tc-optimality theorem; on graphs where simplify is a no-op,
  such as reg3, assert exact equality instead).
- Reproducibility: `surgery_budget = 0.0` + fixed config produces identical
  trees across two runs.
- Interface invariants with `preprocess: true`: leaf count preserved,
  recursive parent/child interfaces consistent (reuse existing helpers).
- Surgery flag: with a small budget on a frozen-waist-style instance,
  result tc <= no-surgery tc; report counters populated.
- Python: round-trip tests for the new bindings and report types; defaults
  visible from Python match Rust.
- Coverage stays above the project's 95% bar.

## 5. Versioning and rollout

- Patch bump via `make bump-patch` (maintainer decision 2026-07-29; the
  default-output change is called out prominently in the changelog instead
  of a minor bump).
- Changelog: states the `preprocess: true` default, the unchanged-behavior
  escape (`TreeSA { preprocess: false, surgery_budget: 0.0, ..cfg }`), and
  the alignment-rule retirement.
- Single PR off `master`; the paper repo is unaffected (its pinned data is
  frozen; nothing in the figure pipeline calls TreeSA).

## Out of scope

- Interleaved anneal/surgery loop as a library optimizer (the warm-start
  API already enables it for power users; revisit on demand).
- Composable stage combinators.
- Changing `GreedyMethod`/`Treewidth` defaults.
- Dropping Julia JSON interop.

## Amendment (2026-07-29): maintainer direction replaced `surgery_budget`
(seconds) with `surgery_iters: u64` (deterministic iteration cap, default 0 =
off); wall-clock budgets remain only on the low-level waist_surgery APIs.
