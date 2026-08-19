# Surgery v2 (warm-restricted / local rebuild) + cold-only control + work-matched ablation

Date: 2026-08-19. Branch: `jg/surgery-v2` (worktree `~/rcode/.worktrees/omeco-surgery-v2`, base `origin/master` = 446c7ca).
Owner of the implementation: Codex (non-interactive). Reviewer: Jinguo / Claude.

## Why

An independent 2x2 ablation (report `benchmark_2x2_ablation_tables.pdf`, OMECO commit 446c7ca,
38 instances x {raw, preprocessed} x 5 seeds) found:

1. `TreeSA + anneal_surgery_rounds(R=8)` beats TreeSA by 1-10 bits, **but a matched cold-only control**
   (same 8 pinned cold span-gated fine-tuning passes, surgery call removed) **ties or beats it**
   (surface code 0/1/3, UAI 2/8/0, TensorCircuit 7/11/6 W/T/L). The gain in PR #34's 13/13 claim was the
   fine-tuning engine (attempt-052's span-gated cold ladder), not the surgery operator.
2. The `surgery_probability` arm (`WaistUpdate::propose`, one leaf-SPR move) is ~50/50 vs TreeSA.

Reading the current code against the original autoresearch attempt-054 (which set the records) shows two
structural weaknesses in the shipped surgery operator that plausibly explain "cheaper top cut trades against
more expensive internal nodes" (attempt-054 LOG verdict 3):

- `Refiner::solve_side` (omeco/src/waist_surgery.rs ~L1139) rebuilds **each side from greedy** + 3 cold
  V-cycles, discarding the incumbent's optimized internal structure.
- `Refiner::rebuild` (~L1114) always promotes the improved cut to the **root over all n tensors**, even when
  the argmax (waist) node is deep and |A| << n.

This task (i) exposes the cold-only control as a public, matched arm; (ii) adds opt-in surgery variants that
fix the two weaknesses; (iii) ships a deterministic ablation driver that also answers the work-matched
question "is TreeSA + k cold passes better than TreeSA with k x the work?" (the real candidate headline for
the paper).

## Hard constraints

- **Default behaviour and artifacts must not change.** `TreeSA::default()`, `anneal_surgery_rounds`,
  `refine`, `refine_capped` must produce byte-identical trees/reports to 446c7ca for every existing test.
  All new behaviour is opt-in via new option structs. Existing tests must not be edited (project rule).
- `make check-all` green (fmt, clippy -D warnings, tests, doctests). Coverage of changed lines ~100%
  (project requires >95% overall).
- No panics/unwraps in library code; `thiserror` for errors; rustdoc with examples on all public items.
- Single-threaded, deterministic experiment code (fixed seeds, no wall-clock-dependent control flow).
- Commit after each phase with a descriptive message. Do **not** merge or push to master; at the end open a
  **draft PR** against `master` with the evidence (or, if `gh` auth is unavailable, leave the branch and write
  the PR body to `docs/design/2026-08-19-surgery-v2-PR.md`).
- Do not touch `benchmarks/graphs`, `benchmarks/results`, Julia-alignment tests, or Python bindings except to
  add pass-through plumbing if trivially needed (not required).

## Phase 1 - Public matched control (cold-only)

Add to `omeco/src/treesa.rs`:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct RoundsOptions {
    /// Run the global waist-surgery call at the start of each round (default true = current behaviour).
    pub surgery: bool,
    /// How rebuilt sides are initialised (default Greedy = current behaviour).
    pub rebuild: waist_surgery::RebuildMode,
    /// Where surgery operates (default Root = current behaviour).
    pub scope: waist_surgery::SurgeryScope,
}
impl Default for RoundsOptions { .. } // == current behaviour
pub fn anneal_refine_rounds<L: Label>(seed, code, size_dict, config, rounds, opts: &RoundsOptions)
    -> (NestedEinsum<L>, RoundsReport);
```

`anneal_surgery_rounds` becomes a thin wrapper calling `anneal_refine_rounds(.., &RoundsOptions::default())`.
With `surgery: false` the round performs **exactly** the same fine-tuning (same `fine_tune_beta_schedule`,
`fine_niters`, `fine_trials`, same per-round seeds `0xA55E + r*fine_trials + trial`, same ratchet and
`RoundTrace` fields; `waist: None`, `surgery_accepted: false`, `surgery_calls_total` unchanged at 0), so the two
arms are matched the way the external report's control was built. Add a unit test asserting that with
`surgery: false` the result equals running the fine-tuning path alone, and a test asserting
`RoundsOptions::default()` reproduces `anneal_surgery_rounds` bit-for-bit on a small instance.

Plumb `RebuildMode`/`SurgeryScope` through `refine_capped_seeded_with_trace` (add an options parameter to an
internal `_opts` variant; keep the existing signatures as wrappers).

## Phase 2 - Surgery variants (opt-in)

In `omeco/src/waist_surgery.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RebuildMode { #[default] Greedy, WarmRestricted }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurgeryScope { #[default] Root, Local }
```

### 2a `RebuildMode::WarmRestricted`

`solve_side(tensors, open)` currently seeds with `optimize_code(&sub_code, GreedyMethod::default())`.
WarmRestricted seeds instead with the **incumbent tree restricted to `tensors`**: delete every leaf not in the
side, contract unary internal nodes, and recompute every interface by outside-occurrence counting against the
side's open labels (reuse `expr_to_nested_counted` / `nested_to_expr_tree` machinery). Leaves that FM moved
into the side from the other side are therefore kept at the position their old ancestors imply; no separate
grafting step. The cold V-cycle anneal afterwards is unchanged (same constants, same RNG usage order as far as
possible; document any unavoidable RNG-stream difference - it only matters for opt-in runs). Fall back to
Greedy if restriction fails (e.g. singleton sides keep the existing unary-reduction path).

Requirements: the restricted tree must be a valid binary tree over exactly `tensors` whose root output equals
`open`; add tests on a grid/ring network that (i) restriction preserves leaf sets and interfaces, (ii) the
rebuilt tree's global complexity is finite and the acceptance gate still only accepts strict tc improvement,
(iii) on a case constructed so that the incumbent halves are well-optimised, WarmRestricted yields a rebuilt
tree whose tc is <= the Greedy rebuild's tc (construct deterministically; if no such case is easy, assert the
weaker property that the restricted seed's own tc is <= the greedy seed's tc on that instance).

### 2b `SurgeryScope::Local`

Instead of promoting the improved cut to the root:

1. Let `w` be the argmax node, `A` its leaf set. Walk up from `w` to the lowest ancestor `S` whose leaf count
   is >= `min(n, 2*|A|)` (if `w` is already >= n/2 use the root: identical to `Root`).
2. Build the FM hypergraph on the sub-network of `S`'s leaves whose outputs are `S`'s output labels (labels of
   `S`'s leaves that also occur outside `S` or in `iy`). Output labels are constant in the cut cost exactly as
   today; the balance target is `|A|` within `S`.
3. Rebuild only the subtree `S` from the improved bipartition (two sides, then a node with `S`'s output
   labels) and splice it back into the incumbent at `S`'s position; the rest of the tree is untouched.
4. Accept iff the **global** tc (independently rescored) strictly drops - same gate as today.

`WaistReport` gains nothing new that breaks `Copy + Eq`; if you need per-call scope diagnostics put them in
`WaistCallTrace` (add `scope_leaves: usize`).

Tests: local surgery on a network where the argmax is deep leaves the subtree outside `S` byte-identical;
a run with `Local` never returns a worse tc than the seed; leaf permutation preserved.

## Phase 3 - Ablation driver (`benchmarks/surgery_ablation/`)

Add example binary `omeco/examples/surgery_ablation.rs` (register as `[[example]] test = true` like
`paper_bench` was) plus `benchmarks/surgery_ablation/{README.md, summarize.py, results/}`.

Instances: JSON in the `writejson` format, passed as `--instances <dir>` (the paper instances live in
`~/rcode/contraction-order-frontiers/benchmarks/omeco/instances/*.json`: surfacecode_d9/d13/d17/d21, ksg,
dbn_13, qft_27, nqueens_28, rqc_97_m24, sycamore_53_20_0, sycamore_m20, reg3_250, reg3_1000, petersen).
`--only name,name` filter; `--out <file.jsonl>` append-only, resumable (skip jobs whose key already exists).

Common protocol (mirror the external report so numbers are comparable):
- Preprocess with `simplify` (the current `optimize_code` pipeline default), anneal on the simplified network,
  splice back, score on the original network; also support `--raw`.
- Baseline TreeSA: betas `0.01:0.05:14.96` (300 levels), `ntrials = 1`, pure-tc score
  (`sc_weight = 0`, `rw_weight = 0`, `sc_target` irrelevant), `niters = max(1, round(1.4e8 / (300 * (n_simplified - 1))))`
  i.e. ~1.4e8 node visits. Record the actual planned visits.
- 5 matched labels r0..r4: tensor relabeling seed `5400 + 2i` (deterministic permutation of tensor order and
  label ids, as in the old `paper_bench` seeded relabeling - see `git show f0fd19a:omeco/examples/paper_bench.rs`), optimizer seed `7000 + 2i`.
- Every job records: instance, label, arm, params, tc, sc, rwc, wall seconds, sweep/visit counts, and for rounds
  arms the per-round `RoundTrace` (tc_before, tc_after_surgery, tc_after_anneal, tc_retained, surgery_accepted).

Arm set B (surgery variants), R in {8, 32}:
- `cold_only` (surgery=false)
- `surg_greedy_root` (= current master)
- `surg_warm_root`
- `surg_greedy_local`
- `surg_warm_local`

Arm set A (work-matched), on the same seeds:
- `treesa_x{1,2,4,8}`: the baseline with `niters` scaled by k (k x planned visits).
- `treesa_x1+cold{8,32}`: baseline then `cold_only` rounds.
- Report tc against (a) wall seconds and (b) a machine-independent work unit = total node visits
  (anneal visits + fine-tune sweeps x (n-1)); expose the fine-tune sweep count from `RoundsReport`
  (add a `fine_tune_sweeps_total: u64` field - additive, does not break existing code).

`summarize.py` reads the JSONL and prints/writes markdown tables: per instance min-of-5 and median-of-5 per
arm, W/T/L of each surgery arm vs `cold_only` (tie tolerance 1e-9), and a work-matched table
(tc vs visits, tc vs wall) for set A; plus a per-arm count of accepted rebuilds.

Deliverable for this phase: the driver + summarizer, a **smoke** result committed under
`benchmarks/surgery_ablation/results/smoke.jsonl` for `surfacecode_d9, dbn_13, qft_27, petersen` with
R=8 only and 2 labels (keep it under ~15 min total on an M3), and the rendered `smoke.md`. The full campaign
is run separately by the reviewer on a quiet machine (timing matters for set A); make the driver print an
ETA and support `--jobs N` process-level parallelism only as an opt-in (default serial).

## Phase 4 - Wrap-up

- CHANGELOG entry (unreleased): new `RoundsOptions`/`anneal_refine_rounds`, `RebuildMode`, `SurgeryScope`,
  `fine_tune_sweeps_total`; defaults unchanged.
- mdBook page or rustdoc module doc paragraph describing the two variants and the cold-only control.
- Draft PR body: what changed, why, the smoke table, and the exact commands to run the full campaign.

## Notes for the implementer

- Read first: `omeco/src/waist_surgery.rs` (Refiner, rebuild, solve_side, extract_waist, Hyper, fm_refine),
  `omeco/src/treesa.rs` (`anneal_surgery_rounds`, `fine_tune_tree_sa`, `prepare_warm_anneal`,
  `warm_exprtree_to_nested`), `omeco/src/preprocess.rs` (simplify/splice), and the project `.claude/CLAUDE.md`.
- The original autoresearch implementation (for reference only, do not copy wholesale):
  `/Users/liujinguo/rcode/omeco/.worktrees/attempt-054/omeco/examples/attempt.rs` and its `LOG.md`.
- Keep the rebuild trigger semantics (issue #23) unchanged; out of scope.
- Deep trees: the crate documents a recursion-depth limitation; the rounds path already runs on a large-stack
  thread inside `optimize_code` - make sure the example uses the same large-stack pattern for the driver.
