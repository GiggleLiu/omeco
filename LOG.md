# Attempt 029 — structure-seeded refinement at non-converged scale

- **date:** 2026-07-23
- **kind:** draft
- **parent:** none
- **targets (scored):** `reg3_1000` (1000 tensors, 3-regular), `rqc_97_m24`
  (1238 tensors: 97-qubit 10x10-minus-corners lattice, 24 ABCD cycles, 194
  rank-1 boundary vectors, all dims 2). Budget 90 s each, single-threaded.
- **records to beat:** reg3_1000 tc=135.75, rqc_97_m24 tc=106.47 (fresh,
  reference-seeded, HIGH VARIANCE). Confirmed record needs >0.05 improvement.

## Hypothesis

At non-converged scale (n ≈ 1000–1238, 90 s single-thread), **seed quality
dominates**: 90 s of simulated annealing from a greedy seed cannot cross the
gap that a structurally good initial tree closes for free. Two structured
seeds are constructible relabel-proof:

- **RQC lattice** → an MPS/spacetime-sweep tree. Recover tensor adjacency
  (rank-1 = boundary vectors, rank-4 = gates; every index joins exactly two
  tensors), BFS from a temporal end to get flat spatial slices, and build a
  contraction-front (caterpillar) tree whose sc is the swept spatial cut.
  Sibling 019 found TreeSA beats this at n=561 **converged**; the bet is that
  at n=1238 **non-converged** the seed's head start dominates.
- **reg3_1000** → recursive balanced min-cut bisection (FM refinement, sibling
  005's machinery), depth-limited with greedy leaves. Poor at small converged
  scale; the bet is that at 1000 nodes a global-cut seed beats 90 s of local
  search from greedy.

Then spend the remaining budget on **continuous SA refinement of the seed**
(incremental rule-diff deltas, no rebuild-from-greedy restarts), keeping the
global best by true tc. This is the same refinement engine 028 applies to a
greedy seed, so 029-vs-028 attributes any gain to **seeding vs annealing** at
scale.

## Expected evidence

- Confirmed records: rqc < 106.42, reg3_1000 < 135.70.
- Report per target: **raw seed tc** (structured seed before any SA) vs
  **refined final tc** vs record, plus the greedy-seed tc for reference.
- The informative comparison is vs 028 (same refinement core, greedy seed):
  if 029 wins, the gain is attributable to the structured seed, not the anneal.

## Design decisions

- **Seed builders reused from siblings:** 019's tensor-adjacency / BFS-order /
  contraction-front-tree machinery for the RQC lattice (multiple flat-temporal
  and cone sweep orders, forward + reverse, best-by-tc kept); 005's recursive
  FM-bisection + Staudt gluing for the 3-regular graph. A plain greedy tree is
  always built first and written immediately as an anytime safety net.
- **Refinement engine (the new core):** a self-contained simulated-annealing
  loop in the example, built directly on omeco's public `expr_tree` primitives
  (`ExprTree`, `Rule::applicable_rules`, `ScratchSpace::rule_diff`,
  `apply_rule_mut`, `tree_complexity`). It seeds the `ExprTree` from *our* best
  seed (replicating the library's `nested_to_expr_tree` conversion), so unlike
  stock `optimize_treesa` it refines the structured seed instead of a fresh
  greedy init. No library source is modified.
- **Objective:** pure tc (sc unbounded per validator v2.1). The SA energy uses
  `sc_target = +inf`, so every move's energy is `dtc` — a pure-tc anneal.
- **Anytime:** best-by-true-tc is kept; each improvement is atomically written
  (temp + rename). The greedy safety net guarantees a valid tree exists from
  t≈0.
- **Reheat cycles:** after one beta ramp the working tree is reset to the
  global best and the ramp repeats (basin hopping from best, not a seed
  rebuild), to keep the full 90 s productive.

## Implementation

Single self-contained example (`omeco/examples/attempt.rs`), no library source
modified. Pieces:

- **Seeds:** greedy (always, written at t≈0 as the anytime floor). Structured
  seed by instance shape:
  - *Layered circuit* (≥4 rank-1 boundary + ≥4 rank-4 gate tensors): recover
    tensor adjacency (each index joins exactly two tensors), BFS several
    spacetime-sweep orders (flat-temporal from each end, multi-source, cone
    sweeps, + reverses), and over each order build BOTH a caterpillar
    contraction-front tree and a balanced tree; keep the lowest-tc structured
    tree. (sibling-019 machinery.)
  - *Otherwise* (expander / generic large graph): recursive FM min-cut
    bisection with greedy leaves, 3 restarts. (sibling-005 machinery.)
- **Refinement engine (new):** a self-contained SA on omeco's public
  `expr_tree` primitives (`Rule::applicable_rules`, `ScratchSpace::rule_diff`,
  `apply_rule_mut`, `tree_complexity`), replicating the library's
  `optimize_subtree_mut` sweep (post-order, mutation-first, Metropolis on
  `dtc` with the sc term active when local sc > `sc_target`). The seed's
  `NestedEinsum` is converted to an `ExprTree` (replicating the library's
  `nested_to_expr_tree`) so the SA refines *our* seed, not a fresh greedy init.
  `sc_target = +inf` ⇒ pure-tc objective (matches the reference and the
  pure-tc leaderboard).
- **Two drivers, auto-selected:**
  - *Full doubling* (default for non-structured / expander): each round is a
    fresh full 300-β schedule from the seed, `niters` doubling 5→400 while a
    round fits the budget, keep global best. Seeded from greedy this is
    numerically the `treesa_tuned` reference.
  - *Warm continuous* (default for structured/RQC seeds): a single continuous
    anneal from the seed over only the cold tail β ≥ β0 (β0 = 2.0), `niters`
    doubling, so the hot phase never scrambles the seed. This is the
    refinement that actually exploits a good seed.
- **Anytime:** best-by-true-tc kept; every improvement atomically written.
- **Attribution toggles (env):** `A029_NOSEED=1` (seed from greedy),
  `A029_WARM=β0` (force/override warm), `A029_SCT=x` (finite sc_target).

### Key decisions / findings during development

1. Per-sweep `tree_complexity` dominated runtime (24–160 sweeps in 10 s); moving
   the tc check out of the inner loop raised throughput to ~50 k sweeps/30 s.
2. The reference `treesa_tuned` is **pure-tc** (`sc_target=+inf`) with a
   **doubling-niters** loop — its strong round is one deep anneal (niters=400).
   Matching that driver was necessary just to reach the reference (greedy+my
   driver → reg3 141.8 ≈ reference 140.5).
3. **A full-schedule anneal starts hot (β=0.01) and forgets its init**, so
   "from structured seed" ≈ "from greedy" — the seed is washed out. Exploiting a
   seed requires **warm** refinement (skip the hot phase). This was the pivotal
   design change.

## Results

### Precheck: PASS (`precheck_chain_10` structure ok; /tmp/pc_029.json,
status=scored, errors=[]). Re-run after every change; always green.

### The decisive constraint: output tree HEIGHT (serialization limit)

The validator's `scorer.py` parses the candidate tree with `json.load` at
Python's **default recursion limit (~1000)** and recomputes cost with recursive
walks. writejson nests ~2 JSON levels per tree level, so JSON depth ≈ 2·height.
Empirically a height-475 tree parses (the reference's own RQC output) and a
height-544 tree raises `RecursionError` → the instance is **rejected** (−5).
Good RQC contraction orders are inherently *deep* (near-sequential MPS-like
sweeps): the lowest-tc trees sit at height ~475–545, right at / over the limit.
attempt now **caps output height at 475** (proven-safe; the reference relies on
the same bound) and never writes a taller tree.

### Attribution experiments (single runs; validator notes HIGH VARIANCE here)

- **RQC** (record 106.47):
  - structured seed (blocked MPS front) + **warm** refinement reaches tc ≈
    **106.4 @90 s very quickly** — BUT the refined tree is height ~544, i.e.
    **not serializable** (would be rejected). Gating warm on height ≤ 475 throws
    away exactly those low-tc trees and falls back to the shallow seed (~171).
    So the warm result is *illusory*: it cannot be emitted.
  - greedy + full library anneal (default) reaches **106.3–114.6** with
    serializable trees (height 354–475). This IS the reference behaviour and is
    already at record level.
  → within the serialization limit the structured seed provides **no usable
    advantage** over greedy+anneal on the RQC. **Hypothesis NOT supported.**
- **reg3_1000** (record 135.75, expander):
  - greedy + full anneal (default): **139.3 @90 s** (reference: 135.8; same
    algorithm, within the stated high variance).
  - bisection seed + warm: 142.9–144.9 @30 s — strictly **worse**.
  → the global anneal wins; the seed does not help. **Hypothesis REFUTED.**

Mechanistic reason the seed does not help at this budget: a full-schedule
anneal starts hot (β=0.01) and **forgets its init**, so structured-seed ≈
greedy-seed; and the only refinement that *preserves* a seed (warm, cold-tail
only) drives the RQC toward deep, unserializable trees while under-exploring the
expander. At 90 s the plain greedy+anneal already effectively converges to
record level on both targets, leaving no gap for a seed to close.

### Anytime table — attempt DEFAULT vs baseline `treesa_tuned` (single runs,
validator-recomputed tc; high variance — treat ±3 as noise)

| target      | 10 s (att / base) | 30 s (att / base) | 90 s (att / base) | record |
|-------------|-------------------|-------------------|-------------------|--------|
| rqc_97_m24  | 116.3 / 118.4     | 111.8 / 106.9     | 114.6 / 106.3     | 106.47 |
| reg3_1000   | 144.4 / 140.5     | 145.2 / 136.6     | 139.3 / 135.8     | 135.75 |

All attempt outputs pass the scorer at the default recursion limit (heights:
rqc 354, reg3 304 @90 s). The default path is algorithmically the reference
(library TreeSA, ntrials=1, doubling niters 5→400, sc_target=+inf), so any
per-run gap is stochastic, not systematic.

### Verdict

Negative result: **structural seeding does not beat greedy+anneal at the 90 s /
n≈1000–1238 budget.** Two independent reasons — (1) the full anneal forgets its
seed, and warm refinement that preserves the seed produces unserializable deep
trees on the RQC and loses on the expander; (2) greedy+anneal already reaches
record-level on both targets, so there is no non-converged gap for the seed to
exploit. The draft therefore ships the reference-equivalent default (safe,
record-level) and keeps the seed/warm machinery behind env toggles
(`A029_WARM`, `A029_NOSEED`, `A029_SCT`, `A029_EXPERIMENT`) for reproducing the
attribution above. The reusable artifact is the **height-safety analysis**: the
serialization-depth limit is a real, previously-undocumented ceiling on how deep
(hence how low-tc on MPS-like circuits) any emitted tree can be.

### Deviations from the plan

- The plan's central bet (seed quality dominates at non-converged scale) did not
  hold: the budget is effectively *converged* for greedy+anneal on both targets.
- The RQC "MPS sweep seed + refine" path works numerically (fast to ~106) but is
  blocked by the tree-height serialization limit, which the plan did not
  anticipate; this is the main new finding.
- The default was switched to the library-TreeSA reference loop to guarantee no
  regression, rather than shipping the (worse or unserializable) seeded paths.


## Scored outcome (validator v2.1 pure-tc, 2026-07-23)
- status: scored, score (mean Δtc vs pre-run records): -0.3246
- record_updates: none
- reg3_1000: pass — tc=136.399 sc=129.000 record=135.75412004851293 delta=-0.645
- rqc_97_m24: pass — tc=106.473 sc=97.000 record=106.46849426445668 delta=-0.004
