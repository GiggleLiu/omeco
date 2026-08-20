# Default Pipeline

Since the flags landed, `TreeSA::default()` (Python: `TreeSA()`) is no longer
just the annealing loop — it is a small pipeline that wraps annealing with a
deterministic simplification front-end and an optional, quality-only refinement
pass. This page documents what runs, what it guarantees, and how to opt out.

## What `TreeSA::default()` runs

```
simplify  ->  anneal trials  ->  [k x (surgery -> cold fine tune)]  ->  splice
                           (all search runs on the reduced network)
```

1. **Simplify** ([`crate::preprocess::simplify`]) deterministically fuses
   locally reducible structure — chains of degree-2 tensors, absorbed
   boundary tensors, and similar neighbours — into a smaller *reduced*
   network.
2. **Anneal trials** run the normal TreeSA search loop on that reduced
   network instead of the raw one, so the search budget is spent where it
   matters. With `surgery_probability > 0`, each scheduled sweep independently
   selects one global waist update with that probability; otherwise it runs the
   ordinary local sweep. The global proposal moves one boundary tensor across
   the current waist with a leaf prune-and-regraft edit, preserving every
   unaffected subtree, and uses the ordinary Metropolis test at the current
   inverse temperature.
   It does not change the temperature, consult a clock, or launch another
   annealing loop. The default probability is `0.0`, so existing runs are
   byte-identical.
3. **Interleaved anneal–surgery rounds** (the companion paper's Algorithm 1)
   only run if [`TreeSA::surgery_iters`] is greater than `0` (the default is
   `0`, off). `surgery_iters` counts *rounds*, and each round is one
   waist-surgery iteration ([`crate::waist_surgery::refine_capped`]) on the
   current reduced-network tree — improving its most expensive contraction,
   its *waist* —
   followed by a cold specified-tree fine-tuning pass over the surgical result:
   at most 15 configured levels with `β >= 1`, 30 span-gated sweeps per
   coarse-to-fine span, and three deterministic serial trials. The loop returns
   the best tree it saw anywhere, so it can only help.

   Rounds use an *incumbent ratchet*: a worse fine-tuning endpoint is reported
   but rejected, and round `r + 1` starts from the best tree retained in round
   `r`. "Best" means best under the configured multi-objective `TreeSA` score,
   so the retained tree's raw time complexity can rise when another weighted
   term improves — see the [`treesa::anneal_surgery_rounds`] docs for the exact
   ratchet contract. Surgery supplies the nonlocal basin jump; fine tuning need
   not destroy the incumbent to provide one.

   **Cost:** on the 761-tensor simplified surface-code network, fine tuning is
   450 sweeps per round, versus 15,000 sweeps for a default cold-start trial.
   Runtime still grows roughly linearly in `surgery_iters`.

   For controlled experiments, [`treesa::anneal_refine_rounds`] exposes four
   opt-in choices through `RoundsOptions`:

   - `surgery: false` is the matched cold-only arm. It runs the identical cold
     schedules, trials, seeds, incumbent ratchet, and trace construction while
     omitting only the waist-surgery call.
   - `RebuildMode::WarmRestricted` deletes off-side leaves from the incumbent,
     suppresses unary ancestors, recomputes interfaces, and uses the resulting
     side topology instead of a new greedy seed before the same cold V-cycles.
   - `SurgeryScope::Local` chooses the lowest waist ancestor with at least
     `min(n, 2|A|)` leaves, runs FM on that induced subnetwork, and splices only
     that subtree. Waists spanning at least half the network still use the root.
   - `RoundsSchedule::BandReheatThenFront { switch_fraction }` replaces only the
     rounds fine tuner: it reheats nodes within the size-scaled cost band around
     the waist (plus their root paths), then switches to a descending log-span
     freeze-out front. The switch is clamped between two band epochs and 40% of
     planned sweeps; `RoundsSchedule::Cold` keeps the original pass.

   `RoundsOptions::default()` is `surgery: true`, `Greedy`, `Root`, and `Cold`,
   exactly the historical behavior. `RoundsReport::fine_tune_sweeps_total`
   supplies a deterministic work counter for comparing these arms; the separate
   `benchmarks/surgery_ablation` driver combines it with planned TreeSA node
   visits and wall time.
4. **Splice** ([`crate::preprocess::splice`]) expands each reduced-network
   leaf back into the binary subtree `simplify` merged it from, so the
   returned tree's leaves are exactly the original tensors again. Splicing
   happens after the optional rounds: the paper's waist and FM cut are defined
   on the simplified hypergraph.

This whole pipeline is what `optimize_code(ixs, out, sizes, TreeSA())` /
`optimize_code(&code, &sizes, &TreeSA::default())` runs by default — no flags
needed to get it.

```python
from omeco import optimize_code, TreeSA

tree = optimize_code(ixs, out, sizes, TreeSA())                      # default pipeline
better = optimize_code(ixs, out, sizes, TreeSA(surgery_iters=3))     # + 3 anneal-surgery rounds
```

```rust
use omeco::{optimize_code, TreeSA};

let tree = optimize_code(&code, &sizes, &TreeSA::default()).unwrap();
let better = optimize_code(&code, &sizes, &TreeSA::default().with_surgery_iters(3)).unwrap();
```

## Determinism

The whole `TreeSA` API is **fully deterministic and machine-independent**:
simplify, the anneal trials, splice, and — since `surgery_iters` counts rounds
rather than seconds — the anneal–surgery loop too are all pure functions of
the input network and config (internal RNG seeds are fixed, and no wall-clock
deadline binds `optimize_treesa`). Re-running the same configuration
reproduces the same tree on any machine, with `surgery_iters` at any value:
`0` (off), or any positive round count. Every positive-round result is guarded
against the rounds-off baseline after splice-back. The standalone
`anneal_surgery_rounds` loop is monotone in its round count, under the
configured `TreeSA` score, on the network it is given.

Wall-clock budgets still exist, but only on the low-level
[`crate::waist_surgery::refine`] / [`refine_capped`] APIs (and the Python
`waist_refine` function) for power users composing their own pipeline outside
`TreeSA`; see [Simplification, Warm-Start, and Waist Surgery](./paper-algorithms.md).

## Preprocessing guarantees

`simplify` + `splice` come with two guarantees and one deliberate
non-guarantee — see the [`crate::preprocess`] module docs for the full
argument:

- **Exactness — yes.** The spliced tree computes the same einsum result as
  optimizing the original network directly, and every original tensor is
  still a leaf of the returned tree (`splice` only expands merged
  super-tensors; it never drops or duplicates a tensor).
- **Space-safety — yes.** `simplify` only fuses a pair when the resulting
  intermediate is no larger than its larger input, so the achievable space
  complexity is never pushed above the original network's floor. Merging can
  only shrink or hold the peak, never grow it.
- **tc-optimality — no.** There is no theorem that the eager, local merge
  rule finds the globally time-optimal reduced network; in principle an
  eager merge could exclude the true tc-optimum. Empirically this is rare —
  on structured circuits the pass is a large net win (see below), and on
  networks with no reducible local structure it is close to a no-op.

On raw Sycamore circuits, simplification shrinks the tensor count by
~89% before annealing ever runs; on 3-regular random graphs, which carry
essentially no locally reducible structure, it is a near no-op (the reduced
network is close to the same size as the original).

## When surgery helps

Waist surgery targets a specific failure mode: local annealing rewrites move
one subtree at a time and can get stuck unable to jump between two distinct,
comparably-sized bipartitions of the network — a **frozen waist**. Surgery
extracts the tree's most expensive cut, re-optimizes it directly on the
tensor hypergraph with balance-constrained Fiduccia–Mattheyses passes, and
only keeps the result if it strictly lowers `tc`.

The cold fine-tuning pass can itself produce most or all of a rounds arm's gain,
so mechanism claims must compare surgery against the `surgery: false` matched
control rather than against plain TreeSA alone. When surgery does find a useful
cut, the following cold pass settles the rest of the tree around that nonlocal
jump; this is why the API exposes rounds rather than back-to-back surgery calls.

- **Helps most** on frozen-waist-prone instances, where the network has
  genuinely distinct good bipartitions of similar cost that local mutations
  struggle to reach. In the companion paper's measurements, 76% of surgery
  calls on a surface-code d=21 network and 31% on a king-graph network found
  a strictly cheaper comparable-balance cut.
- **Near-no-op elsewhere.** On instances without a frozen waist, surgery's
  bounded search typically returns early after confirming the incumbent cut
  is locally minimal — the round then degenerates to a cold fine-tuning pass,
  which costs time but never regresses the returned tree.

If you don't know in advance whether your network has this structure, a small
positive `surgery_iters` is a safe thing to try: worst case you spend a few
extra fine-tuning passes for no improvement, best case you recover a
meaningfully cheaper tree. Start at 2–5 rounds and scale up only if the
retained trajectory is still improving.

## Escape hatches

| Want | Set |
|---|---|
| Skip simplification, anneal the raw network directly | `preprocess: false` (Rust: `.with_preprocess(false)`; Python: `TreeSA(preprocess=False)`) |
| Skip the anneal–surgery rounds (already the default) | `surgery_iters: 0` (Rust: `.with_surgery_iters(0)`; Python: `TreeSA(surgery_iters=0)`) |

`TreeSA::path()` sets `preprocess: false` by construction, deliberately —
`splice` is decomposition-agnostic: it substitutes each reduced-network leaf
with whatever binary subtree `simplify` produced for it, which is not
path-shaped in general. Running the front-end under `path()` could give the
spliced tree a node with two non-leaf children and break that preset's
documented linear-contraction-order guarantee, so it opts out unconditionally
rather than risk it.

For the same reason, a `Path` decomposition also skips both forms of surgery:
the anneal–surgery rounds and the probabilistic surgery update rule. Neither
the surgical rebuild nor the specified-tree fine tuning is path-preserving.
Setting `surgery_iters` or `surgery_probability` on a path config is silently
ignored rather than allowed to return a non-path tree.

## See also

- [Simplification, Warm-Start, and Waist Surgery](./paper-algorithms.md) —
  the lower-level building blocks (`simplify_then_optimize`, `refine`,
  warm-start annealing) if you want to compose your own pipeline instead of
  the managed default.
- [TreeSA](./tree-sa.md) — the annealing search itself and its tuning knobs.

[`crate::preprocess`]: https://docs.rs/omeco/latest/omeco/preprocess/index.html
[`crate::preprocess::simplify`]: https://docs.rs/omeco/latest/omeco/preprocess/fn.simplify.html
[`crate::preprocess::splice`]: https://docs.rs/omeco/latest/omeco/preprocess/fn.splice.html
[`crate::waist_surgery::refine`]: https://docs.rs/omeco/latest/omeco/waist_surgery/fn.refine.html
[`crate::waist_surgery::refine_capped`]: https://docs.rs/omeco/latest/omeco/waist_surgery/fn.refine_capped.html
[`refine_capped`]: https://docs.rs/omeco/latest/omeco/waist_surgery/fn.refine_capped.html
[`TreeSA::surgery_iters`]: https://docs.rs/omeco/latest/omeco/treesa/struct.TreeSA.html#structfield.surgery_iters
[`treesa::anneal_surgery_rounds`]: https://docs.rs/omeco/latest/omeco/treesa/fn.anneal_surgery_rounds.html
[`treesa::anneal_refine_rounds`]: https://docs.rs/omeco/latest/omeco/treesa/fn.anneal_refine_rounds.html
