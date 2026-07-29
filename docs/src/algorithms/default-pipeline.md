# Default Pipeline

Since the flags landed, `TreeSA::default()` (Python: `TreeSA()`) is no longer
just the annealing loop — it is a small pipeline that wraps annealing with a
deterministic simplification front-end and an optional, quality-only refinement
pass. This page documents what runs, what it guarantees, and how to opt out.

## What `TreeSA::default()` runs

```
simplify  ->  anneal trials (on the reduced network)  ->  splice  ->  [k x (surgery -> anneal), if k > 0]
```

1. **Simplify** ([`crate::preprocess::simplify`]) deterministically fuses
   locally reducible structure — chains of degree-2 tensors, absorbed
   boundary tensors, and similar neighbours — into a smaller *reduced*
   network.
2. **Anneal trials** run the normal TreeSA search loop on that reduced
   network instead of the raw one, so the search budget is spent where it
   matters.
3. **Splice** ([`crate::preprocess::splice`]) expands each reduced-network
   leaf back into the binary subtree `simplify` merged it from, so the
   returned tree's leaves are exactly the original tensors again.
4. **Interleaved anneal–surgery rounds** (the companion paper's Algorithm 1)
   only run if [`TreeSA::surgery_iters`] is greater than `0` (the default is
   `0`, off). `surgery_iters` counts *rounds*, and each round is one
   waist-surgery iteration ([`crate::waist_surgery::refine_capped`]) on the
   current tree — improving its most expensive contraction, its *waist* —
   followed by a full warm-started annealing pass over the surgical result.
   The loop returns the best tree it saw anywhere, so it can only help.

   Rounds are *chained*: round `r + 1` continues from round `r`'s annealed
   tree even when that tree was worse than the incumbent best. Letting the
   trajectory go uphill is the point — it is what lets surgery carry the
   search out of a basin that local annealing moves cannot leave.

   **Cost: one round ≈ one more full anneal of the network**, so runtime grows
   roughly linearly in `surgery_iters`. Budget it like extra trials, not like
   a cheap post-pass.

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
`0` (off), or any positive round count. More rounds can only be equal or
better, never worse.

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

Interleaving matters as much as the surgery itself: a surgical cut is a jump
to a different basin, and the annealing pass that follows it in the same
round is what settles the rest of the tree around the new cut. That is why
`surgery_iters` runs rounds rather than back-to-back surgery iterations.

- **Helps most** on frozen-waist-prone instances, where the network has
  genuinely distinct good bipartitions of similar cost that local mutations
  struggle to reach. In the companion paper's measurements, 76% of surgery
  calls on a surface-code d=21 network and 31% on a king-graph network found
  a strictly cheaper comparable-balance cut.
- **Near-no-op elsewhere.** On instances without a frozen waist, surgery's
  bounded search typically returns early after confirming the incumbent cut
  is locally minimal — the round then degenerates to an extra warm-started
  anneal, which costs time but never regresses the returned tree.

If you don't know in advance whether your network has this structure, a small
positive `surgery_iters` is a safe thing to try: worst case you spend a few
extra anneals for no improvement, best case you recover a meaningfully
cheaper tree. Since each round costs about one anneal, start at 2–5 rounds
and scale up only if the trajectory is still improving.

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

For the same reason, a `Path` decomposition also skips the anneal–surgery
rounds: neither the surgical rebuild nor the warm-started anneal is
path-preserving. Setting `surgery_iters` on a path config is silently ignored
rather than allowed to return a non-path tree.

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
