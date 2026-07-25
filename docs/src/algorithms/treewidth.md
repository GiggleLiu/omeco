# Treewidth

Scalable treewidth-heuristic optimizer based on variable-elimination ordering.

## How It Works

The tensor network is viewed through its *primal graph*: index labels are
vertices, and every input tensor is a clique over the labels it carries.
Contracting the network is equivalent to eliminating the labels one at a time;
the cost is governed by the largest clique (bag) formed along the way — the
*treewidth* of the network.

Treewidth computes an elimination order with a **weighted minimum-degree**
heuristic, on a *quotient graph* with element absorption in the style of
sparse-matrix AMD codes:

```
build the quotient graph (tensors = initial elements/cliques)
while a live label remains:
    1. pick the label of smallest weighted degree
       (weighted degree = summed log2 dims of its current neighborhood)
    2. form a new clique over that neighborhood; absorb its old elements
    3. rescore the affected neighbors
```

Fill edges are never materialized, so the ordering scales to tens of thousands
of tensors and labels in milliseconds. Ties break by label id, so the order —
and the resulting contraction tree — is fully **deterministic**. The order is
then replayed into a binary contraction tree: at each elimination step the
tensors sharing the eliminated label are contracted into one intermediate.

Output labels (`iy`) are never eliminated, so they survive to the root.

**Time Complexity**: near-linear in the number of tensors for sparse,
structured networks.

## Basic Usage

### Python

```python
from omeco import optimize_code, Treewidth

tree = optimize_code(ixs, out, sizes, Treewidth())

# Or the standalone helper
from omeco import optimize_treewidth
tree = optimize_treewidth(ixs, out, sizes, Treewidth())
```

### Rust

```rust
use omeco::{EinCode, Treewidth, optimize_code};

let method = Treewidth::min_degree();
let tree = optimize_code(&code, &sizes, &method)
    .expect("optimization succeeds");
```

## When to Use

Treewidth shines on large, structured networks — probabilistic graphical
models, factor graphs, and relational-inference instances — where a
low-treewidth elimination order is dramatically better than what pairwise-greedy
or simulated-annealing search can reach in the same time. On such instances the
elimination order can find the optimal treewidth (e.g. width 100 on a
30,400-tensor / 10,200-label relational instance) in well under a second, while
annealing-family optimizers stall far above it.

For small or unstructured networks, [Greedy](./greedy-method.md) and
[TreeSA](./tree-sa.md) remain good choices.

## Alignment with Julia

This mirrors the `Treewidth` optimizer of
[OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl),
which backs its orderings with CliqueTrees.jl. omeco currently ships the
weighted minimum-degree heuristic (`EliminationAlgorithm::MinDegree`); the
algorithm enum is designed so additional heuristics (e.g. minimum-fill) can be
added later without breaking the API.
