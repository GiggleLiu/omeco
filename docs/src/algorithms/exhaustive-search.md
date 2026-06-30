# Exhaustive Search

`ExhaustiveSearch` is an exact contraction-order optimizer for small tensor
networks. It uses dynamic programming over tensor subsets and minimizes total
contraction FLOP count within each connected component.

## Usage

Python:

```python
from omeco import ExhaustiveSearch, optimize_code

ixs = [[0, 1], [1, 2], [2, 3]]
out = [0, 3]
sizes = {0: 2, 1: 3, 2: 4, 3: 5}

tree = optimize_code(ixs, out, sizes, ExhaustiveSearch())
```

Rust:

```rust
use omeco::{EinCode, ExhaustiveSearch, optimize_code};

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    vec!['i', 'l'],
);
let sizes = omeco::uniform_size_dict(&code, 10);

let tree = optimize_code(&code, &sizes, &ExhaustiveSearch::default()).unwrap();
```

## Scope

The exact search supports hyperedges and shared output indices, including batch
or diagonal-style indices that appear in multiple tensors and remain in the
output. Disconnected networks are optimized component by component and combined
with outer products.

For nontrivial networks, `ExhaustiveSearch` rejects partial traces and dangling
summed indices because those require unary tensor operations outside the binary
contraction tree search. One- and two-tensor inputs are returned directly.

## When To Use

Use `ExhaustiveSearch` for small networks, regression tests, and exact baselines
when comparing heuristic optimizers. For larger networks, use `GreedyMethod` for
speed or `TreeSA` for a higher-quality heuristic search.
