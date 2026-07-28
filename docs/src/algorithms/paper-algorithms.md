# Simplification, Warm-Start, and Waist Surgery

Three composable building blocks let you push contraction quality beyond a single
optimizer pass. They are most effective on structured networks (quantum circuits,
error-correcting codes) where a raw greedy or TreeSA pass leaves value on the table.

## Structural simplification (`omeco::preprocess`)

Raw circuits carry large amounts of locally reducible structure: chains of
single-qubit gates, absorbed boundary tensors, and other neighbours whose merge
cannot grow an intermediate. [`simplify`] deterministically fuses pairs that are
both rank- and dimension-aware size-non-increasing, [`splice`] expands the
collapsed super-tensors back to the original tensor indices, and
[`simplify_then_optimize`] chains the two around any optimizer.

```rust
use omeco::preprocess::simplify_then_optimize;
use omeco::{EinCode, GreedyMethod};
use std::collections::HashMap;

let code = EinCode::new(
    vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd'], vec!['d', 'e']],
    vec!['a', 'e'],
);
let sizes: HashMap<char, usize> =
    [('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)].into();
let (tree, report) =
    simplify_then_optimize(&code, &sizes, &GreedyMethod::default()).unwrap();
println!("shrink = {:.1}%", 100.0 * report.shrink);
```

On raw Sycamore-53 circuits the pass removes ~89% of tensors (3369 → 381) before
the optimizer ever runs, and the returned tree's leaves are exactly the original
tensors. See the `preprocess_optimize` example.

## Warm-start annealing (`omeco::treesa`)

[`prepare_warm_anneal`] turns any seed `NestedEinsum` (e.g. a greedy result) into
a mutable `ExprTree` plus the metadata needed to drive your own annealing loop
over the public `expr_tree` rewrite utilities; [`warm_exprtree_to_nested`] converts
the annealed tree back. This gives you full control of the schedule, stopping rule,
and acceptance criterion — useful for wall-clock-bounded, eager-output search — in
contrast to the fully-managed `optimize_treesa`. See the `warm_start_anneal`
example.

## Waist surgery (`omeco::waist_surgery`)

The root contraction induces a whole-network *waist*, a bipartition of the
tensors. Local SA rewrites can struggle to jump between distinct good
bipartitions of the same size. [`refine`] extracts that root cut, improves it on
the tensor hypergraph with balance-constrained Fiduccia–Mattheyses passes,
rebuilds both sides, and accepts only when the global `tc` strictly drops. Its
budget is checked cooperatively within search loops and between rebuild stages.

```rust
use omeco::waist_surgery::refine;
use omeco::{optimize_code, EinCode, GreedyMethod};
use std::collections::HashMap;
use std::time::Duration;

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    vec!['i', 'l'],
);
let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(200));
println!("rebuild accepts = {}", report.rebuild_accepts);
```

The refined tree is always over the original tensor indices and never worse than
the seed. On `surfacecode_d21` (2203 tensors) it takes a greedy seed from `tc=67.1`
to `tc=49.1` in 20 seconds. See the `waist_refine` example.

[`simplify`]: https://docs.rs/omeco/latest/omeco/preprocess/fn.simplify.html
[`splice`]: https://docs.rs/omeco/latest/omeco/preprocess/fn.splice.html
[`simplify_then_optimize`]: https://docs.rs/omeco/latest/omeco/preprocess/fn.simplify_then_optimize.html
[`prepare_warm_anneal`]: https://docs.rs/omeco/latest/omeco/treesa/fn.prepare_warm_anneal.html
[`warm_exprtree_to_nested`]: https://docs.rs/omeco/latest/omeco/treesa/fn.warm_exprtree_to_nested.html
[`refine`]: https://docs.rs/omeco/latest/omeco/waist_surgery/fn.refine.html
