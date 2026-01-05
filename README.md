# One More Einsum Contraction Order (OMECO)

[![CI](https://github.com/GiggleLiu/omeco/workflows/CI/badge.svg)](https://github.com/GiggleLiu/omeco/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/omeco/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/omeco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tensor network contraction order optimization in Rust.

Ported from [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).

## Overview

When contracting multiple tensors together, the order of contractions significantly affects computational cost. Finding the optimal contraction order is NP-complete, but good heuristics can find near-optimal solutions quickly.

This library provides:

- **GreedyMethod**: Fast O(n² log n) greedy algorithm for contraction order
- **TreeSA**: Simulated annealing for higher quality contraction orders
- **TreeSASlicer**: Automatic slicing optimization to reduce memory usage

## Installation

### Python

```bash
pip install omeco
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
omeco = "0.1"
```

## Python Quick Start

```python
from omeco import (
    optimize_code, slice_code, contraction_complexity, sliced_complexity,
    GreedyMethod, TreeSA, TreeSASlicer, ScoreFunction
)

# Matrix chain: A[0,1] × B[1,2] × C[2,3] → D[0,3]
ixs = [[0, 1], [1, 2], [2, 3]]
out = [0, 3]
sizes = {0: 100, 1: 200, 2: 50, 3: 100}

# 1) Optimize contraction order
tree = optimize_code(ixs, out, sizes, GreedyMethod())
complexity = contraction_complexity(tree, ixs, sizes)
print(f"Time: 2^{complexity.tc:.2f}, Space: 2^{complexity.sc:.2f}")

# 2) Slice to reduce memory (automatic optimization)
slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=10.0))
sliced = slice_code(tree, ixs, sizes, slicer)
sliced_comp = sliced_complexity(sliced, ixs, sizes)
print(f"Sliced indices: {sliced.slicing()}")
print(f"Sliced space: 2^{sliced_comp.sc:.2f}")

# Use with PyTorch (see examples/pytorch_tensor_network_example.py)
tree_dict = tree.to_dict()  # Convert to dict for traversal
```

## Rust Quick Start

Two core features are exposed in the quick start below: optimizing contraction
orders and slicing for lower peak memory.

```rust
use omeco::{
    EinCode, GreedyMethod, TreeSASlicer, slice_code,
    contraction_complexity, optimize_code, sliced_complexity,
};
use std::collections::HashMap;

// Matrix chain: A[i,j] * B[j,k] * C[k,l] -> D[i,l]
let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    vec!['i', 'l']
);

// Define tensor dimensions
let mut sizes = HashMap::new();
sizes.insert('i', 100);
sizes.insert('j', 200);
sizes.insert('k', 50);
sizes.insert('l', 100);

// 1) Optimize contraction order
let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();

let complexity = contraction_complexity(&optimized, &sizes, &code.ixs);
println!("Time complexity: 2^{:.2}", complexity.tc);
println!("Space complexity: 2^{:.2}", complexity.sc);

// 2) Slice to reduce memory (automatic optimization)
let slicer = TreeSASlicer::fast().with_sc_target(10.0);
let sliced = slice_code(&optimized, &sizes, &slicer, &code.ixs).unwrap();
let sliced_comp = sliced_complexity(&sliced, &sizes, &code.ixs);
println!("Sliced space complexity: 2^{:.2}", sliced_comp.sc);
```

## Documentation

API documentation is available via `cargo doc --open` or at [docs.rs/omeco](https://docs.rs/omeco).

## Algorithms

### GreedyMethod

Iteratively contracts the tensor pair with minimum cost:

```rust
use omeco::{EinCode, optimize_code, GreedyMethod};

let code = EinCode::new(
    vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd']],
    vec!['a', 'd']
);
let sizes = omeco::uniform_size_dict(&code, 10);

// Default: deterministic greedy
let result = optimize_code(&code, &sizes, &GreedyMethod::default());

// Stochastic greedy with temperature
let stochastic = GreedyMethod::new(0.0, 1.0);
let result = optimize_code(&code, &sizes, &stochastic);
```

**Parameters:**
- `alpha`: Balances output size vs input size reduction (0.0 to 1.0)
- `temperature`: Enables Boltzmann sampling (0.0 = deterministic)

### TreeSA

Simulated annealing with local tree mutations:

```rust
use omeco::{EinCode, optimize_code, TreeSA, ScoreFunction};

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l'], vec!['l', 'm']],
    vec!['i', 'm']
);
let sizes = omeco::uniform_size_dict(&code, 10);

// Fast configuration (fewer iterations)
let result = optimize_code(&code, &sizes, &TreeSA::fast());

// Default configuration (higher quality)
let result = optimize_code(&code, &sizes, &TreeSA::default());

// Custom configuration with space constraint
let custom = TreeSA::default()
    .with_sc_target(15.0)  // Target space complexity
    .with_ntrials(20);     // More parallel trials
let result = optimize_code(&code, &sizes, &custom);
```

**Parameters:**
- `betas`: Inverse temperature schedule
- `ntrials`: Number of parallel trials (uses rayon)
- `niters`: Iterations per temperature level
- `score`: Scoring function with complexity weights

## Complexity Metrics

Three complexity metrics are computed (all in log2 scale):

```rust
use omeco::{EinCode, optimize_code, GreedyMethod, contraction_complexity};

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k']],
    vec!['i', 'k']
);
let mut sizes = std::collections::HashMap::new();
sizes.insert('i', 100);
sizes.insert('j', 200);
sizes.insert('k', 100);

let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
let c = contraction_complexity(&optimized, &sizes, &code.ixs);

println!("Time complexity (FLOPs): 2^{:.2} = {:.2e}", c.tc, c.flops());
println!("Space complexity (memory): 2^{:.2} = {:.2e}", c.sc, c.peak_memory());
println!("Read-write complexity: 2^{:.2}", c.rwc);
```

## Custom Scoring

Control the trade-off between time and space complexity:

```rust
use omeco::{ScoreFunction, TreeSA};

// Optimize primarily for time (ignore space)
let time_score = ScoreFunction::time_optimized();

// Optimize for space with target of 2^15 elements
let space_score = ScoreFunction::space_optimized(15.0);

// Custom weights
let custom_score = ScoreFunction::new(
    1.0,   // tc_weight
    2.0,   // sc_weight (penalize space more)
    0.0,   // rw_weight
    20.0,  // sc_target
);

let config = TreeSA {
    score: custom_score,
    ..TreeSA::default()
};
```

## Working with NestedEinsum

The optimized result is a `NestedEinsum` representing the contraction tree:

```rust
use omeco::{EinCode, NestedEinsum, optimize_code, GreedyMethod};

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    vec!['i', 'l']
);
let sizes = omeco::uniform_size_dict(&code, 10);

let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();

// Inspect the tree
println!("Is binary tree: {}", tree.is_binary());
println!("Number of tensors: {}", tree.leaf_count());
println!("Tree depth: {}", tree.depth());

// Get contraction order (leaf indices)
let order = tree.leaf_indices();
println!("Contraction involves tensors: {:?}", order);
```

## Sliced Contractions

For memory-constrained scenarios, use `SlicedEinsum`:

```rust
use omeco::{NestedEinsum, SlicedEinsum, sliced_complexity};

// Assume we have an optimized tree
let leaf0 = NestedEinsum::<char>::leaf(0);
let leaf1 = NestedEinsum::<char>::leaf(1);
let eins = omeco::EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k']],
    vec!['i', 'k']
);
let tree = NestedEinsum::node(vec![leaf0, leaf1], eins.clone());

// Slice over index 'j' to reduce memory
let sliced = SlicedEinsum::new(vec!['j'], tree);

println!("Number of slices: {}", sliced.num_slices());
```

## Integer Labels

For programmatic use, integer labels are often more convenient:

```rust
use omeco::{EinCode, optimize_code, GreedyMethod};
use std::collections::HashMap;

// Using usize labels instead of char
let code: EinCode<usize> = EinCode::new(
    vec![vec![0, 1], vec![1, 2], vec![2, 3]],
    vec![0, 3]
);

let mut sizes = HashMap::new();
sizes.insert(0, 100);
sizes.insert(1, 200);
sizes.insert(2, 200);
sizes.insert(3, 100);

let result = optimize_code(&code, &sizes, &GreedyMethod::default());
```

## Performance Tips

1. **Use TreeSA::fast() for quick results** - Fewer iterations, single trial
2. **Increase ntrials for large problems** - More parallel exploration
3. **Set sc_target based on available memory** - Constrains space complexity
4. **Use GreedyMethod for very large networks** - O(n² log n) vs O(n² × iterations)

## Benchmarks

We benchmark TreeSA performance by comparing the Rust implementation (exposed to Python via PyO3) against the original Julia implementation ([OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl)).

**Hardware:** Intel Xeon Gold 6226R @ 2.90GHz

**Configuration:** `ntrials=1`, `niters=50-100`, `βs=0.01:0.05:15.0`

### TreeSA Results

| Problem | Tensors | Indices | Rust (ms) | Julia (ms) | Rust tc | Julia tc | Speedup |
|---------|---------|---------|-----------|------------|---------|----------|---------|
| chain_10 | 10 | 11 | 22.9 | 31.6 | 23.10 | 23.10 | **1.38x** |
| grid_4x4 | 16 | 24 | 132.4 | 150.7 | 9.18 | 9.26 | **1.14x** |
| grid_5x5 | 25 | 40 | 264.0 | 297.7 | 10.96 | 10.96 | **1.13x** |
| reg3_250 | 250 | 372 | 4547 | 5099 | 48.01 | 47.17 | **1.12x** |

**Key observations:**
- Rust is **10-40% faster** than Julia for TreeSA optimization
- Both implementations find solutions with comparable time complexity (tc)
- The `reg3_250` benchmark (random 3-regular graph with 250 nodes) shows TreeSA reduces tc from ~66 (greedy) to ~48, a **27% improvement**

To run the benchmarks yourself:

```bash
# Rust (via Python)
cd benchmarks && python benchmark_python.py

# Julia
cd benchmarks && julia --project=. benchmark_julia.jl
```

## License

MIT
