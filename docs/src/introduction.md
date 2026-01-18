# omeco

High-performance tensor network contraction order optimization in Rust.

## What is omeco?

omeco (**O**ne **M**ore **E**insum **C**ontraction **O**rder) is a library for optimizing tensor network contractions. When contracting multiple tensors together, the order of contractions can make an exponential difference in computational cost. omeco provides fast algorithms to find near-optimal contraction orders.

Ported from the Julia library [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).

## Why Optimize Contraction Order?

Consider contracting three matrices: `A[10×100] × B[100×20] × C[20×5]`.

Different orders give vastly different costs:

| Order | Operations | Peak Memory |
|-------|-----------|-------------|
| `(A×B)×C` | 20,000 + 1,000 = **21,000** | 200 elements |
| `A×(B×C)` | 10,000 + 50,000 = **60,000** | 2,000 elements |

The first order is **3x faster** and uses **10x less memory**! For larger tensor networks, differences can be exponential.

Finding the optimal order is NP-complete, but heuristics find near-optimal solutions in seconds.

## Key Features

- **Fast Greedy Algorithm**: O(n² log n) greedy method for quick optimization
- **Simulated Annealing**: TreeSA finds higher-quality solutions for complex networks
- **Automatic Slicing**: TreeSASlicer reduces memory by trading time for space
- **Hardware-Aware**: Configure for CPU or GPU with memory I/O optimization
- **Python & Rust**: Native Rust with Python bindings via PyO3

## Example

**Python:**

```python
from omeco import optimize_code, contraction_complexity

# Matrix chain: A[i,j] × B[j,k] × C[k,l]
ixs = [[0, 1], [1, 2], [2, 3]]
out = [0, 3]
sizes = {0: 10, 1: 100, 2: 20, 3: 5}

tree = optimize_code(ixs, out, sizes)
complexity = contraction_complexity(tree, ixs, sizes)
print(f"Time: 2^{complexity.tc:.2f}, Space: 2^{complexity.sc:.2f}")
```

**Rust:**

```rust
use omeco::{EinCode, GreedyMethod, optimize_code, contraction_complexity};
use std::collections::HashMap;

let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3]);
let sizes = HashMap::from([(0, 10), (1, 100), (2, 20), (3, 5)]);

let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
let complexity = contraction_complexity(&tree, &code, &sizes);
println!("Time: 2^{:.2}, Space: 2^{:.2}", complexity.tc, complexity.sc);
```

## Use Cases

- **Quantum Circuit Simulation**: Optimizing tensor network contractions in quantum algorithms
- **Neural Networks**: Efficient einsum operations in deep learning
- **Scientific Computing**: Large-scale tensor computations in physics and chemistry
- **Graph Algorithms**: Computing graph polynomials and partition functions

## Next Steps

- [Installation](./installation.md) - Install omeco for Python or Rust
- [Quick Start](./quick-start.md) - Your first tensor contraction optimization
- [Concepts](./concepts/README.md) - Understand tensor networks and complexity metrics
