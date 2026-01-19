# Quick Start

This guide shows the basics of tensor contraction optimization.

## Basic Contraction Optimization

### Python

```python
from omeco import optimize_code

# Matrix chain: A[i,j] × B[j,k] × C[k,l] → D[i,l]
ixs = [[0, 1], [1, 2], [2, 3]]  # Input tensor indices
out = [0, 3]                     # Output indices
sizes = {0: 10, 1: 100, 2: 20, 3: 5}  # Dimension sizes

# Optimize contraction order
tree = optimize_code(ixs, out, sizes)

# Compute complexity
complexity = tree.complexity(ixs, sizes)
print(f"Time complexity: 2^{complexity.tc:.2f} FLOPs")
print(f"Space complexity: 2^{complexity.sc:.2f} elements")
print(f"Actual FLOPs: {complexity.flops():,.0f}")
print(f"Peak memory: {complexity.peak_memory():,.0f} elements")
```

**Output:**
```
Time complexity: 2^14.29 FLOPs
Space complexity: 2^11.29 elements
Actual FLOPs: 21,000
Peak memory: 2,500 elements
```

**Visualize the contraction tree:**

```python
print(tree)
```

```
ab, bd -> ad
├─ tensor_0
└─ bc, cd -> bd
   ├─ tensor_1
   └─ tensor_2
```

### Rust

```rust
use omeco::{
    EinCode, GreedyMethod, optimize_code
};
use std::collections::HashMap;

fn main() {
    // Matrix chain: A[i,j] × B[j,k] × C[k,l] → D[i,l]
    let code = EinCode::new(
        vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        vec![0, 3]
    );
    let sizes = HashMap::from([
        (0, 10),
        (1, 100),
        (2, 20),
        (3, 5),
    ]);

    // Optimize with default greedy method
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default())
        .expect("Optimization failed");

    // Compute complexity
    let complexity = contraction_complexity(&tree, &code, &sizes);
    
    println!("Time: 2^{:.2}, Space: 2^{:.2}", 
             complexity.tc, complexity.sc);
    println!("Leaves: {}, Depth: {}", 
             tree.leaf_count(), tree.depth());
}
```

## Understanding the Output

### NestedEinsum Structure

The optimized contraction tree shows the order of operations:

```
ab, bd -> ad
├─ tensor_0
└─ bc, cd -> bd
   ├─ tensor_1
   └─ tensor_2
```

This means:
1. First contract `tensor_1` (indices `bc`) with `tensor_2` (indices `cd`) → result has indices `bd`
2. Then contract `tensor_0` (indices `ab`) with the result → final output has indices `ad`

### Complexity Metrics

- **tc (time complexity)**: log₂ of total FLOPs needed
- **sc (space complexity)**: log₂ of peak memory usage
- **rwc (read-write complexity)**: log₂ of memory I/O operations (for GPU optimization)

Lower values = better performance.

## Using Different Optimizers

### Greedy Method (Fast)

```python
from omeco import GreedyMethod

# Deterministic greedy (default)
tree = optimize_code(ixs, out, sizes, GreedyMethod())

# Stochastic greedy (explores more options)
tree = optimize_code(ixs, out, sizes, GreedyMethod(alpha=0.5, temperature=1.0))
```

### TreeSA (Higher Quality)

```python
from omeco import TreeSA, ScoreFunction

# Fast preset (good for most cases)
tree = optimize_code(ixs, out, sizes, TreeSA.fast())

# Custom configuration
score = ScoreFunction(sc_target=15.0)
optimizer = TreeSA(ntrials=10, niters=50, score=score)
tree = optimize_code(ixs, out, sizes, optimizer)
```

## Working with the Tree

### Convert to Dictionary

```python
tree_dict = tree.to_dict()

# Traverse the tree
def traverse(node):
    if "tensor_index" in node:
        print(f"Leaf: tensor {node['tensor_index']}")
    else:
        print(f"Contraction: {node['eins']}")
        for child in node["args"]:
            traverse(child)

traverse(tree_dict)
```

### Tree Properties

```python
# Number of input tensors
num_tensors = tree.leaf_count()

# Depth of the tree
depth = tree.depth()

# Check if binary tree
is_binary = tree.is_binary()

# Get leaf indices in order
leaves = tree.leaf_indices()
```

## Memory Reduction with Slicing

For large tensor networks that don't fit in memory:

```python
from omeco import slice_code, TreeSASlicer, ScoreFunction

# Optimize contraction first
tree = optimize_code(ixs, out, sizes)

# Check if memory is too large
complexity = tree.complexity(ixs, sizes)
if complexity.sc > 25.0:  # More than 2^25 elements (~256MB for float64)
    # Slice to reduce memory
    score = ScoreFunction(sc_target=25.0)
    slicer = TreeSASlicer.fast(score=score)
    sliced = slice_code(tree, ixs, sizes, slicer)
    
    # Check new complexity
    sliced_comp = sliced.complexity(ixs, sizes)
    print(f"Original space: 2^{complexity.sc:.2f}")
    print(f"Sliced space: 2^{sliced_comp.sc:.2f}")
    print(f"Sliced indices: {sliced.slicing()}")
```

## Next Steps

- [Concepts](./concepts/README.md) - Understand tensor networks and complexity
- [Algorithms](./algorithms/README.md) - Learn about GreedyMethod and TreeSA
- [Score Function Configuration](./guides/score-function.md) - Optimize for your hardware
- [PyTorch Integration](./guides/pytorch-integration.md) - Use with PyTorch tensors
