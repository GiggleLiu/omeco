# Greedy Method

Fast greedy algorithm for tensor contraction order optimization.

## How It Works

The greedy algorithm repeatedly contracts the pair of tensors with minimum cost until one tensor remains.

**Algorithm**:
```
while more than one tensor remains:
    1. Consider all pairs of tensors
    2. Compute cost of contracting each pair
    3. Contract the pair with minimum cost
    4. Replace the pair with their contraction result
```

**Time Complexity**: O(n² log n) where n is the number of tensors.

## Basic Usage

### Python

```python
from omeco import optimize_code, GreedyMethod

# Deterministic greedy (default optimizer)
tree = optimize_code(ixs, out, sizes)

# Or explicitly
tree = optimize_code(ixs, out, sizes, GreedyMethod())
```

### Rust

```rust
use omeco::{EinCode, GreedyMethod, optimize_code};

let method = GreedyMethod::default();
let tree = optimize_code(&code, &sizes, &method)?;
```

## Stochastic Variants

Add randomness to explore more solutions:

```python
# alpha: controls randomness (0 = deterministic, 1 = fully random)
# temperature: softmax temperature for selection
method = GreedyMethod(alpha=0.5, temperature=1.0)
tree = optimize_code(ixs, out, sizes, method)
```

**Parameters**:
- `alpha=0.0` (default): Always pick minimum cost (deterministic)
- `alpha=0.5`: Mix of greedy and random choices
- `alpha=1.0`: Uniform random selection
- `temperature`: Controls selection distribution (higher = more random)

## Performance Characteristics

**Advantages**:
- ⚡ Very fast: seconds for 100+ tensors
- 🎯 Deterministic by default (reproducible)
- 📈 Scales well to large networks
- 💡 Good baseline for most cases

**Limitations**:
- 🎲 Can get stuck in local optima
- 🔍 Myopic: only considers immediate cost
- 📊 May miss global optimal solution

## Example: Matrix Chain

```python
from omeco import optimize_code, contraction_complexity

# A[100×10] × B[10×20] × C[20×5]
ixs = [[0, 1], [1, 2], [2, 3]]
out = [0, 3]
sizes = {0: 100, 1: 10, 2: 20, 3: 5}

tree = optimize_code(ixs, out, sizes)
print(tree)
```

Output:
```
ab, bd -> ad
├─ tensor_0
└─ bc, cd -> bd
   ├─ tensor_1
   └─ tensor_2
```

This contracts `B×C` first (cost: 10×20×5 = 1,000), then `A×(BC)` (cost: 100×10×5 = 5,000).
Total: 6,000 FLOPs.

Alternative order `(A×B)×C` would cost: 100×10×20 + 100×20×5 = 30,000 FLOPs (5x worse!).

## When to Use

✅ **Use GreedyMethod when**:
- You need quick results (prototyping, iteration)
- Network is straightforward (chains, grids)
- Memory/time constraints are relaxed

❌ **Consider TreeSA instead when**:
- Greedy result is too slow/large
- Network is complex (irregular graphs)
- You have time for better optimization
- Result will be used repeatedly

## Tips

1. **Start with default**: `GreedyMethod()` works for most cases

2. **Try stochastic for variety**:
   ```python
   # Run 10 times with randomness, pick best
   best_tree = None
   best_complexity = float('inf')

   for _ in range(10):
       tree = optimize_code(ixs, out, sizes, GreedyMethod(alpha=0.3))
       complexity = contraction_complexity(tree, ixs, sizes)
       if complexity.tc < best_complexity:
           best_tree = tree
           best_complexity = complexity.tc
   ```

3. **Combine with slicing** if memory is tight:
   ```python
   tree = optimize_code(ixs, out, sizes)
   if contraction_complexity(tree, ixs, sizes).sc > 25.0:
       sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast())
   ```

## Next Steps

- [TreeSA](./tree-sa.md) - For higher quality solutions
- [Algorithm Comparison](./comparison.md) - Benchmark results
- [Quick Start](../quick-start.md) - Complete examples
