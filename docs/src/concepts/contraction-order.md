# Contraction Order Problem

## The Problem

When contracting multiple tensors, the order matters exponentially.

### Simple Example: Three Matrices

Consider `A[10×100] × B[100×20] × C[20×5]`:

**Option 1: Left-to-right `(A×B)×C`**
```
Step 1: A[10×100] × B[100×20] = T[10×20]
        Cost: 10 × 100 × 20 = 20,000 FLOPs
        Memory: 200 elements

Step 2: T[10×20] × C[20×5] = D[10×5]
        Cost: 10 × 20 × 5 = 1,000 FLOPs
        Memory: 50 elements

Total: 21,000 FLOPs, peak memory 200 elements
```

**Option 2: Right-to-left `A×(B×C)`**
```
Step 1: B[100×20] × C[20×5] = T[100×5]
        Cost: 100 × 20 × 5 = 10,000 FLOPs
        Memory: 500 elements

Step 2: A[10×100] × T[100×5] = D[10×5]
        Cost: 10 × 100 × 5 = 50,000 FLOPs
        Memory: 50 elements

Total: 60,000 FLOPs, peak memory 500 elements
```

**Result**: Option 1 is **3x faster** and uses **2.5x less memory**!

## Why It's Hard

Finding the optimal contraction order is **NP-complete**:

- With N tensors, there are approximately (2N)!/(N+1)! possible orders
- For 10 tensors: ~17 million orders
- For 20 tensors: ~6 × 10²² orders (more than atoms in your body!)

**Brute force is impossible** for realistic problems.

## Heuristic Approaches

Since optimal search is intractable, we use heuristics:

### 1. Greedy Algorithm

**Idea**: At each step, contract the pair with minimum cost.

**Pros**:
- Fast: O(n² log n)
- Deterministic
- Good for many practical cases

**Cons**:
- Can get stuck in local optima
- May miss better global solutions

### 2. Simulated Annealing (TreeSA)

**Idea**: Random tree mutations with temperature-based acceptance.

**Pros**:
- Explores more of the search space
- Often finds better solutions than greedy
- Configurable trade-off (time vs quality)

**Cons**:
- Slower than greedy
- Non-deterministic
- Requires parameter tuning

## Real-World Impact

### Quantum Circuit with 50 Qubits

- Random order: ~10⁵⁰ operations (impossible)
- Greedy order: ~10³⁰ operations (still impossible)
- Optimized order: ~10²⁰ operations (feasible on supercomputer)

**100,000,000,000x speedup** from optimization!

### Neural Network Attention

```python
# Naive: einsum("bqd,bkd,bkv->bqv", Q, K, V)
# Time: O(q²d + qkd + qkv)

# Optimized: Q @ K.T @ V
# Time: O(qkd + qkv)
# Speedup: ~10x for typical dimensions
```

## Optimization Objectives

Different scenarios need different trade-offs:

| Scenario | Optimize | Accept | Example |
|----------|----------|--------|---------|
| GPU limited | Space (sc) | Higher time | Large models on 8GB GPU |
| CPU compute | Time (tc) | Higher space | Batch processing with 256GB RAM |
| Both | Balanced | sc_target = available memory | Most cases |

## Practical Guidelines

1. **Start with greedy**: Fast baseline for most cases
2. **Use TreeSA if**:
   - Greedy result is too slow/large
   - You have time to optimize (minutes, not seconds)
   - Problem is large and complex

3. **Set sc_target** to available memory:
   ```python
   # For 8GB GPU with float32 tensors
   sc_target = log2(8 * 1024**3 / 4) ≈ 31
   ```

4. **Use slicing** when memory is the bottleneck:
   ```python
   if complexity.sc > sc_target:
       sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast())
   ```

## Next Steps

- [Complexity Metrics](./complexity-metrics.md) - Understand tc, sc, rwc
- [Greedy Method](../algorithms/greedy-method.md) - Fast optimization
- [TreeSA](../algorithms/tree-sa.md) - Higher quality optimization
