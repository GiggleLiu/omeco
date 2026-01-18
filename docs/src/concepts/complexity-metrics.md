# Complexity Metrics

omeco tracks three key metrics for tensor contractions.

## The Three Metrics

### Time Complexity (tc)

**Definition**: log₂ of total floating-point operations (FLOPs).

**Interpretation**:
- `tc = 20` means 2²⁰ = 1,048,576 FLOPs
- `tc = 30` means 2³⁰ ≈ 1 billion FLOPs  
- Lower is better (less computation)

**Example**:
```python
complexity = contraction_complexity(tree, ixs, sizes)
print(f"Time: 2^{complexity.tc:.2f} FLOPs")
print(f"Actual FLOPs: {complexity.flops():,.0f}")
```

### Space Complexity (sc)

**Definition**: log₂ of peak memory usage in number of tensor elements.

**Interpretation**:
- `sc = 20` means 2²⁰ = 1,048,576 elements (8MB for float64, 4MB for float32)
- `sc = 30` means 2³⁰ ≈ 1 billion elements (8GB for float64, 4GB for float32)
- Lower is better (less memory)

**Example**:
```python
complexity = contraction_complexity(tree, ixs, sizes)
print(f"Space: 2^{complexity.sc:.2f} elements")
print(f"Peak memory: {complexity.peak_memory():,.0f} elements")

# For float64 (8 bytes)
memory_gb = complexity.peak_memory() * 8 / 1e9
print(f"Memory: {memory_gb:.2f} GB")
```

### Read-Write Complexity (rwc)

**Definition**: log₂ of total memory I/O operations.

**Use Case**: GPU optimization where memory bandwidth is the bottleneck.

**Interpretation**:
- On CPU: Usually ignored (rw_weight=0)
- On GPU: Critical! Compute-to-memory-bandwidth ratio is high (10-30x)
- Lower is better (less data movement)

**Example**:
```python
# CPU optimization (ignore I/O)
score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.0)

# GPU optimization (I/O matters! See GPU Optimization Guide)
score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=10.0)
```

## Why Logarithmic Scale?

Using log₂ makes the numbers manageable:

| Linear Scale | Log Scale | Human Readable |
|--------------|-----------|----------------|
| 1,024 | 10 | ~thousand |
| 1,048,576 | 20 | ~million |
| 1,073,741,824 | 30 | ~billion |
| 1,099,511,627,776 | 40 | ~trillion |

**Advantages**:
- Easy to compare: `tc=25` vs `tc=30` is clear 32x difference
- Handles huge ranges: from thousands to quintillions
- Matches memory sizes: 2ⁿ bytes

## Calculating Complexity

### For a Single Contraction

Contracting `A[i,j]` with `B[j,k]` → `C[i,k]`:

```
Time: i × j × k FLOPs
Space: max(i×j, j×k, i×k) elements
```

### For a Contraction Tree

Sum time over all contractions, take max space:

```python
def compute_complexity(tree):
    if tree.is_leaf():
        return Complexity(tc=0, sc=log2(tensor_size))
    
    # Recurse on children
    left_comp = compute_complexity(tree.left)
    right_comp = compute_complexity(tree.right)
    
    # Contraction cost
    contraction_tc = log2(flops_for_this_step)
    contraction_sc = log2(intermediate_size)
    
    # Combine
    tc = log2sumexp2([left_comp.tc, right_comp.tc, contraction_tc])
    sc = max(left_comp.sc, right_comp.sc, contraction_sc)
    
    return Complexity(tc=tc, sc=sc)
```

## Practical Interpretation

### Time Complexity

| tc | FLOPs | CPU Time (1 TFLOP/s) | GPU Time (10 TFLOP/s) |
|----|-------|----------------------|-----------------------|
| 20 | 1M | 0.001 ms | 0.0001 ms |
| 30 | 1B | 1 ms | 0.1 ms |
| 40 | 1T | 1 second | 0.1 second |
| 50 | 1P | 17 minutes | 1.7 minutes |

### Space Complexity

| sc | Elements | float32 Memory | float64 Memory |
|----|----------|----------------|----------------|
| 20 | 1M | 4 MB | 8 MB |
| 25 | 32M | 128 MB | 256 MB |
| 30 | 1B | 4 GB | 8 GB |
| 35 | 32B | 128 GB | 256 GB |

## Optimization Goals

### Minimize Time (Speed-Critical)

```python
score = ScoreFunction(tc_weight=1.0, sc_weight=0.0, sc_target=float('inf'))
```

Use when:
- Memory is abundant
- Need fastest execution
- Real-time applications

### Minimize Space (Memory-Limited)

```python
score = ScoreFunction(tc_weight=0.0, sc_weight=1.0, sc_target=25.0)
```

Use when:
- Memory is constrained (e.g., 8GB GPU)
- Can afford longer computation
- Embedded systems

### Balanced (Most Common)

```python
score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, sc_target=28.0)
```

Use when:
- Both time and space matter
- Setting sc_target to available memory
- General-purpose optimization

## Example: Comparing Orders

```python
# Greedy result
tree_greedy = optimize_greedy(ixs, out, sizes)
comp_greedy = contraction_complexity(tree_greedy, ixs, sizes)

# TreeSA result
tree_sa = optimize_treesa(ixs, out, sizes, TreeSA.fast())
comp_sa = contraction_complexity(tree_sa, ixs, sizes)

# Compare
print(f"Greedy - Time: 2^{comp_greedy.tc:.2f}, Space: 2^{comp_greedy.sc:.2f}")
print(f"TreeSA - Time: 2^{comp_sa.tc:.2f}, Space: 2^{comp_sa.sc:.2f}")

# Speedup
speedup = 2 ** (comp_greedy.tc - comp_sa.tc)
print(f"TreeSA is {speedup:.1f}x faster")
```

## Next Steps

- [Score Function Configuration](../guides/score-function.md) - Optimize for your hardware
- [GPU Optimization](../guides/gpu-optimization.md) - Maximize GPU performance
- [Performance Benchmarks](../performance.md) - Real-world performance data
