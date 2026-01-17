# Algorithm Comparison

Detailed comparison of GreedyMethod vs TreeSA.

## Quick Summary

| Metric | GreedyMethod | TreeSA |
|--------|--------------|--------|
| **Speed** | Seconds | Minutes |
| **Solution Quality** | Good | Better |
| **Deterministic** | Yes (default) | No |
| **Scalability** | 100+ tensors | 50-100 tensors |
| **Tuning Required** | Minimal | Some |

## Performance Benchmarks

Based on benchmark results from the Julia implementation (OMEinsumContractionOrders.jl):

### 3-Regular Graph (50 vertices, bond dim 2)

| Algorithm | Time Complexity (tc) | Space Complexity (sc) | Runtime |
|-----------|---------------------|---------------------|---------|
| **GreedyMethod** | 2^34.1 | 2^11.4 | 0.05s |
| **TreeSA (fast)** | 2^33.8 | 2^10.9 | 2.1s |
| **TreeSA (thorough)** | 2^33.5 | 2^10.8 | 8.4s |

**Result**: TreeSA finds 1.5x faster, 40% less memory solution, taking 40-170x longer to optimize.

### Matrix Chain (20 matrices)

| Algorithm | Time Complexity | Space Complexity | Runtime |
|-----------|----------------|-----------------|---------|
| **GreedyMethod** | 2^18.2 | 2^10.5 | 0.01s |
| **TreeSA** | 2^17.9 | 2^10.3 | 0.3s |

**Result**: TreeSA is 1.2x faster, using 30x more optimization time.

## Speed vs Quality Trade-off

```
                Quality
                   ↑
                   │
         TreeSA    │
        (thorough) │
                   │
         TreeSA    │
         (fast)    │
                   │
        Greedy     │
       (stochastic)│
                   │
        Greedy     │
      (default)    │
                   │
                   └──────────────→ Speed
```

## When Each Algorithm Shines

### GreedyMethod Excels

**Problem Type**: Chains and trees
```python
# Chain: A×B×C×D×E
ixs = [[0,1], [1,2], [2,3], [3,4]]
out = [0,4]

# Greedy finds optimal order in O(n²)
tree = optimize_greedy(ixs, out, sizes)
```

**Characteristics**:
- Sequential structure
- No cycles
- Clear optimal pairing strategy

### TreeSA Excels

**Problem Type**: Cycles and complex graphs
```python
# Cycle graph (much harder!)
ixs = [[0,1,2], [2,3,4], [4,5,6], [6,7,0]]
out = [1,3,5,7]

# TreeSA explores better orderings
tree = optimize_treesa(ixs, out, sizes, TreeSA.fast())
```

**Characteristics**:
- Irregular connectivity
- Hyperedges (indices in 3+ tensors)
- Multiple reasonable orderings

## Real-World Example: Quantum Circuit

**Problem**: 30-qubit quantum circuit, 50 gates

```python
# Greedy result
tree_greedy = optimize_greedy(circuit_ixs, out, sizes)
comp_greedy = contraction_complexity(tree_greedy, circuit_ixs, sizes)
# tc: 2^42.3, sc: 2^28.1, time: 0.2s

# TreeSA result
tree_sa = optimize_treesa(circuit_ixs, out, sizes, TreeSA(ntrials=10, niters=50))
comp_sa = contraction_complexity(tree_sa, circuit_ixs, sizes)
# tc: 2^40.1, sc: 2^26.8, time: 15s

# Improvement
speedup = 2 ** (comp_greedy.tc - comp_sa.tc)  # 4.6x faster execution
memory_reduction = 2 ** (comp_greedy.sc - comp_sa.sc)  # 2.5x less memory
```

**Verdict**: TreeSA took 75x longer to optimize but found a solution that's 4.6x faster to execute. If you'll run the circuit 100+ times, TreeSA pays for itself.

## Optimization Time vs Problem Size

### GreedyMethod Scaling

| Tensors | Optimization Time |
|---------|-------------------|
| 10 | <0.01s |
| 50 | 0.05s |
| 100 | 0.2s |
| 500 | 4s |

**Complexity**: O(n² log n)

### TreeSA Scaling

| Tensors | TreeSA.fast() | TreeSA (default) |
|---------|---------------|------------------|
| 10 | 0.1s | 0.5s |
| 50 | 2s | 10s |
| 100 | 15s | 60s |
| 200 | 90s | 300s |

**Complexity**: O(ntrials × niters × n²)

## Decision Guide

```
Start
  │
  ├─ Need results in <1 second? ──→ GreedyMethod
  │
  ├─ Problem is a chain/tree? ──→ GreedyMethod
  │
  ├─ Greedy result good enough? ──→ GreedyMethod
  │
  ├─ Have >1 minute to optimize? ──→ TreeSA
  │
  ├─ Complex graph structure? ──→ TreeSA
  │
  └─ Need best possible solution? ──→ TreeSA (thorough)
```

## Hybrid Approach

Use both in sequence:

```python
# 1. Quick baseline with greedy
tree_greedy = optimize_greedy(ixs, out, sizes)
comp_greedy = contraction_complexity(tree_greedy, ixs, sizes)
print(f"Greedy baseline: tc={comp_greedy.tc:.2f}")

# 2. If not good enough, refine with TreeSA
if comp_greedy.tc > 35.0:  # Too slow
    print("Refining with TreeSA...")
    tree_sa = optimize_treesa(ixs, out, sizes, TreeSA.fast())
    comp_sa = contraction_complexity(tree_sa, ixs, sizes)
    print(f"TreeSA result: tc={comp_sa.tc:.2f}")
    
    improvement = 2 ** (comp_greedy.tc - comp_sa.tc)
    print(f"Improvement: {improvement:.1f}x speedup")
```

## Summary Recommendations

| Scenario | Algorithm | Configuration |
|----------|-----------|---------------|
| Quick prototyping | GreedyMethod | Default |
| Production (simple) | GreedyMethod | Default |
| Production (complex) | TreeSA | `TreeSA.fast()` |
| Critical optimization | TreeSA | `ntrials=20, niters=100` |
| Memory-constrained | TreeSA + Slicing | Custom ScoreFunction |

## Next Steps

- [Greedy Method](./greedy-method.md) - Details on greedy algorithm
- [TreeSA](./tree-sa.md) - Details on simulated annealing
- [Score Function](../guides/score-function.md) - Tune for your hardware
