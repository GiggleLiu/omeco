# Algorithms

omeco provides two main optimization algorithms with different speed-quality trade-offs.

## Algorithm Comparison

| Algorithm | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| **GreedyMethod** | Fast (seconds) | Good | Quick optimization, large networks |
| **TreeSA** | Slower (minutes) | Better | High-quality solutions, important workloads |

## Quick Guide

**Use GreedyMethod when:**
- You need results quickly
- Network has <100 tensors
- Greedy result is good enough

**Use TreeSA when:**
- You have time to optimize
- Need best possible solution
- Greedy result is too slow/large
- Working with complex tensor networks

## Topics

- [Greedy Method](./greedy-method.md) - Fast O(n² log n) optimization
- [TreeSA](./tree-sa.md) - Simulated annealing for quality
- [Algorithm Comparison](./comparison.md) - Detailed benchmarks

## Next Steps

Choose an algorithm to learn more, or see the [comparison](./comparison.md) for benchmarks.
