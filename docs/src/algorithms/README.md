# Algorithms

omeco provides four optimization algorithms with different speed-quality trade-offs.

## Algorithm Comparison

| Algorithm | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| **GreedyMethod** | Fast (seconds) | Good | Quick optimization, large networks |
| **ExhaustiveSearch** | Exponential | Optimal | Small networks, exact baselines |
| **TreeSA** | Slower (minutes) | Better | High-quality solutions, important workloads |
| **Treewidth** | Fast (ms) | Best on structured nets | Large graphical-model / relational networks |

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

**Use ExhaustiveSearch when:**
- The network is small enough for exact dynamic programming
- You need a known-optimal FLOP-count baseline
- You are testing or benchmarking heuristic optimizers

**Use Treewidth when:**
- The network is large and structured (graphical models, factor graphs, relational instances)
- A low-treewidth elimination order is likely to exist
- You need a fast, deterministic, high-quality order in milliseconds

## Topics

- [Greedy Method](./greedy-method.md) - Fast O(n² log n) optimization
- [Exhaustive Search](./exhaustive-search.md) - Exact dynamic programming for small networks
- [TreeSA](./tree-sa.md) - Simulated annealing for quality
- [Treewidth](./treewidth.md) - Scalable elimination-order heuristic
- [Algorithm Comparison](./comparison.md) - Detailed benchmarks

## Next Steps

Choose an algorithm to learn more, or see the [comparison](./comparison.md) for benchmarks.
