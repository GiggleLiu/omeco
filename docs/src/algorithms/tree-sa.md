# TreeSA

Simulated annealing optimizer for high-quality tensor contraction orders.

## How It Works

TreeSA uses **simulated annealing** to explore the space of contraction trees:

1. Start with a random or greedy tree
2. Repeatedly apply random mutations (subtree rotations, swaps)
3. Accept better mutations, sometimes accept worse ones (to escape local optima)
4. Gradually decrease "temperature" to converge

**Result**: Often finds significantly better orders than greedy.

## Basic Usage

### Fast Preset (Recommended)

```python
from omeco import optimize_code, TreeSA

# Quick optimization with sensible defaults
tree = optimize_code(ixs, out, sizes, TreeSA.fast())
```

### Custom Configuration

```python
from omeco import optimize_code, TreeSA, ScoreFunction

# More thorough search
score = ScoreFunction(sc_target=25.0)
optimizer = TreeSA(
    ntrials=10,      # Number of independent runs
    niters=50,       # Iterations per run
    score=score      # Custom scoring function
)
tree = optimize_code(ixs, out, sizes, optimizer)
```

### Rust

```rust
use omeco::{TreeSA, ScoreFunction, optimize_code};

let score = ScoreFunction::default();
let optimizer = TreeSA::fast(score);
let tree = optimize_code(&code, &sizes, &optimizer)?;
```

## Parameters

### `ntrials` (default: 10)

Number of independent optimization runs. The best result is returned.

- Higher = better quality (more exploration)
- Lower = faster
- Typical range: 5-20

### `niters` (default: 50)

Number of iterations per trial.

- Higher = better convergence
- Lower = faster
- Typical range: 20-100

### `betas` (optional)

Temperature schedule for simulated annealing.

- Default: logarithmic schedule from 1 to 40
- Custom: `[β₁, β₂, ..., βₙ]` where β = 1/temperature
- Higher β = less randomness (exploitation)
- Lower β = more randomness (exploration)

### `score` (optional)

Custom [ScoreFunction](../guides/score-function.md) for hardware-specific optimization.

```python
# GPU optimization (see GPU Optimization Guide)
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=10.0,
    sc_target=30.0
)
optimizer = TreeSA(ntrials=5, niters=50, score=score)
```

## Performance Characteristics

**Advantages**:
- 🏆 Higher quality solutions than greedy
- 🔍 Explores global search space
- ⚙️ Configurable time-quality trade-off
- 🎯 Often finds near-optimal solutions

**Limitations**:
- ⏱️ Slower than greedy (minutes vs seconds)
- 🎲 Non-deterministic (different runs give different results)
- 🔧 Requires parameter tuning for best results

## Example Comparison

```python
from omeco import optimize_code, GreedyMethod, TreeSA, 

# Same problem, different optimizers
ixs = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]]  # Cycle
out = [1, 3, 5, 7]
sizes = {i: 2 for i in range(8)}

# Greedy
tree_greedy = optimize_code(ixs, out, sizes, GreedyMethod())
comp_greedy = tree_greedy.complexity(ixs, sizes)

# TreeSA
tree_sa = optimize_code(ixs, out, sizes, TreeSA.fast())
comp_sa = tree_sa.complexity(ixs, sizes)

print(f"Greedy - tc: 2^{comp_greedy.tc:.2f}, sc: 2^{comp_greedy.sc:.2f}")
print(f"TreeSA - tc: 2^{comp_sa.tc:.2f}, sc: 2^{comp_sa.sc:.2f}")

# Speedup
speedup = 2 ** (comp_greedy.tc - comp_sa.tc)
print(f"TreeSA is {speedup:.1f}x faster")
```

Typical output:
```
Greedy - tc: 2^16.00, sc: 2^12.00
TreeSA - tc: 2^14.58, sc: 2^11.00
TreeSA is 2.7x faster
```

## When to Use

✅ **Use TreeSA when**:
- Greedy result is too slow/uses too much memory
- Network is complex (cycles, irregular graphs)
- Result will be used many times (worth optimizing once)
- You have minutes to spare for optimization

❌ **Stick with GreedyMethod when**:
- Need results in seconds
- Network is simple (chains, small grids)
- Iterating/prototyping
- Greedy is already good enough

## Optimization Workflow

```python
# 1. Quick baseline with greedy
tree_greedy = optimize_code(ixs, out, sizes, GreedyMethod())
comp_greedy = tree_greedy.complexity(ixs, sizes)
print(f"Greedy: tc={comp_greedy.tc:.2f}, sc={comp_greedy.sc:.2f}")

# 2. If not satisfactory, try TreeSA
if comp_greedy.sc > 28.0:  # Too much memory
    score = ScoreFunction(sc_target=28.0, sc_weight=2.0)
    tree_sa = optimize_code(ixs, out, sizes, TreeSA(ntrials=10, score=score))
    comp_sa = tree_sa.complexity(ixs, sizes)
    print(f"TreeSA: tc={comp_sa.tc:.2f}, sc={comp_sa.sc:.2f}")
```

## Tuning Parameters

### For Speed

```python
# Fast TreeSA (2-3x slower than greedy)
TreeSA(ntrials=2, niters=20)
```

### For Quality

```python
# Thorough TreeSA (10-20x slower than greedy)
TreeSA(ntrials=20, niters=100)
```

### For Memory-Constrained

```python
# Prioritize space over time
score = ScoreFunction(tc_weight=0.1, sc_weight=10.0, sc_target=25.0)
TreeSA(ntrials=10, niters=50, score=score)
```

## Advanced: Custom Temperature Schedule

```python
# Custom annealing schedule
import numpy as np

# Slow cooling for thorough search
betas = np.logspace(0, 2, 50)  # 1 to 100

optimizer = TreeSA(ntrials=5, betas=betas.tolist())
```

## Next Steps

- [Score Function Configuration](../guides/score-function.md) - Optimize for your hardware
- [Algorithm Comparison](./comparison.md) - Detailed benchmarks
- [Slicing Strategy](../guides/slicing.md) - Reduce memory further
