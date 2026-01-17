# Score Function Configuration

Configure omeco's optimization for different hardware and use cases.

## What is the Score Function?

The `ScoreFunction` controls how the optimizer balances three objectives:

- **tc (time complexity)**: Total FLOPs
- **sc (space complexity)**: Peak memory usage
- **rwc (read-write complexity)**: Memory I/O operations

**Formula**:
```
score = tc_weight × 2^tc + rw_weight × 2^rwc + sc_weight × max(0, 2^sc - 2^sc_target)
```

**Lower score is better.** The optimizer tries to minimize this score.

## Basic Usage

### Default (Balanced)

```python
from omeco import ScoreFunction

# Balanced optimization (works for most cases)
score = ScoreFunction()
# Equivalent to:
# ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.0, sc_target=20.0)
```

### Custom Weights

```python
# Prioritize memory over time
score = ScoreFunction(
    tc_weight=1.0,      # Time matters
    sc_weight=2.0,      # Memory matters 2x more
    rw_weight=0.0,      # Ignore I/O (CPU)
    sc_target=25.0      # Target 2^25 elements (~256MB for float64)
)
```

### Using with Optimizers

```python
from omeco import TreeSA, optimize_code

score = ScoreFunction(sc_target=28.0)
tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
```

## Hardware-Specific Configuration

### CPU Optimization

**Characteristics**:
- Memory bandwidth is not the bottleneck
- Balance time and space
- Ignore read-write complexity

```python
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=0.0,      # ← CPU: I/O is not a bottleneck
    sc_target=28.0      # ~256MB (2^28 × 8 bytes for float64)
)
```

**Calculate sc_target**:
```python
import math

# For 16GB RAM, reserve half for tensors
available_gb = 8
bytes_per_element = 8  # float64
sc_target = math.log2(available_gb * 1e9 / bytes_per_element)
# sc_target ≈ 30.0 (8GB)
```

### GPU Optimization

**Critical**: On GPUs, memory I/O is **20x slower** than arithmetic!

```python
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,     # ⚠️ KEY FOR GPU!
    sc_target=30.0      # ~8GB GPU (2^30 × 4 bytes for float32)
)
```

**Why rw_weight=20?**
- GPU arithmetic: ~10 TFLOP/s
- GPU memory bandwidth: ~500 GB/s = ~0.5 TFLOP/s equivalent
- Ratio: 10/0.5 = **20x**

**Calculate sc_target for GPU**:
```python
import math

# NVIDIA RTX 3090: 24GB VRAM
gpu_gb = 24
bytes_per_element = 4  # float32 (most common on GPU)
sc_target = math.log2(gpu_gb * 1e9 / bytes_per_element)
# sc_target ≈ 32.5

# Be conservative (leave room for framework overhead)
sc_target = 32.0  # ~16GB usable
```

## Common Scenarios

| Use Case | tc_weight | sc_weight | rw_weight | sc_target | Notes |
|----------|-----------|-----------|-----------|-----------|-------|
| **CPU (balanced)** | 1.0 | 1.0 | 0.0 | 28.0 | Default works well |
| **GPU** | 1.0 | 1.0 | **20.0** | 30.0 | **Must set rw_weight=20** |
| **Low memory** | 1.0 | 3.0 | 0.0 | 25.0 | Penalize memory 3x |
| **Speed critical** | 1.0 | 0.0 | 0.0 | ∞ | Ignore memory |
| **Embedded** | 0.1 | 10.0 | 0.0 | 20.0 | Memory is scarce |

## Advanced Configuration

### Memory-Constrained Optimization

When memory is the primary constraint:

```python
# Heavily penalize exceeding target
score = ScoreFunction(
    tc_weight=0.1,      # Time is secondary
    sc_weight=10.0,     # Memory is critical
    rw_weight=0.0,
    sc_target=25.0      # Hard limit: 256MB
)

# Use with TreeSA for best results
optimizer = TreeSA(ntrials=10, niters=50, score=score)
tree = optimize_code(ixs, out, sizes, optimizer)
```

### Speed-Critical Optimization

When execution speed is paramount:

```python
# Minimize time complexity only
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=0.0,      # Ignore memory
    rw_weight=0.0,
    sc_target=float('inf')  # No memory limit
)
```

### Hybrid CPU+GPU

For heterogeneous systems:

```python
# Moderate I/O penalty (between CPU and GPU)
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=10.0,     # Moderate I/O cost
    sc_target=29.0      # 4GB limit
)
```

## sc_target Reference

| Memory | float32 | float64 | sc_target |
|--------|---------|---------|-----------|
| 256 MB | 64M elements | 32M elements | 25.0-26.0 |
| 1 GB | 256M | 128M | 27.0-28.0 |
| 4 GB | 1B | 512M | 29.0-30.0 |
| 8 GB | 2B | 1B | 30.0-31.0 |
| 16 GB | 4B | 2B | 31.0-32.0 |
| 32 GB | 8B | 4B | 32.0-33.0 |

## Tuning Workflow

1. **Start with defaults**:
   ```python
   tree = optimize_code(ixs, out, sizes)
   complexity = contraction_complexity(tree, ixs, sizes)
   print(f"tc: {complexity.tc:.2f}, sc: {complexity.sc:.2f}")
   ```

2. **Identify bottleneck**:
   - If `sc` too high → increase `sc_weight` or lower `sc_target`
   - If `tc` too high → try TreeSA with default score
   - If running on GPU → set `rw_weight=20.0`

3. **Adjust and re-optimize**:
   ```python
   score = ScoreFunction(sc_weight=2.0, sc_target=28.0)
   tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
   ```

4. **Verify improvement**:
   ```python
   new_complexity = contraction_complexity(tree, ixs, sizes)
   print(f"New tc: {new_complexity.tc:.2f}, sc: {new_complexity.sc:.2f}")
   ```

## Examples

See [examples/score_function_examples.py](https://github.com/GiggleLiu/omeco/blob/master/examples/score_function_examples.py) for complete examples:
- CPU optimization
- GPU optimization with rw_weight=20
- Memory-limited environments
- Dynamic sc_target calculation

## Next Steps

- [GPU Optimization](./gpu-optimization.md) - Maximize GPU performance
- [Slicing Strategy](./slicing.md) - Reduce memory with slicing
- [Complexity Metrics](../concepts/complexity-metrics.md) - Understand tc, sc, rwc
