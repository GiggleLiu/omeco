# Score Function Configuration Guide

Quick guide for configuring `ScoreFunction` to optimize tensor contractions for different hardware.

## Quick Start

The `ScoreFunction` balances three objectives:
- **tc** (time): Total operations (FLOPs)
- **sc** (space): Peak memory usage
- **rwc** (read-write): Memory I/O operations

**Formula:** `score = tc_weight × 2^tc + rw_weight × 2^rwc + sc_weight × max(0, 2^sc - 2^sc_target)`

**Lower is better.**

## Basic Usage

```python
from omeco import ScoreFunction, TreeSA, optimize_code

# Default (balanced)
score = ScoreFunction()

# Custom weights
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=2.0,
    rw_weight=0.0,
    sc_target=25.0
)

# Presets
score = ScoreFunction.time_optimized()  # Speed only
score = ScoreFunction.space_optimized(25.0)  # Memory only
```

## Hardware-Specific Settings

### CPU

```python
# Balanced (default is fine)
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=0.0,      # CPU: I/O not a bottleneck
    sc_target=28.0      # ~256MB (2^28 × 8 bytes)
)
```

### GPU

**Key:** On GPUs, memory I/O is **20x slower** than arithmetic. Set `rw_weight=20.0`.

```python
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,     # ⚠️ Critical for GPU!
    sc_target=30.0      # ~8GB GPU (2^30 × 4 bytes for float32)
)
```

**Calculate sc_target for your GPU:**
```python
import math
gpu_gb = 8  # Your GPU memory
sc_target = math.log2(gpu_gb * 1e9 / 4)  # For float32
```

## Slicing (When Memory is Tight)

Use slicing when the contraction needs more memory than available. Trades time for space.

```python
from omeco import slice_code, TreeSASlicer, contraction_complexity

# Check memory usage
complexity = contraction_complexity(optimized, sizes, code.ixs)
print(f"Space: 2^{complexity.sc:.2f}")

# If too large, slice
if complexity.sc > 28.0:
    slicer = TreeSASlicer.default()
    sliced = slice_code(optimized, sizes, slicer, code.ixs)
```

**For severe memory constraints:**
```python
score = ScoreFunction(
    tc_weight=0.1,
    sc_weight=10.0,     # Heavily penalize memory
    rw_weight=0.0,
    sc_target=25.0
)
slicer = TreeSASlicer(ntrials=4, niters=50, score=score)
```

## Common Scenarios

| Use Case | tc_weight | sc_weight | rw_weight | sc_target | Notes |
|----------|-----------|-----------|-----------|-----------|-------|
| CPU (balanced) | 1.0 | 1.0 | 0.0 | 28.0 | Default works well |
| GPU | 1.0 | 1.0 | **20.0** | 30.0 | **Key: Set rw_weight=20** |
| Low memory | 1.0 | 3.0 | 0.0 | 25.0 | Penalize memory |
| Latency-critical | 1.0 | 0.0 | 0.0 | ∞ | Speed only |
| Quantum simulation | 0.1 | 5.0 | 2.0 | 35.0 | Large tc, limit memory |

## Key Points

1. **For GPU: Always set `rw_weight=20.0`** - Memory I/O is 20x slower than arithmetic
2. **For low memory: Increase `sc_weight` or use slicing**
3. **Calculate `sc_target`:** `log2(available_GB × 1e9 / bytes_per_element)`
4. **Profile first, tune later** - Measure actual performance

## See Also

- [Examples](../examples/score_function_examples.py)
- [API Documentation](https://docs.rs/omeco)
