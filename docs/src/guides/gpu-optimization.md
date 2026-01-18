# GPU Optimization

Maximize performance when running tensor contractions on GPUs.

## The GPU Bottleneck

GPUs have 10-100x higher compute-to-memory-bandwidth ratio than CPUs. Memory I/O is the bottleneck.

## Configuration

```python
from omeco import ScoreFunction, TreeSA, optimize_code

# GPU optimization
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=10.0,     # Tune based on profiling (see below)
    sc_target=30.0      # Your GPU memory limit
)

tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
```

**Tuning rw_weight:**

No established "best" value—tune empirically:
1. Start with `rw_weight=1.0`
2. Try `10.0`, then `20.0` or higher
3. Profile actual GPU execution time
4. Adjust based on results

Range: 0.1 to >20 depending on GPU model, tensor sizes, and precision.

### Calculate sc_target for Your GPU

```python
import math

# NVIDIA A100: 40GB
gpu_gb = 40
bytes_per_element = 4  # float32

# Maximum usable memory (leave ~20% for overhead)
usable_gb = gpu_gb * 0.8
sc_target = math.log2(usable_gb * 1024**3 / bytes_per_element)
# sc_target ≈ 32.0

score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=10.0,
    sc_target=sc_target
)
```

## GPU Memory Reference

| GPU | VRAM | sc_target (float32) |
|-----|------|---------------------|
| RTX 3060 | 12GB | 31.0 |
| RTX 3090 | 24GB | 32.2 |
| A100 (40GB) | 40GB | 32.9 |
| A100 (80GB) | 80GB | 33.9 |
| H100 | 80GB | 33.9 |

## When It Matters

**High impact**: Complex networks (quantum circuits, GNNs, irregular contractions)
**Low impact**: Simple chains, matrix multiplications, small networks (<10 tensors)

## Common Mistakes

- ❌ Using `rw_weight=0.0` (CPU default) on GPU
- ❌ Wrong `sc_target` (e.g., using CPU memory size for GPU)
- ❌ Using GreedyMethod instead of TreeSA (greedy doesn't optimize for I/O)

## Checklist

1. Calculate `sc_target` from GPU memory (float32: 4 bytes/element, reserve 20% overhead)
2. Set `rw_weight` (start with 1.0, try 10.0 and 20.0, profile)
3. Use TreeSA optimizer (not greedy)

## Next Steps

- [Score Function Configuration](./score-function.md) - Full parameter guide
- [Slicing Strategy](./slicing.md) - Reduce memory further
- [PyTorch Integration](./pytorch-integration.md) - Use with PyTorch
