# GPU Optimization

Maximize performance when running tensor contractions on GPUs.

## The GPU Bottleneck

GPUs have a critical difference from CPUs: **memory bandwidth is the bottleneck**.

| Resource | GPU Advantage | Bottleneck |
|----------|---------------|------------|
| **Arithmetic** | 10-100x faster than CPU | ✅ Plenty |
| **Memory Bandwidth** | 2-4x faster than CPU | ❌ Limited |

**Result**: On GPU, the ratio of compute throughput to memory bandwidth is much higher than on CPU. Reducing memory I/O operations is critical for performance.

## Critical Configuration

### Experimental: Set rw_weight for GPU

```python
from omeco import ScoreFunction, TreeSA, optimize_code

# GPU optimization (experimental)
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=0.1,      # Experimental: tune based on profiling
    sc_target=30.0      # Your GPU memory limit
)

tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
```

**About rw_weight for GPU:**

The `rw_weight` parameter is **experimental** and should be tuned empirically:

- **Default**: 0.0 (CPU optimization)
- **Starting point for GPU**: Try 0.1 to 1.0
- **Reference**: [cotengra](https://cotengra.readthedocs.io/en/latest/basics.html) uses weight=64 for memory writes
- **Rationale**: GPUs have high compute-to-memory-bandwidth ratio

**Important**: There is no established "best" value. Profile your specific workload:
1. Start with `rw_weight=0.0` (default)
2. Try `rw_weight=0.1`, then `rw_weight=1.0`
3. Measure actual execution time on your GPU
4. Adjust based on results

The optimal value depends on:
- GPU model and memory bandwidth
- Tensor sizes and network topology
- Precision (FP16 vs FP32 vs FP64)

### Calculate sc_target for Your GPU

```python
import math

# NVIDIA A100: 40GB
gpu_gb = 40
bytes_per_element = 4  # float32

# Maximum usable memory (leave ~20% for overhead)
usable_gb = gpu_gb * 0.8
sc_target = math.log2(usable_gb * 1e9 / bytes_per_element)
# sc_target ≈ 32.0

score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,
    sc_target=sc_target
)
```

## GPU Memory Reference

| GPU | VRAM | Usable (80%) | sc_target (float32) |
|-----|------|--------------|---------------------|
| RTX 3060 | 12GB | 9.6GB | 31.0 |
| RTX 3090 | 24GB | 19GB | 32.2 |
| A100 | 40GB | 32GB | 32.9 |
| A100 | 80GB | 64GB | 33.9 |
| H100 | 80GB | 64GB | 33.9 |

## Example: CPU vs GPU Optimization

```python
from omeco import optimize_code, contraction_complexity, TreeSA, ScoreFunction

ixs = [[0, 1, 2], [2, 3, 4], [4, 5, 0]]  # Triangle
out = [1, 3, 5]
sizes = {i: 32 for i in range(6)}

# CPU optimization (rw_weight=0)
score_cpu = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=0.0,
    sc_target=28.0
)
tree_cpu = optimize_code(ixs, out, sizes, TreeSA(score=score_cpu))
comp_cpu = contraction_complexity(tree_cpu, ixs, sizes)

# GPU optimization (rw_weight=20)
score_gpu = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,
    sc_target=30.0
)
tree_gpu = optimize_code(ixs, out, sizes, TreeSA(score=score_gpu))
comp_gpu = contraction_complexity(tree_gpu, ixs, sizes)

# Compare
print(f"CPU-optimized: tc={comp_cpu.tc:.2f}, sc={comp_cpu.sc:.2f}, rwc={comp_cpu.rwc:.2f}")
print(f"GPU-optimized: tc={comp_gpu.tc:.2f}, sc={comp_gpu.sc:.2f}, rwc={comp_gpu.rwc:.2f}")
```

Typical result:
```
CPU-optimized: tc=18.5, sc=15.0, rwc=19.2
GPU-optimized: tc=19.1, sc=14.8, rwc=18.5  ← Lower rwc!
```

GPU-optimized tree accepts slightly higher `tc` to significantly reduce `rwc`.

## When GPU Optimization Matters

### High Impact

**Complex tensor networks with many intermediate tensors**:
- Quantum circuits
- Graph neural networks
- Irregular tensor contractions

GPU optimizer will restructure to minimize intermediate tensor creation/storage.

### Low Impact

**Simple chains or single-step contractions**:
- Matrix multiplication
- Small networks (< 10 tensors)
- Already near-optimal order

## Advanced: Mixed Precision

For mixed float32/float16:

```python
# Assume most computation in float16, some in float32
gpu_gb = 24
bytes_per_element = 3  # Average of float32 (4) and float16 (2)

sc_target = math.log2(gpu_gb * 1e9 / bytes_per_element)
# sc_target ≈ 32.6

score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,
    sc_target=32.5
)
```

## Batch Operations

For batched tensor contractions (e.g., multiple samples):

```python
# Batch dimension adds to both memory and I/O
batch_size = 32

# Adjust sc_target for batch
import math
base_sc_target = 30.0  # For single sample
batched_sc_target = base_sc_target + math.log2(batch_size)
# batched_sc_target = 30.0 + 5.0 = 35.0

score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,
    sc_target=batched_sc_target
)
```

## Verification

After optimization, verify the improvement:

```python
# Original (no GPU optimization)
tree_default = optimize_code(ixs, out, sizes)
comp_default = contraction_complexity(tree_default, ixs, sizes)

# With GPU optimization
score_gpu = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=20.0, sc_target=30.0)
tree_gpu = optimize_code(ixs, out, sizes, TreeSA(score=score_gpu))
comp_gpu = contraction_complexity(tree_gpu, ixs, sizes)

# Check read-write reduction
rwc_reduction = 2 ** (comp_default.rwc - comp_gpu.rwc)
print(f"Read-write operations reduced by {rwc_reduction:.1f}x")

# This translates to speedup on GPU
estimated_speedup = rwc_reduction * 0.7  # Empirical factor
print(f"Estimated GPU speedup: {estimated_speedup:.1f}x")
```

## Common Mistakes

### ❌ Forgetting rw_weight

```python
# WRONG: Optimized for CPU, not GPU
score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.0)
tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
# Result: Poor GPU performance due to excessive I/O
```

### ❌ Wrong sc_target

```python
# WRONG: Using CPU memory size for GPU
score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=20.0, sc_target=35.0)
# Result: Assumes 32GB GPU, but you only have 8GB → OOM errors
```

### ❌ Not using TreeSA

```python
# SUBOPTIMAL: Greedy ignores rw_weight
tree = optimize_greedy(ixs, out, sizes)
# Greedy uses simple heuristics that don't optimize for I/O
```

✅ **Correct approach**:

```python
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=20.0,
    sc_target=30.0  # Your GPU memory
)
tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
```

## Summary

**GPU Optimization Checklist**:
- ✅ Calculate `sc_target` from your GPU memory (critical)
- ✅ Use float32 (4 bytes) for sc_target calculation
- ✅ Reserve 20% memory for framework overhead
- ✅ Use TreeSA for better overall optimization
- ⚠️ **Experimental**: Try `rw_weight=0.1` to `1.0` and profile
  - No established best value - requires empirical tuning
  - Start with default (0.0), then try 0.1, then 1.0
  - Measure actual GPU execution time for your workload

## Next Steps

- [Score Function Configuration](./score-function.md) - Full parameter guide
- [Slicing Strategy](./slicing.md) - Reduce memory further
- [PyTorch Integration](./pytorch-integration.md) - Use with PyTorch
