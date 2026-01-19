# Slicing Strategy

Reduce memory usage by slicing tensor indices, trading time for space.

## What is Slicing?

**Slicing** breaks a tensor contraction into smaller chunks by looping over one or more indices.

**Example**:
```python
# Original: C[i,k] = Σⱼ A[i,j] * B[j,k]
# Too large: j has size 10000

# Sliced: Loop over j in chunks
C = zeros(i, k)
for j_val in range(10000):
    C += A[:, j_val] * B[j_val, :]
```

**Trade-off**:
- ✅ Memory reduced: Only store one slice at a time
- ❌ Time increased: Repeat computation for each slice

## When to Use Slicing

Use slicing when:
- Contraction exceeds available memory
- `sc` (space complexity) is too high
- Can't reduce memory by changing contraction order alone

**Decision flowchart**:
```
Optimize contraction order
    ↓
Check space complexity
    ↓
sc > sc_target?
  ├─ No  → Done
  └─ Yes → Apply slicing
```

## Basic Usage

### Automatic Slicing

```python
from omeco import optimize_code, slice_code, TreeSASlicer, ScoreFunction, 

# 1. Optimize contraction order
tree = optimize_code(ixs, out, sizes)

# 2. Check if memory is too large
complexity = tree.complexity(ixs, sizes)
if complexity.sc > 28.0:  # More than 256MB
    # 3. Slice to reduce memory
    score = ScoreFunction(sc_target=28.0)
    slicer = TreeSASlicer.fast(score=score)
    sliced = slice_code(tree, ixs, sizes, slicer)
    
    # 4. Verify reduction
    sliced_comp = sliced.complexity(ixs, sizes)
    print(f"Original: 2^{complexity.sc:.2f} elements")
    print(f"Sliced: 2^{sliced_comp.sc:.2f} elements")
    print(f"Sliced indices: {sliced.slicing()}")
```

### Manual Slicing

Specify which indices to slice:

```python
# Force slicing on specific indices
slicer = TreeSASlicer.fast(fixed_slices=[1, 2])
sliced = slice_code(tree, ixs, sizes, slicer)

# Verify that indices 1 and 2 are sliced
assert 1 in sliced.slicing()
assert 2 in sliced.slicing()
```

## TreeSASlicer Configuration

### Fast Preset

```python
# Quick slicing optimization
slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=25.0))
sliced = slice_code(tree, ixs, sizes, slicer)
```

### Custom Configuration

```python
from omeco import TreeSASlicer, ScoreFunction

score = ScoreFunction(
    tc_weight=0.1,      # Accept higher time
    sc_weight=10.0,     # Prioritize space
    sc_target=25.0      # Target 256MB
)

slicer = TreeSASlicer(
    ntrials=10,          # Number of optimization runs
    niters=50,           # Iterations per run
    fixed_slices=[1],    # Force slicing on index 1
    score=score
)

sliced = slice_code(tree, ixs, sizes, slicer)
```

## Understanding the Trade-off

### Example: Large Matrix Multiplication

```python
# A[10000 × 10000] × B[10000 × 10000]
ixs = [[0, 1], [1, 2]]
out = [0, 2]
sizes = {0: 10000, 1: 10000, 2: 10000}

# Without slicing
tree = optimize_code(ixs, out, sizes)
comp = tree.complexity(ixs, sizes)
# tc ≈ 33.2 (10^10 FLOPs)
# sc ≈ 26.6 (100M elements = 800MB for float64)

# With slicing on index 1 (middle dimension)
slicer = TreeSASlicer.fast(fixed_slices=[1])
sliced = slice_code(tree, ixs, sizes, slicer)
sliced_comp = sliced.complexity(ixs, sizes)
# tc ≈ 33.2 (same FLOPs, distributed over slices)
# sc ≈ 23.3 (10M elements = 80MB for float64)

# Result: 10x memory reduction, same total time (but distributed)
```

## Choosing Which Indices to Slice

### Automatic Selection

TreeSASlicer automatically chooses which indices to slice:

```python
# Let the optimizer decide
slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=25.0))
sliced = slice_code(tree, ixs, sizes, slicer)

print(f"Automatically sliced indices: {sliced.slicing()}")
```

**Algorithm**: Tries slicing different indices and picks the combination that best satisfies `sc_target` while minimizing `tc` increase.

### Manual Selection Guidelines

**Good candidates for slicing**:
- Indices with large dimensions
- Indices that appear in many tensors (high degree)
- Indices not in the output (no outer loop needed)

**Poor candidates**:
- Output indices (can't slice without changing result structure)
- Small dimensions (little memory benefit)
- Indices that would cause many small intermediate tensors

## Advanced: Slicing with GPU

For GPU workloads, consider both memory and I/O:

```python
# GPU slicing: balance memory and I/O
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=2.0,      # Space is critical on GPU
    rw_weight=0.1,      # Experimental: tune empirically
    sc_target=30.0      # 8GB GPU limit
)

slicer = TreeSASlicer(
    ntrials=10,
    niters=50,
    score=score
)

sliced = slice_code(tree, ixs, sizes, slicer)
```

**Note**: Slicing may increase `rwc` (read-write complexity) because data is loaded multiple times. The optimizer balances this with the `rw_weight` parameter.

## Example Workflow

```python
from omeco import optimize_code, slice_code, 
from omeco import TreeSA, TreeSASlicer, ScoreFunction

# Problem setup
ixs = [[0,1,2], [2,3,4], [4,5,6], [6,7,0]]  # Cycle graph
out = [1,3,5,7]
sizes = {i: 32 for i in range(8)}

# Step 1: Optimize contraction order
tree = optimize_code(ixs, out, sizes, TreeSA.fast())
comp = tree.complexity(ixs, sizes)
print(f"Optimized: sc = 2^{comp.sc:.2f}")

# Step 2: Check if slicing needed
if comp.sc > 20.0:  # Exceeds 1MB
    print("Applying slicing...")
    
    # Step 3: Slice
    slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=18.0))
    sliced = slice_code(tree, ixs, sizes, slicer)
    
    # Step 4: Verify
    sliced_comp = sliced.complexity(ixs, sizes)
    print(f"Sliced: sc = 2^{sliced_comp.sc:.2f}")
    print(f"Sliced indices: {sliced.slicing()}")
    
    # Calculate overhead
    time_overhead = 2 ** (sliced_comp.tc - comp.tc)
    memory_reduction = 2 ** (comp.sc - sliced_comp.sc)
    print(f"Time overhead: {time_overhead:.1f}x")
    print(f"Memory reduction: {memory_reduction:.1f}x")
```

## Multiple Slices

Slice multiple indices for extreme memory reduction:

```python
# Very tight memory constraint
score = ScoreFunction(
    tc_weight=0.01,     # Accept much higher time
    sc_weight=100.0,    # Memory is critical
    sc_target=15.0      # Only 32KB available!
)

slicer = TreeSASlicer(ntrials=20, niters=100, score=score)
sliced = slice_code(tree, ixs, sizes, slicer)

# May slice multiple indices
print(f"Sliced {len(sliced.slicing())} indices: {sliced.slicing()}")
```

## Performance Impact

| Scenario | Memory Reduction | Time Overhead | When to Use |
|----------|------------------|---------------|-------------|
| Slice 1 index | 2-10x | 1-2x | Moderate memory constraint |
| Slice 2 indices | 10-100x | 2-5x | Severe memory constraint |
| Slice 3+ indices | 100-1000x | 5-20x | Extreme memory constraint |

## Tips

1. **Optimize order first, slice second**:
   ```python
   # Good workflow
   tree = optimize_code(ixs, out, sizes, TreeSA.fast())
   sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast())
   ```

2. **Set realistic sc_target**:
   ```python
   # Target 80% of available memory
   available_gb = 8
   sc_target = math.log2(available_gb * 0.8 * 1024**3 / 8)
   ```

3. **Profile actual memory usage**:
   ```python
   # Theoretical vs actual memory may differ
   # Always test with real data
   ```

4. **Consider time-memory trade-off**:
   ```python
   # For one-time computation: accept high memory
   # For repeated computation: slice to fit memory
   ```

## Next Steps

- [Score Function Configuration](./score-function.md) - Tune slicing parameters
- [GPU Optimization](./gpu-optimization.md) - GPU-specific slicing
- [Complexity Metrics](../concepts/complexity-metrics.md) - Understand sc and tc
