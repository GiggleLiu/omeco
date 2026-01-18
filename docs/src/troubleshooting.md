# Troubleshooting

Common issues and solutions when using omeco.

## Installation Issues

### Python: pip install fails

**Problem**: `pip install omeco` fails with compilation errors.

**Solutions**:

1. **Ensure Rust toolchain is installed**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update
   ```

2. **Update pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Install from source**:
   ```bash
   git clone https://github.com/GiggleLiu/omeco
   cd omeco/omeco-python
   pip install maturin
   maturin develop --release
   ```

### Rust: Dependency resolution errors

**Problem**: Cargo fails to resolve dependencies.

**Solution**: Update Cargo.lock and rebuild:
```bash
cargo clean
cargo update
cargo build --release
```

## Runtime Errors

### ImportError: No module named 'omeco'

**Problem**: Python cannot find the omeco module after installation.

**Diagnosis**:
```python
import sys
print(sys.path)  # Check if package is in path
```

**Solutions**:

1. **Verify installation**:
   ```bash
   pip show omeco
   ```

2. **Check virtual environment**:
   ```bash
   which python  # Ensure you're using the right Python
   ```

3. **Reinstall in development mode**:
   ```bash
   cd omeco-python
   maturin develop --release
   ```

### Memory Errors (OOM)

**Problem**: Program crashes with "Out of Memory" error during optimization.

**Cause**: Space complexity (`sc`) exceeds available memory.

**Solutions**:

1. **Check space complexity**:
   ```python
   from omeco import optimize_code, contraction_complexity
   
   tree = optimize_code(ixs, out, sizes)
   comp = contraction_complexity(tree, ixs, sizes)
   print(f"Space complexity: 2^{comp.sc:.2f} elements")

   # Calculate memory usage
   bytes_per_element = 8  # float64
   memory_gib = (2 ** comp.sc) * bytes_per_element / 1024**3
   print(f"Estimated memory: {memory_gib:.2f} GiB")
   ```

2. **Use slicing to reduce memory**:
   ```python
   from omeco import slice_code, TreeSASlicer, ScoreFunction

   # Set target memory (e.g., 8GB = sc_target ≈ 30.0 for float64)
   import math
   available_gb = 8
   sc_target = math.log2(available_gb * 1024**3 / 8)

   score = ScoreFunction(sc_target=sc_target, sc_weight=2.0)
   slicer = TreeSASlicer.fast(score=score)
   sliced = slice_code(tree, ixs, sizes, slicer)
   ```

3. **Optimize with memory constraints from the start**:
   ```python
   from omeco import TreeSA, ScoreFunction
   
   score = ScoreFunction(sc_target=28.0, sc_weight=3.0)
   tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
   ```

### TypeError: Invalid index type

**Problem**: Indices must be integers, but got strings or floats.

**Cause**: Einstein indices in `ixs` and `out` must be integers (or strings in some contexts).

**Solution**: Convert indices to integers:
```python
# Wrong
ixs = [["a", "b"], ["b", "c"]]

# Correct
ixs = [[0, 1], [1, 2]]
out = [0, 2]
sizes = {0: 10, 1: 20, 2: 10}
```

## Performance Issues

### Optimization is too slow

**Problem**: TreeSA takes too long to find a good contraction order.

**Solutions**:

1. **Use faster preset**:
   ```python
   from omeco import TreeSA
   
   # Instead of default
   tree = optimize_code(ixs, out, sizes, TreeSA.fast())
   ```

2. **Try GreedyMethod first**:
   ```python
   from omeco import optimize_code, GreedyMethod
   
   # Much faster, good baseline
   tree = optimize_code(ixs, out, sizes)
   ```

3. **Reduce TreeSA iterations**:
   ```python
   from omeco import TreeSA
   
   optimizer = TreeSA(ntrials=5, niters=20)  # Faster but lower quality
   tree = optimize_code(ixs, out, sizes, optimizer)
   ```

### GPU performance is poor

**Problem**: Execution on GPU is slower than expected.

**Diagnosis**: Check if optimization used correct GPU settings:
```python
comp = contraction_complexity(tree, ixs, sizes)
print(f"Read-write complexity: 2^{comp.rwc:.2f}")
```

If `rwc` is high, the contraction does excessive memory I/O.

**Solution**: Re-optimize with GPU-specific score function:
```python
from omeco import ScoreFunction, TreeSA

# GPU optimization (see GPU Optimization Guide for rw_weight tuning)
score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=10.0,
    sc_target=30.0
)

tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
```

See [GPU Optimization Guide](./guides/gpu-optimization.md) for details.

### Results differ from expected

**Problem**: Optimized contraction order gives different complexity than expected.

**Diagnosis**:

1. **Print the contraction tree**:
   ```python
   tree = optimize_code(ixs, out, sizes)
   print(tree)  # Shows ASCII tree structure
   ```

2. **Check complexity metrics**:
   ```python
   comp = contraction_complexity(tree, ixs, sizes)
   print(f"tc: {comp.tc:.2f}, sc: {comp.sc:.2f}, rwc: {comp.rwc:.2f}")
   ```

3. **Verify input**:
   ```python
   print(f"ixs: {ixs}")
   print(f"out: {out}")
   print(f"sizes: {sizes}")
   
   # Check all indices are covered
   all_indices = set()
   for ix in ixs:
       all_indices.update(ix)
   print(f"All indices: {all_indices}")
   print(f"Missing sizes: {all_indices - set(sizes.keys())}")
   ```

**Common causes**:
- Missing indices in `sizes` dictionary
- Wrong output indices in `out`
- Mismatched dimensions between `ixs` and `sizes`

## Algorithm-Specific Issues

### GreedyMethod gives poor results

**Problem**: Greedy optimization produces sub-optimal contraction order.

**Expected**: Greedy is fast but not optimal. It uses heuristics.

**Solution**: Use TreeSA for better quality:
```python
from omeco import TreeSA

tree = optimize_code(ixs, out, sizes, TreeSA.fast())
```

Comparison:
```python
from omeco import optimize_code, GreedyMethod, TreeSA, contraction_complexity

# Greedy (fast, lower quality)
greedy_tree = optimize_code(ixs, out, sizes)
greedy_comp = contraction_complexity(greedy_tree, ixs, sizes)

# TreeSA (slower, higher quality)
sa_tree = optimize_code(ixs, out, sizes, TreeSA.fast())
sa_comp = contraction_complexity(sa_tree, ixs, sizes)

print(f"Greedy tc: {greedy_comp.tc:.2f}")
print(f"TreeSA tc: {sa_comp.tc:.2f}")
print(f"Improvement: {2**(greedy_comp.tc - sa_comp.tc):.1f}x faster")
```

### TreeSA gets stuck in local minimum

**Problem**: Multiple runs of TreeSA give inconsistent results.

**Cause**: Simulated annealing is stochastic.

**Solutions**:

1. **Increase trials**:
   ```python
   optimizer = TreeSA(ntrials=20, niters=50)  # More exploration
   tree = optimize_code(ixs, out, sizes, optimizer)
   ```

2. **Use best-of-many**:
   ```python
   from omeco import TreeSA, contraction_complexity
   
   best_tree = None
   best_tc = float('inf')
   
   for _ in range(10):
       tree = optimize_code(ixs, out, sizes, TreeSA.fast())
       comp = contraction_complexity(tree, ixs, sizes)
       if comp.tc < best_tc:
           best_tc = comp.tc
           best_tree = tree
   
   print(f"Best tc: {best_tc:.2f}")
   ```

## Integration Issues

### PyTorch: Contraction gives wrong results

**Problem**: Executing the contraction tree with PyTorch tensors produces incorrect output.

**Diagnosis**:

1. **Verify with einsum**:
   ```python
   import torch
   from omeco import optimize_code
   
   # Direct einsum (slow but correct)
   result_ref = torch.einsum("ab,bc,cd->ad", A, B, C)
   
   # Optimized tree
   tree = optimize_code([[0,1], [1,2], [2,3]], [0,3], {0:10, 1:20, 2:30, 3:10})
   result_tree = execute_tree(tree.to_dict(), [A, B, C])
   
   # Compare
   print(f"Match: {torch.allclose(result_ref, result_tree)}")
   print(f"Max error: {(result_ref - result_tree).abs().max()}")
   ```

2. **Check tensor shapes**:
   ```python
   for i, t in enumerate([A, B, C]):
       print(f"Tensor {i}: {t.shape}")
   ```

**Common causes**:
- Mismatched tensor shapes and `sizes` dictionary
- Wrong index mapping in `execute_tree`
- Incorrect einsum string construction

See [PyTorch Integration Guide](./guides/pytorch-integration.md) for complete example.

### Slicing: Incorrect memory calculation

**Problem**: Slicing doesn't reduce memory as expected.

**Diagnosis**:
```python
from omeco import slice_code, sliced_complexity, contraction_complexity

# Before slicing
comp_original = contraction_complexity(tree, ixs, sizes)

# After slicing
sliced = slice_code(tree, ixs, sizes, slicer)
comp_sliced = sliced_complexity(sliced, ixs, sizes)

print(f"Original sc: 2^{comp_original.sc:.2f}")
print(f"Sliced sc: 2^{comp_sliced.sc:.2f}")
print(f"Reduction: {2**(comp_original.sc - comp_sliced.sc):.1f}x")
print(f"Sliced indices: {sliced.slicing()}")
```

**Solutions**:

1. **Increase sc_weight**:
   ```python
   score = ScoreFunction(sc_weight=5.0, sc_target=25.0)
   slicer = TreeSASlicer.fast(score=score)
   ```

2. **Force specific indices**:
   ```python
   slicer = TreeSASlicer.fast(fixed_slices=[1, 2])
   sliced = slice_code(tree, ixs, sizes, slicer)
   ```

## Debugging Tips

### Enable verbose output

Rust library doesn't have built-in verbose mode, but you can add print statements:

```python
# Check optimization result
tree = optimize_code(ixs, out, sizes)
print(f"Tree:\n{tree}")

# Check complexity
comp = contraction_complexity(tree, ixs, sizes)
print(f"tc: 2^{comp.tc:.2f}, sc: 2^{comp.sc:.2f}, rwc: 2^{comp.rwc:.2f}")

# Check slicing
sliced = slice_code(tree, ixs, sizes, slicer)
print(f"Sliced indices: {sliced.slicing()}")
```

### Compare with Julia implementation

If you suspect a bug, compare with the original Julia implementation:

```julia
using OMEinsumContractionOrders

ixs = [[1, 2], [2, 3], [3, 1]]
out = [1]
sizes = Dict(1 => 10, 2 => 20, 3 => 10)

# Greedy
tree = optimize_code(ixs, out, sizes)
println(tree)

# TreeSA
tree = optimize_code(ixs, out, sizes)
println(tree)
```

### Minimal reproducible example

When reporting issues, provide:

```python
from omeco import optimize_code, contraction_complexity

# Minimal example
ixs = [[0, 1], [1, 2]]
out = [0, 2]
sizes = {0: 10, 1: 20, 2: 10}

tree = optimize_code(ixs, out, sizes)
print(f"Tree:\n{tree}")

comp = contraction_complexity(tree, ixs, sizes)
print(f"Complexity: tc={comp.tc:.2f}, sc={comp.sc:.2f}")

# What was expected vs what you got
print(f"Expected tc: ~13.0")
print(f"Got tc: {comp.tc:.2f}")
```

## FAQ

### Q: How do I use different optimizers?

**A**: `optimize_code` is the unified function that accepts different optimizer types:
```python
from omeco import optimize_code, GreedyMethod, TreeSA

# Use default optimizer (GreedyMethod)
tree1 = optimize_code(ixs, out, sizes)

# Explicitly specify GreedyMethod
tree2 = optimize_code(ixs, out, sizes, GreedyMethod())

# Use TreeSA optimizer
tree3 = optimize_code(ixs, out, sizes, TreeSA.fast())
```

### Q: How do I choose between Greedy and TreeSA?

**A**: 
- **GreedyMethod**: Fast (milliseconds to seconds), good for prototyping
- **TreeSA**: Slower (seconds to minutes), better quality, use for production

Rule of thumb:
- < 50 tensors → Greedy is fine
- 50-200 tensors → TreeSA.fast()
- \> 200 tensors → Start with Greedy, then try TreeSA if needed

### Q: What does sc_target mean?

**A**: `sc_target` is the log₂ of the target memory limit in elements.

Calculate it:
```python
import math

# For 4GB memory, float64 (8 bytes)
memory_gb = 4
bytes_per_element = 8
sc_target = math.log2(memory_gb * 1024**3 / bytes_per_element)
# sc_target ≈ 29.0

# For 8GB memory, float32 (4 bytes)
memory_gb = 8
bytes_per_element = 4
sc_target = math.log2(memory_gb * 1024**3 / bytes_per_element)
# sc_target ≈ 31.0
```

### Q: Why is GPU optimization different?

**A**: GPUs have limited memory bandwidth relative to compute (10-100x higher compute-to-memory-bandwidth ratio than CPUs).

- CPU: Memory I/O is fast → `rw_weight=0.0`
- GPU: Use `rw_weight` to penalize memory I/O

See [GPU Optimization Guide](./guides/gpu-optimization.md) for complete tuning methodology.

### Q: Can I save and load optimized trees?

**A**: Yes, use `to_dict()` and JSON:

```python
import json
from omeco import optimize_code

# Optimize
tree = optimize_code(ixs, out, sizes)

# Save
tree_dict = tree.to_dict()
with open("tree.json", "w") as f:
    json.dump(tree_dict, f)

# Load
with open("tree.json", "r") as f:
    tree_dict = json.load(f)

# Use dict directly or reconstruct (implementation dependent)
```

### Q: What if my tensor network has repeated indices?

**A**: Repeated indices (traces) are supported:

```python
# Trace: C[i,j] = Σₖ A[i,k,k] * B[k,j]
ixs = [[0, 1, 1], [1, 2]]  # Note: index 1 appears twice in first tensor
out = [0, 2]
sizes = {0: 10, 1: 20, 2: 10}

tree = optimize_code(ixs, out, sizes)
```

## Still Having Issues?

1. **Check the documentation**:
   - [Quick Start Guide](./quick-start.md)
   - [API Reference](./api-reference.md)
   - [Examples](https://github.com/GiggleLiu/omeco/tree/master/examples)

2. **Search existing issues**:
   - [GitHub Issues](https://github.com/GiggleLiu/omeco/issues)

3. **Report a bug**:
   - Include minimal reproducible example
   - Include Python/Rust version, OS, omeco version
   - Show expected vs actual behavior

4. **Ask for help**:
   - GitHub Discussions
   - Include context and what you've tried
