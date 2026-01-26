# Performance Benchmarks

Real-world performance comparison with Julia implementation.

## Benchmark Methodology

All benchmarks compare against [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl), the original Julia implementation.

**Test Environment**:
- Benchmarks run via Python bindings (PyO3)
- Configuration: `ntrials=1`, `niters=50-100`
- See `benchmarks/` directory for scripts

## Rust vs Julia (TreeSA)

Comparison of TreeSA optimizer performance:

| Problem | Tensors | Indices | Rust (ms) | Julia (ms) | tc (Rust) | tc (Julia) | Speedup |
|---------|---------|---------|-----------|------------|-----------|------------|---------|
| chain_10 | 10 | 11 | 18.5 | 31.6 | 23.10 | 23.10 | **1.7x** |
| grid_4x4 | 16 | 24 | 88.0 | 150.7 | 9.18 | 9.26 | **1.7x** |
| grid_5x5 | 25 | 40 | 155.4 | 297.7 | 10.96 | 10.96 | **1.9x** |
| reg3_250 | 250 | 372 | 2,435 | 5,099 | 48.00 | 47.17 | **2.1x** |

**Results**:
- Rust is **1.7-2.1x faster** than Julia for TreeSA optimization
- Both implementations find solutions with comparable time complexity (tc)
- Solution quality is nearly identical between implementations

## Greedy Method Performance

GreedyMethod is extremely fast for quick optimization:

| Problem | Tensors | Time (ms) | tc | sc |
|---------|---------|-----------|-----|-----|
| chain_10 | 10 | 0.04 | 23.10 | 13.29 |
| grid_4x4 | 16 | 0.12 | 9.54 | 5.0 |
| grid_5x5 | 25 | 0.23 | 11.28 | 6.0 |
| reg3_250 | 250 | 7.5 | 69.00 | 47.0 |

**Key Observations**:
- Greedy is **100-300x faster** than TreeSA
- Greedy gives good results for small problems (chain_10, grids)
- For large problems (reg3_250), TreeSA finds much better solutions:
  - Greedy: tc = 69.00
  - TreeSA: tc = 48.00
  - **Improvement**: 2^(69-48) = 2^21 = **2 million times faster** execution!

## Python Overhead

Python bindings via PyO3 add minimal overhead:

**TreeSA (grid_4x4, 100 iterations)**:
- Pure Rust backend: ~88ms
- Python call overhead: <1ms (~1%)

**Greedy (reg3_250)**:
- Pure optimization: ~7.5ms
- Python overhead: <0.5ms (~6%)

**Conclusion**: Python overhead is negligible, especially for TreeSA optimization.

## Algorithm Comparison

When to use each algorithm:

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| **Quick prototyping** | Greedy | Sub-millisecond optimization |
| **Production (< 50 tensors)** | Greedy | Fast enough, good quality |
| **Production (50-200 tensors)** | TreeSA.fast() | Better quality, still reasonable time |
| **Large networks (> 200 tensors)** | TreeSA | Significant quality improvement |
| **Time-critical** | Greedy | Predictable fast performance |
| **Execution-critical** | TreeSA | Better contraction order = faster execution |

## Execution Time Savings

The time spent optimizing pays off during execution:

**Example: reg3_250 network**
- Greedy optimization: 7.5ms
- TreeSA optimization: 2,435ms (~2.4 seconds)
- **Extra optimization time**: +2.4 seconds

But the execution improvement:
- Greedy tc: 69.00 → ~2^69 FLOPs
- TreeSA tc: 48.00 → ~2^48 FLOPs
- **Execution speedup**: 2^21 = **2 million times faster**

If executing even once, TreeSA optimization is worth it!

## Hardware Recommendations

| Tensor Count | RAM | CPU | Recommendation |
|--------------|-----|-----|----------------|
| < 20 | 4GB | Any | Greedy is sufficient |
| 20-100 | 8GB | 4+ cores | TreeSA.fast() for production |
| 100-500 | 16GB | 8+ cores | TreeSA with multiple trials |
| > 500 | 32GB+ | 16+ cores | TreeSA (may take minutes) |

## Running Benchmarks

Reproduce the benchmarks yourself:

```bash
cd benchmarks

# Python (Rust via PyO3)
python benchmark_python.py

# Julia (original implementation)
julia --project=. benchmark_julia.jl

# Compare results
python -c "
import json
with open('results_rust_treesa.json') as f:
    rust = json.load(f)
with open('results_julia_treesa.json') as f:
    julia = json.load(f)

for problem in rust['results']:
    r = rust['results'][problem]['avg_ms']
    j = julia['results'][problem]['avg_ms']
    print(f'{problem}: Rust {r:.1f}ms, Julia {j:.1f}ms, Speedup {j/r:.2f}x')
"
```

## Profiling Your Code

### Python Profiling

```python
import time
from omeco import optimize_code, TreeSA

# Time optimization
start = time.time()
tree = optimize_code(ixs, out, sizes, TreeSA.fast())
opt_time = time.time() - start

# Check complexity
comp = tree.complexity(ixs, sizes)

print(f"Optimization: {opt_time:.3f}s")
print(f"Time complexity: 2^{comp.tc:.2f} = {2**comp.tc:.2e} FLOPs")
print(f"Space complexity: 2^{comp.sc:.2f} elements")

# Estimate execution time (assuming 10 GFLOP/s CPU)
execution_seconds = 2**comp.tc / 1e10
print(f"Estimated execution: {execution_seconds:.1f}s @ 10 GFLOP/s")
```

### Rust Profiling

```bash
# CPU profiling with flamegraph
cargo install flamegraph
cargo flamegraph --example your_example

# Criterion benchmarks (if available)
cargo bench
```

## Performance Tips

1. **Start with Greedy**: Always try Greedy first to get a baseline
2. **Use TreeSA for production**: The extra optimization time pays off
3. **Increase ntrials for critical code**: More trials = better solutions
4. **Profile execution time**: Verify that better tc actually improves runtime
5. **Consider memory**: Use sc_target if memory is constrained

## Next Steps

- [Algorithm Comparison](./algorithms/comparison.md) - Detailed algorithm trade-offs
- [Quick Start](./quick-start.md) - Get started optimizing
- [Troubleshooting](./troubleshooting.md) - Common performance issues
