# Benchmarks: Python (Rust) vs Julia

Comparing tensor network contraction order optimization between:
- **Python**: omeco (Rust via PyO3)
- **Julia**: OMEinsumContractionOrders.jl

## Running

```bash
# Run both benchmarks
./run_benchmarks.sh

# Or run individually
python3 benchmark_python.py
julia --project=. benchmark_julia.jl
```

## Test Cases

1. **Matrix Chain (n=10)**: Chain of 10 matrix multiplications
2. **Grid 4×4**: 16-tensor 2D grid network (PEPS-like)
3. **Grid 5×5**: 25-tensor 2D grid network

## Results (example on typical hardware)

| Problem   | Python/Rust (ms) | Julia (ms) | Speedup |
|-----------|------------------|------------|---------|
| chain_10  | 316              | 251        | 0.79×   |
| grid_4x4  | 1662             | 1452       | 0.87×   |
| grid_5x5  | 3479             | 2887       | 0.83×   |

Note: Julia implementation is mature and highly optimized. The Rust implementation
provides comparable performance with the advantage of easy Python integration.

