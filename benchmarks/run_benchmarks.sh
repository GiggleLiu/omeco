#!/bin/bash
# Run Python and Julia benchmarks for TreeSA optimization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "TreeSA Optimization Benchmark"
echo "Comparing Python (Rust/PyO3) vs Julia"
echo "=========================================="
echo

# Build Python bindings if needed
if ! python3 -c "import omeco" 2>/dev/null; then
    echo "Building Python bindings..."
    cd omeco-python
    pip install maturin
    maturin develop --release
    cd ..
fi

echo
echo "=========================================="
echo "Running Python Benchmark"
echo "=========================================="
python3 benchmarks/benchmark_python.py

echo
echo "=========================================="
echo "Running Julia Benchmark"  
echo "=========================================="
cd benchmarks && julia --project=. benchmark_julia.jl
