#!/bin/bash
# Run all benchmarks (Rust, Python, Julia) and compare results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "OMECO Benchmark Suite"
echo "========================================"
echo

# Generate graphs if needed
if [ ! -d "$SCRIPT_DIR/graphs" ] || [ -z "$(ls -A $SCRIPT_DIR/graphs 2>/dev/null)" ]; then
    echo "Generating benchmark graphs..."
    python "$SCRIPT_DIR/generate_graphs.py"
    echo
fi

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run Rust benchmark (release mode)
echo "========================================"
echo "Running Rust benchmark (release mode)..."
echo "========================================"
cargo run --release --example benchmark -p omeco
echo

# Run Julia benchmark
echo "========================================"
echo "Running Julia benchmark..."
echo "========================================"
julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/benchmark_julia.jl"
echo

# Run Python benchmark
echo "========================================"
echo "Running Python benchmark..."
echo "========================================"
python "$SCRIPT_DIR/benchmark_python.py"
echo

# Compare results
echo "========================================"
echo "Comparing results..."
echo "========================================"
python "$SCRIPT_DIR/compare_results.py"
