#!/usr/bin/env python3
"""
Benchmark TreeSA contraction order optimization in Python.
Uses omeco (Rust via PyO3)

Note: The Rust bindings use single-character labels.
"""

import time
import string
from typing import Dict, List, Tuple

# Character pool for labels (62 chars: a-z, A-Z, 0-9)
CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits


def chain_network(n: int, d: int) -> Tuple[List[List[str]], List[str], Dict[str, int]]:
    """Matrix chain of n matrices. Requires n+1 <= 62 labels."""
    if n + 1 > len(CHARS):
        raise ValueError(f"Chain too long: need {n+1} labels, have {len(CHARS)}")
    labels = list(CHARS[: n + 1])
    ixs = [[labels[i], labels[i + 1]] for i in range(n)]
    iy = [labels[0], labels[-1]]
    sizes = {l: d for l in labels}
    return ixs, iy, sizes


def grid_network(rows: int, cols: int, d: int) -> Tuple[List[List[str]], List[str], Dict[str, int]]:
    """
    2D grid tensor network (like PEPS).
    Number of edges: (rows-1)*cols + rows*(cols-1) horizontal + vertical edges.
    """
    # Count edges
    h_edges = rows * (cols - 1)  # horizontal edges
    v_edges = (rows - 1) * cols  # vertical edges
    total_edges = h_edges + v_edges
    
    if total_edges > len(CHARS):
        raise ValueError(f"Grid too large: need {total_edges} labels, have {len(CHARS)}")
    
    # Assign characters to edges
    edge_chars = list(CHARS[:total_edges])
    char_idx = 0
    
    # Map (edge_type, r, c) -> char
    h_edge_map = {}  # horizontal edge from (r, c) to (r, c+1)
    v_edge_map = {}  # vertical edge from (r, c) to (r+1, c)
    
    for r in range(rows):
        for c in range(cols - 1):
            h_edge_map[(r, c)] = edge_chars[char_idx]
            char_idx += 1
    
    for r in range(rows - 1):
        for c in range(cols):
            v_edge_map[(r, c)] = edge_chars[char_idx]
            char_idx += 1
    
    ixs = []
    sizes = {}
    
    for r in range(rows):
        for c in range(cols):
            tensor_ixs = []
            # Left edge
            if c > 0:
                e = h_edge_map[(r, c - 1)]
                tensor_ixs.append(e)
                sizes[e] = d
            # Right edge
            if c < cols - 1:
                e = h_edge_map[(r, c)]
                tensor_ixs.append(e)
                sizes[e] = d
            # Top edge
            if r > 0:
                e = v_edge_map[(r - 1, c)]
                tensor_ixs.append(e)
                sizes[e] = d
            # Bottom edge
            if r < rows - 1:
                e = v_edge_map[(r, c)]
                tensor_ixs.append(e)
                sizes[e] = d
            if tensor_ixs:  # Skip if no edges (shouldn't happen for grid > 1x1)
                ixs.append(tensor_ixs)
    
    iy = []  # scalar output
    return ixs, iy, sizes


def run_benchmark(name: str, ixs, iy, sizes, ntrials=10, niters=50):
    from omeco import (
        GreedyMethod,
        TreeSA,
        optimize_greedy,
        optimize_treesa,
        contraction_complexity,
    )
    
    print("=" * 60)
    print(f"Benchmark: {name}")
    print(f"  Tensors: {len(ixs)}")
    print(f"  Indices: {len(sizes)}")
    print()
    
    # Greedy warmup + benchmark
    print("GreedyMethod:")
    # Warmup
    _ = optimize_greedy(ixs, iy, sizes)
    
    start = time.perf_counter()
    for _ in range(10):
        greedy_result = optimize_greedy(ixs, iy, sizes)
    greedy_time = time.perf_counter() - start
    
    greedy_cc = contraction_complexity(greedy_result, ixs, sizes)
    print(f"  tc={greedy_cc.tc:.2f}, sc={greedy_cc.sc:.2f}, rwc={greedy_cc.rwc:.2f}")
    print(f"  Time (10 runs): {greedy_time*1000:.2f}ms, avg: {greedy_time/10*1000:.4f}ms")
    print()
    
    # TreeSA
    print(f"TreeSA (ntrials={ntrials}, niters={niters}):")
    treesa_cfg = TreeSA().with_ntrials(ntrials).with_niters(niters)
    
    # Warmup
    _ = optimize_treesa(ixs, iy, sizes, treesa_cfg)
    
    start = time.perf_counter()
    for _ in range(3):
        treesa_result = optimize_treesa(ixs, iy, sizes, treesa_cfg)
    treesa_time = time.perf_counter() - start
    
    treesa_cc = contraction_complexity(treesa_result, ixs, sizes)
    print(f"  tc={treesa_cc.tc:.2f}, sc={treesa_cc.sc:.2f}, rwc={treesa_cc.rwc:.2f}")
    print(f"  Time (3 runs): {treesa_time*1000:.2f}ms, avg: {treesa_time/3*1000:.2f}ms")
    print()
    
    return {
        "greedy_avg_ms": greedy_time / 10 * 1000,
        "treesa_avg_ms": treesa_time / 3 * 1000,
        "greedy_tc": greedy_cc.tc,
        "treesa_tc": treesa_cc.tc,
    }


def main():
    print()
    print("Python TreeSA Benchmark")
    print("omeco (Rust via PyO3)")
    print("=" * 60)
    print()
    
    results = {}
    
    # Small: matrix chain
    ixs, iy, sizes = chain_network(10, 100)
    results["chain_10"] = run_benchmark("Matrix Chain (n=10)", ixs, iy, sizes)
    
    # Medium: small grid
    ixs, iy, sizes = grid_network(4, 4, 2)
    results["grid_4x4"] = run_benchmark("Grid 4x4", ixs, iy, sizes, ntrials=10, niters=100)
    
    # Large: bigger grid  
    ixs, iy, sizes = grid_network(5, 5, 2)
    results["grid_5x5"] = run_benchmark("Grid 5x5", ixs, iy, sizes, ntrials=10, niters=100)
    
    # Summary
    print("=" * 60)
    print("Summary (Python/Rust):")
    print("-" * 60)
    print(f"{'Problem':<15} {'Greedy (ms)':<15} {'TreeSA (ms)':<15}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<15} {r['greedy_avg_ms']:<15.3f} {r['treesa_avg_ms']:<15.2f}")


if __name__ == "__main__":
    main()
