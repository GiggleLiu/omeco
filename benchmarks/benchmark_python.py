#!/usr/bin/env python3
"""
Benchmark TreeSA contraction order optimization in Python.
Uses omeco (Rust via PyO3)
"""

import json
import time
import random
from typing import Dict, List, Tuple


def chain_network(n: int, d: int) -> Tuple[List[List[int]], List[int], Dict[int, int]]:
    """Matrix chain of n matrices."""
    labels = list(range(n + 1))
    ixs = [[labels[i], labels[i + 1]] for i in range(n)]
    iy = [labels[0], labels[-1]]
    sizes = {l: d for l in labels}
    return ixs, iy, sizes


def grid_network(rows: int, cols: int, d: int) -> Tuple[List[List[int]], List[int], Dict[int, int]]:
    """2D grid tensor network (like PEPS)."""
    # Count edges
    h_edges = rows * (cols - 1)  # horizontal edges
    v_edges = (rows - 1) * cols  # vertical edges
    
    # Assign integer labels to edges
    label = 0
    h_edge_map = {}  # horizontal edge from (r, c) to (r, c+1)
    v_edge_map = {}  # vertical edge from (r, c) to (r+1, c)
    
    for r in range(rows):
        for c in range(cols - 1):
            h_edge_map[(r, c)] = label
            label += 1
    
    for r in range(rows - 1):
        for c in range(cols):
            v_edge_map[(r, c)] = label
            label += 1
    
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
            if tensor_ixs:
                ixs.append(tensor_ixs)
    
    iy = []  # scalar output
    return ixs, iy, sizes


def random_regular_graph(n: int, degree: int, d: int, seed: int = 42) -> Tuple[List[List[int]], List[int], Dict[int, int]]:
    """
    Random regular graph tensor network.
    
    Each vertex is a tensor with `degree` indices.
    Total edges = n * degree / 2.
    """
    random.seed(seed)
    
    # Generate random regular graph using configuration model
    # Each vertex has `degree` half-edges
    half_edges = []
    for v in range(n):
        for _ in range(degree):
            half_edges.append(v)
    
    random.shuffle(half_edges)
    
    # Pair up half-edges to form edges
    edges = []
    edge_label = 0
    vertex_edges: Dict[int, List[int]] = {v: [] for v in range(n)}
    
    for i in range(0, len(half_edges), 2):
        v1, v2 = half_edges[i], half_edges[i + 1]
        # Skip self-loops and multi-edges for simplicity
        if v1 != v2:
            vertex_edges[v1].append(edge_label)
            vertex_edges[v2].append(edge_label)
            edge_label += 1
    
    ixs = [vertex_edges[v] for v in range(n) if vertex_edges[v]]
    sizes = {e: d for e in range(edge_label)}
    iy = []  # scalar output
    
    return ixs, iy, sizes


def run_benchmark(name: str, ixs, iy, sizes, ntrials=1, niters=50):
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
        "tensors": len(ixs),
        "indices": len(sizes),
        "ntrials": ntrials,
        "niters": niters,
        "greedy_avg_ms": greedy_time / 10 * 1000,
        "greedy_tc": greedy_cc.tc,
        "greedy_sc": greedy_cc.sc,
        "greedy_rwc": greedy_cc.rwc,
        "treesa_avg_ms": treesa_time / 3 * 1000,
        "treesa_tc": treesa_cc.tc,
        "treesa_sc": treesa_cc.sc,
        "treesa_rwc": treesa_cc.rwc,
    }


def main():
    import os
    
    print()
    print("Python TreeSA Benchmark")
    print("omeco (Rust via PyO3)")
    print("=" * 60)
    print()
    
    results = {}
    
    # Small: matrix chain
    ixs, iy, sizes = chain_network(10, 100)
    results["chain_10"] = run_benchmark("Matrix Chain (n=10)", ixs, iy, sizes, ntrials=1, niters=50)
    
    # Medium: small grid
    ixs, iy, sizes = grid_network(4, 4, 2)
    results["grid_4x4"] = run_benchmark("Grid 4x4", ixs, iy, sizes, ntrials=1, niters=100)
    
    # Large: bigger grid  
    ixs, iy, sizes = grid_network(5, 5, 2)
    results["grid_5x5"] = run_benchmark("Grid 5x5", ixs, iy, sizes, ntrials=1, niters=100)
    
    # Random 3-regular graph n=250
    ixs, iy, sizes = random_regular_graph(250, 3, 2)
    results["reg3_250"] = run_benchmark("Random 3-regular n=250", ixs, iy, sizes, ntrials=1, niters=100)
    
    # Summary
    print("=" * 60)
    print("Summary (Python/Rust):")
    print("-" * 60)
    print(f"{'Problem':<20} {'Greedy (ms)':<15} {'TreeSA (ms)':<15} {'Greedy tc':<12} {'TreeSA tc':<12}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['greedy_avg_ms']:<15.3f} {r['treesa_avg_ms']:<15.2f} {r['greedy_tc']:<12.2f} {r['treesa_tc']:<12.2f}")
    
    # Save Greedy results to JSON
    greedy_output = {
        "language": "rust",
        "backend": "omeco (Rust via PyO3)",
        "method": "greedy",
        "results": {name: {
            "tensors": r["tensors"],
            "indices": r["indices"],
            "avg_ms": r["greedy_avg_ms"],
            "tc": r["greedy_tc"],
            "sc": r["greedy_sc"],
            "rwc": r["greedy_rwc"],
        } for name, r in results.items()}
    }
    greedy_path = os.path.join(os.path.dirname(__file__), "results_rust_greedy.json")
    with open(greedy_path, "w") as f:
        json.dump(greedy_output, f, indent=2)
    
    # Save TreeSA results to JSON
    treesa_output = {
        "language": "rust",
        "backend": "omeco (Rust via PyO3)",
        "method": "treesa",
        "results": {name: {
            "tensors": r["tensors"],
            "indices": r["indices"],
            "ntrials": r["ntrials"],
            "niters": r["niters"],
            "avg_ms": r["treesa_avg_ms"],
            "tc": r["treesa_tc"],
            "sc": r["treesa_sc"],
            "rwc": r["treesa_rwc"],
        } for name, r in results.items()}
    }
    treesa_path = os.path.join(os.path.dirname(__file__), "results_rust_treesa.json")
    with open(treesa_path, "w") as f:
        json.dump(treesa_output, f, indent=2)
    
    print()
    print(f"Results saved to:")
    print(f"  {greedy_path}")
    print(f"  {treesa_path}")


if __name__ == "__main__":
    main()
