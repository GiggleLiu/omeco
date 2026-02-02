#!/usr/bin/env python3
"""
Benchmark TreeSA contraction order optimization in Python.
Uses omeco (Rust via PyO3) with shared graph files.
"""

import json
import time
from pathlib import Path


GRAPHS_DIR = Path(__file__).parent / "graphs"
RESULTS_DIR = Path(__file__).parent / "results"


def load_graph(name: str):
    """Load a graph from the graphs directory."""
    path = GRAPHS_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)

    ixs = data["ixs"]
    iy = data["iy"]
    sizes = {int(k): v for k, v in data["sizes"].items()}
    return ixs, iy, sizes, data.get("description", name)


def run_benchmark(name: str, ntrials: int = 1, niters: int = 50):
    from omeco import GreedyMethod, TreeSA, optimize_code

    ixs, iy, sizes, description = load_graph(name)

    print("=" * 70)
    print(f"Benchmark: {name}")
    print(f"  Description: {description}")
    print(f"  Tensors: {len(ixs)}")
    print(f"  Indices: {len(sizes)}")
    print()

    # ========== Greedy ==========
    print("GreedyMethod:")
    greedy = GreedyMethod()

    # Warmup
    _ = optimize_code(ixs, iy, sizes, greedy)

    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        greedy_result = optimize_code(ixs, iy, sizes, greedy)
    greedy_time = time.perf_counter() - start

    greedy_cc = greedy_result.complexity(ixs, sizes)
    print(f"  Time complexity (tc):       {greedy_cc.tc:.6f}")
    print(f"  Space complexity (sc):      {greedy_cc.sc:.6f}")
    print(f"  Read-write complexity (rwc): {greedy_cc.rwc:.6f}")
    print(f"  Execution time (10 runs):   {greedy_time*1000:.2f} ms")
    print(f"  Average per run:            {greedy_time/10*1000:.4f} ms")
    print()

    # ========== TreeSA ==========
    print(f"TreeSA (ntrials={ntrials}, niters={niters}):")
    treesa_cfg = TreeSA(ntrials=ntrials, niters=niters)

    # Warmup
    _ = optimize_code(ixs, iy, sizes, treesa_cfg)

    # Benchmark
    start = time.perf_counter()
    for _ in range(3):
        treesa_result = optimize_code(ixs, iy, sizes, treesa_cfg)
    treesa_time = time.perf_counter() - start

    treesa_cc = treesa_result.complexity(ixs, sizes)
    print(f"  Time complexity (tc):       {treesa_cc.tc:.6f}")
    print(f"  Space complexity (sc):      {treesa_cc.sc:.6f}")
    print(f"  Read-write complexity (rwc): {treesa_cc.rwc:.6f}")
    print(f"  Execution time (3 runs):    {treesa_time*1000:.2f} ms")
    print(f"  Average per run:            {treesa_time/3*1000:.2f} ms")
    print()

    # ========== Improvement ==========
    tc_improvement = greedy_cc.tc - treesa_cc.tc
    sc_improvement = greedy_cc.sc - treesa_cc.sc
    print(f"  Improvement over Greedy:")
    print(f"    tc reduction: {tc_improvement:.2f} ({tc_improvement/greedy_cc.tc*100:.1f}%)")
    print(f"    sc reduction: {sc_improvement:.2f}")
    print()

    return {
        "name": name,
        "description": description,
        "tensors": len(ixs),
        "indices": len(sizes),
        "greedy": {
            "tc": greedy_cc.tc,
            "sc": greedy_cc.sc,
            "rwc": greedy_cc.rwc,
            "avg_ms": greedy_time / 10 * 1000,
            "total_ms": greedy_time * 1000,
            "runs": 10,
        },
        "treesa": {
            "ntrials": ntrials,
            "niters": niters,
            "tc": treesa_cc.tc,
            "sc": treesa_cc.sc,
            "rwc": treesa_cc.rwc,
            "avg_ms": treesa_time / 3 * 1000,
            "total_ms": treesa_time * 1000,
            "runs": 3,
        },
    }


def main():
    print()
    print("=" * 70)
    print("Python TreeSA Benchmark")
    print("Backend: omeco (Rust via PyO3)")
    print("=" * 70)
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    results = {}

    # Define benchmarks: (graph_name, ntrials, niters)
    benchmarks = [
        ("chain_10", 1, 50),
        ("chain_20", 1, 50),
        ("grid_4x4", 1, 100),
        ("grid_5x5", 1, 100),
        ("grid_6x6", 1, 100),
        ("petersen", 1, 50),
        ("reg3_50", 1, 100),
        ("reg3_100", 1, 100),
        ("reg3_250", 1, 100),
    ]

    for name, ntrials, niters in benchmarks:
        if not (GRAPHS_DIR / f"{name}.json").exists():
            print(f"Skipping {name} (graph file not found)")
            continue
        results[name] = run_benchmark(name, ntrials=ntrials, niters=niters)

    # ========== Summary Table ==========
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Graph':<15} {'Tensors':>8} {'Indices':>8} │ {'Greedy tc':>10} {'TreeSA tc':>10} │ {'Greedy ms':>10} {'TreeSA ms':>10}")
    print("─" * 15 + "─" * 8 + "─" * 8 + "─┼" + "─" * 10 + "─" * 10 + "─┼" + "─" * 10 + "─" * 10)
    for name, r in results.items():
        print(
            f"{name:<15} {r['tensors']:>8} {r['indices']:>8} │ "
            f"{r['greedy']['tc']:>10.2f} {r['treesa']['tc']:>10.2f} │ "
            f"{r['greedy']['avg_ms']:>10.3f} {r['treesa']['avg_ms']:>10.2f}"
        )
    print()

    # ========== Save Results ==========
    # Save combined results
    output = {
        "language": "python",
        "backend": "omeco (Rust via PyO3)",
        "results": results,
    }
    output_path = RESULTS_DIR / "python_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save separate greedy results (old format compatibility)
    greedy_output = {
        "language": "python",
        "backend": "omeco (Rust via PyO3)",
        "method": "greedy",
        "results": {
            name: {
                "tensors": r["tensors"],
                "indices": r["indices"],
                "tc": r["greedy"]["tc"],
                "sc": r["greedy"]["sc"],
                "rwc": r["greedy"]["rwc"],
                "avg_ms": r["greedy"]["avg_ms"],
            }
            for name, r in results.items()
        },
    }
    with open(RESULTS_DIR / "results_rust_greedy.json", "w") as f:
        json.dump(greedy_output, f, indent=2)

    # Save separate treesa results (old format compatibility)
    treesa_output = {
        "language": "python",
        "backend": "omeco (Rust via PyO3)",
        "method": "treesa",
        "results": {
            name: {
                "tensors": r["tensors"],
                "indices": r["indices"],
                "ntrials": r["treesa"]["ntrials"],
                "niters": r["treesa"]["niters"],
                "tc": r["treesa"]["tc"],
                "sc": r["treesa"]["sc"],
                "rwc": r["treesa"]["rwc"],
                "avg_ms": r["treesa"]["avg_ms"],
            }
            for name, r in results.items()
        },
    }
    with open(RESULTS_DIR / "results_rust_treesa.json", "w") as f:
        json.dump(treesa_output, f, indent=2)

    print(f"Results saved to:")
    print(f"  {output_path}")
    print(f"  {RESULTS_DIR / 'results_rust_greedy.json'}")
    print(f"  {RESULTS_DIR / 'results_rust_treesa.json'}")


if __name__ == "__main__":
    main()
