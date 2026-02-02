#!/usr/bin/env python3
"""Compare benchmark results from Rust, Python, and Julia."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    rust = load_results("rust_results.json")
    python = load_results("python_results.json")
    julia = load_results("julia_results.json")

    print("=" * 100)
    print("BENCHMARK COMPARISON: Rust vs Python vs Julia")
    print("=" * 100)
    print()

    # Collect all graph names (results are dictionaries keyed by graph name)
    graphs = set()
    if rust:
        graphs.update(rust["results"].keys())
    if python:
        graphs.update(python["results"].keys())
    if julia:
        graphs.update(julia["results"].keys())

    graphs = sorted(graphs)

    # Print Greedy comparison
    print("GREEDY METHOD")
    print("-" * 100)
    print(f"{'Graph':<15} {'Rust tc':<10} {'Python tc':<10} {'Julia tc':<10} {'Rust ms':<12} {'Python ms':<12} {'Julia ms':<12}")
    print("-" * 100)

    for name in graphs:
        rust_r = rust["results"].get(name) if rust else None
        python_r = python["results"].get(name) if python else None
        julia_r = julia["results"].get(name) if julia else None

        rust_tc = f"{rust_r['greedy']['tc']:.2f}" if rust_r else "-"
        python_tc = f"{python_r['greedy']['tc']:.2f}" if python_r else "-"
        julia_tc = f"{julia_r['greedy']['tc']:.2f}" if julia_r else "-"
        rust_ms = f"{rust_r['greedy']['avg_ms']:.3f}" if rust_r else "-"
        python_ms = f"{python_r['greedy']['avg_ms']:.3f}" if python_r else "-"
        julia_ms = f"{julia_r['greedy']['avg_ms']:.3f}" if julia_r else "-"

        print(f"{name:<15} {rust_tc:<10} {python_tc:<10} {julia_tc:<10} {rust_ms:<12} {python_ms:<12} {julia_ms:<12}")

    print()

    # Print TreeSA comparison
    print("TREESA METHOD")
    print("-" * 100)
    print(f"{'Graph':<15} {'Rust tc':<10} {'Python tc':<10} {'Julia tc':<10} {'Rust ms':<12} {'Python ms':<12} {'Julia ms':<12}")
    print("-" * 100)

    for name in graphs:
        rust_r = rust["results"].get(name) if rust else None
        python_r = python["results"].get(name) if python else None
        julia_r = julia["results"].get(name) if julia else None

        rust_tc = f"{rust_r['treesa']['tc']:.2f}" if rust_r else "-"
        python_tc = f"{python_r['treesa']['tc']:.2f}" if python_r else "-"
        julia_tc = f"{julia_r['treesa']['tc']:.2f}" if julia_r else "-"
        rust_ms = f"{rust_r['treesa']['avg_ms']:.2f}" if rust_r else "-"
        python_ms = f"{python_r['treesa']['avg_ms']:.2f}" if python_r else "-"
        julia_ms = f"{julia_r['treesa']['avg_ms']:.2f}" if julia_r else "-"

        print(f"{name:<15} {rust_tc:<10} {python_tc:<10} {julia_tc:<10} {rust_ms:<12} {python_ms:<12} {julia_ms:<12}")

    print()
    print("=" * 100)
    print("NOTES:")
    print("- tc = time complexity (log2 scale, lower is better)")
    print("- ms = execution time in milliseconds")
    print("- Rust benchmark runs in release mode, Python uses Rust backend via PyO3")
    print("- All implementations should produce similar tc values for the same graph")
    print("=" * 100)


if __name__ == "__main__":
    main()
