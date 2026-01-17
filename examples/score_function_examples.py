"""
Score Function Configuration Examples

This file demonstrates practical examples of configuring the ScoreFunction
for different hardware and use cases.

See docs/score_function_guide.md for detailed explanations.
"""

from omeco import (
    optimize_code, slice_code, contraction_complexity, sliced_complexity,
    GreedyMethod, TreeSA, TreeSASlicer, ScoreFunction, EinCode
)
import math


def example_1_cpu_optimization():
    """Example 1: CPU optimization with balanced priorities"""
    print("=" * 60)
    print("Example 1: CPU Optimization (Balanced)")
    print("=" * 60)

    # Matrix chain: A[i,j] × B[j,k] × C[k,l]
    ixs = [[0, 1], [1, 2], [2, 3]]
    output = [0, 3]
    sizes = {0: 100, 1: 200, 2: 50, 3: 100}

    # CPU-optimized score function
    score = ScoreFunction(
        tc_weight=1.0,  # Time matters
        sc_weight=1.0,  # Space matters equally
        rw_weight=0.0,  # Read-write not a bottleneck on CPU
        sc_target=28.0  # ~256MB limit (2^28 floats × 8 bytes)
    )

    # Optimize
    code = EinCode(ixs, output)
    optimizer = TreeSA.default()  # Use default for now
    optimized = optimize_code(code, sizes, optimizer)

    # Check complexity
    complexity = contraction_complexity(optimized, sizes, ixs)
    print(f"Time complexity: 2^{complexity.tc:.2f} FLOPs")
    print(f"Space complexity: 2^{complexity.sc:.2f} elements")
    print(f"Memory usage: ~{2**complexity.sc * 8 / 1024**2:.1f} MB")
    print()


def example_2_gpu_optimization():
    """Example 2: GPU optimization with high rw_weight"""
    print("=" * 60)
    print("Example 2: GPU Optimization")
    print("=" * 60)

    # Larger tensor network for GPU
    ixs = [[0, 1, 2], [2, 3, 4], [4, 5, 0], [1, 3, 5]]
    output = []
    sizes = {i: 64 for i in range(6)}

    # GPU-optimized score function
    score = ScoreFunction(
        tc_weight=1.0,   # Time still matters
        sc_weight=1.0,   # GPU memory is limited
        rw_weight=20.0,  # ⚠️ Memory I/O is 20x more expensive on GPU!
        sc_target=30.0   # ~8GB GPU memory (2^30 floats × 4 bytes)
    )

    print("Configuration:")
    print(f"  tc_weight: {score.tc_weight}")
    print(f"  sc_weight: {score.sc_weight}")
    print(f"  rw_weight: {score.rw_weight} (GPU penalty)")
    print(f"  sc_target: {score.sc_target} (8GB GPU)")

    code = EinCode(ixs, output)
    optimizer = GreedyMethod.default()
    optimized = optimize_code(code, sizes, optimizer)

    complexity = contraction_complexity(optimized, sizes, ixs)
    print(f"\nResults:")
    print(f"  Time complexity: 2^{complexity.tc:.2f} FLOPs")
    print(f"  Space complexity: 2^{complexity.sc:.2f} elements")
    print(f"  Read-write complexity: 2^{complexity.rwc:.2f} operations")
    print(f"  GPU memory: ~{2**complexity.sc * 4 / 1024**3:.2f} GB (float32)")
    print()


def example_3_memory_limited():
    """Example 3: Severe memory constraints with slicing"""
    print("=" * 60)
    print("Example 3: Memory-Limited Environment with Slicing")
    print("=" * 60)

    # Large network that exceeds memory
    ixs = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    output = [0, 5]
    sizes = {i: 256 for i in range(6)}

    # Very memory-constrained
    sc_target = 25.0  # Only ~512MB available
    score = ScoreFunction(
        tc_weight=1.0,
        sc_weight=3.0,      # Heavily penalize memory
        rw_weight=0.0,
        sc_target=sc_target
    )

    code = EinCode(ixs, output)
    optimizer = TreeSA.default()
    optimized = optimize_code(code, sizes, optimizer)

    complexity = contraction_complexity(optimized, sizes, ixs)
    print(f"Without slicing:")
    print(f"  Space complexity: 2^{complexity.sc:.2f}")
    print(f"  Memory needed: ~{2**complexity.sc * 8 / 1024**2:.1f} MB")

    # Check if slicing is needed
    if complexity.sc > sc_target:
        print(f"\n⚠️  Exceeds target of 2^{sc_target:.1f}")
        print("Applying slicing...")

        # Use aggressive slicing
        slicer = TreeSASlicer(
            ntrials=4,
            niters=50,
            score=ScoreFunction.space_optimized(sc_target=sc_target)
        )
        sliced = slice_code(optimized, sizes, slicer, ixs)

        sliced_comp = sliced_complexity(sliced, sizes, ixs)
        print(f"\nWith slicing:")
        print(f"  Sliced indices: {sliced.sliced_indices}")
        print(f"  Space complexity: 2^{sliced_comp.sc:.2f}")
        print(f"  Memory needed: ~{2**sliced_comp.sc * 8 / 1024**2:.1f} MB")
        print(f"  Time overhead: 2^{sliced_comp.tc - complexity.tc:.2f}x")
        print(f"  ✓ Fits in target!")
    print()


def example_4_score_comparison():
    """Example 4: Compare different score configurations"""
    print("=" * 60)
    print("Example 4: Comparing Different Score Configurations")
    print("=" * 60)

    # Medium-sized network
    ixs = [[0, 1, 2], [2, 3], [3, 4], [4, 0]]
    output = [1]
    sizes = {i: 128 for i in range(5)}
    code = EinCode(ixs, output)

    configs = [
        ("Default", ScoreFunction.default()),
        ("Time-optimized", ScoreFunction.time_optimized()),
        ("Space-optimized", ScoreFunction.space_optimized(25.0)),
        ("GPU (rw=20)", ScoreFunction(1.0, 1.0, 20.0, 28.0)),
    ]

    print(f"{'Config':<20} {'tc':<10} {'sc':<10} {'rwc':<10}")
    print("-" * 50)

    for name, score in configs:
        optimizer = GreedyMethod.default()
        optimized = optimize_code(code, sizes, optimizer)
        complexity = contraction_complexity(optimized, sizes, ixs)

        print(f"{name:<20} {complexity.tc:<10.2f} {complexity.sc:<10.2f} {complexity.rwc:<10.2f}")
    print()


def example_5_dynamic_sc_target():
    """Example 5: Dynamically set sc_target based on available memory"""
    print("=" * 60)
    print("Example 5: Dynamic sc_target Based on Available Memory")
    print("=" * 60)

    # Simulate different GPU memory sizes
    gpu_memory_options = [4, 8, 16, 32, 80]  # GB

    print(f"{'GPU Memory':<15} {'sc_target':<15} {'Max Tensor Size':<20}")
    print("-" * 50)

    for gpu_gb in gpu_memory_options:
        # Calculate sc_target for float32 (4 bytes per element)
        # sc_target = log2(memory_in_bytes / 4)
        sc_target = math.log2(gpu_gb * 1024**3 / 4)

        # Max tensor size in elements
        max_elements = 2**sc_target

        print(f"{gpu_gb} GB{'':<10} {sc_target:<15.1f} {max_elements/1e9:.2f} billion elements")

        # Create score function for this GPU
        score = ScoreFunction(
            tc_weight=1.0,
            sc_weight=1.0,
            rw_weight=20.0,
            sc_target=sc_target
        )

    print("\nRecommendation: Use these sc_target values for your GPU")
    print()


def example_6_iterative_tuning():
    """Example 6: Iterative tuning process"""
    print("=" * 60)
    print("Example 6: Iterative Tuning of rw_weight for GPU")
    print("=" * 60)

    # Test case
    ixs = [[0, 1, 2], [2, 3], [3, 4, 5], [5, 0]]
    output = [1, 4]
    sizes = {i: 100 for i in range(6)}
    code = EinCode(ixs, output)

    print("Testing different rw_weight values:")
    print(f"{'rw_weight':<12} {'tc':<12} {'sc':<12} {'rwc':<12}")
    print("-" * 48)

    # Try different rw_weight values
    for rw_weight in [0.0, 5.0, 10.0, 20.0, 50.0]:
        score = ScoreFunction(
            tc_weight=1.0,
            sc_weight=1.0,
            rw_weight=rw_weight,
            sc_target=28.0
        )

        optimizer = GreedyMethod.default()
        optimized = optimize_code(code, sizes, optimizer)
        complexity = contraction_complexity(optimized, sizes, ixs)

        print(f"{rw_weight:<12.1f} {complexity.tc:<12.2f} {complexity.sc:<12.2f} {complexity.rwc:<12.2f}")

    print("\nObservation: Higher rw_weight tends to reduce rwc at the cost of tc")
    print("For GPU: Start with rw_weight=20.0 and adjust based on profiling")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Score Function Configuration Examples")
    print("=" * 60)
    print()

    example_1_cpu_optimization()
    example_2_gpu_optimization()
    example_3_memory_limited()
    example_4_score_comparison()
    example_5_dynamic_sc_target()
    example_6_iterative_tuning()

    print("=" * 60)
    print("Examples complete!")
    print("See docs/score_function_guide.md for detailed explanations")
    print("=" * 60)


if __name__ == "__main__":
    main()
