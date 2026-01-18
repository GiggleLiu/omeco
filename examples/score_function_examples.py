"""
Score Function Configuration Examples

This file demonstrates practical examples of configuring the ScoreFunction
for different hardware and use cases.

See docs/score_function_guide.md for detailed explanations.
"""

from omeco import ScoreFunction
import math


def example_1_cpu_optimization():
    """Example 1: CPU optimization with balanced priorities"""
    print("=" * 60)
    print("Example 1: CPU Optimization (Balanced)")
    print("=" * 60)

    score = ScoreFunction(
        tc_weight=1.0,  # Time matters
        sc_weight=1.0,  # Space matters equally
        rw_weight=0.0,  # Read-write not a bottleneck on CPU
        sc_target=28.0  # ~256MB limit (2^28 floats × 8 bytes)
    )

    print("Configuration:")
    print(f"  tc_weight: {score.tc_weight}")
    print(f"  sc_weight: {score.sc_weight}")
    print(f"  rw_weight: {score.rw_weight} (CPU: I/O not bottleneck)")
    print(f"  sc_target: {score.sc_target} (~256MB)")
    print("\nUse with: TreeSA(score=score)")
    print()


def example_2_gpu_optimization():
    """Example 2: GPU optimization (experimental rw_weight)"""
    print("=" * 60)
    print("Example 2: GPU Optimization (Experimental)")
    print("=" * 60)

    score = ScoreFunction(
        tc_weight=1.0,   # Time still matters
        sc_weight=1.0,   # GPU memory is limited
        rw_weight=10.0,  # Experimental: tune based on profiling (try 1.0 to >20)
        sc_target=30.0   # ~8GB GPU memory (2^30 floats × 4 bytes)
    )

    print("Configuration:")
    print(f"  tc_weight: {score.tc_weight}")
    print(f"  sc_weight: {score.sc_weight}")
    print(f"  rw_weight: {score.rw_weight} (experimental - tune empirically)")
    print(f"  sc_target: {score.sc_target} (~8GB GPU)")
    print("\nAbout rw_weight for GPU:")
    print("  - No established 'best' value - requires empirical tuning")
    print("  - Range: 1.0 to >20 (depends on GPU compute-to-memory-bandwidth ratio)")
    print("  - Start with 1.0, try 10.0, then 20.0, profile actual GPU execution time")
    print("  - Reference: cotengra uses weight=64 for memory writes")
    print()


def example_3_memory_limited():
    """Example 3: Severe memory constraints"""
    print("=" * 60)
    print("Example 3: Memory-Limited Environment")
    print("=" * 60)

    score = ScoreFunction(
        tc_weight=1.0,
        sc_weight=3.0,      # Heavily penalize memory
        rw_weight=0.0,
        sc_target=25.0      # Only ~512MB available
    )

    print("Configuration:")
    print(f"  tc_weight: {score.tc_weight}")
    print(f"  sc_weight: {score.sc_weight} (3x penalty for memory)")
    print(f"  rw_weight: {score.rw_weight}")
    print(f"  sc_target: {score.sc_target} (~512MB)")
    print("\nFor even tighter constraints, use slicing:")
    print("  slicer = TreeSASlicer(score=score)")
    print()


def example_4_score_presets():
    """Example 4: Common score function configurations"""
    print("=" * 60)
    print("Example 4: Common Configurations")
    print("=" * 60)

    # Time-optimized (ignore memory)
    time_score = ScoreFunction(
        tc_weight=1.0,
        sc_weight=0.0,
        rw_weight=0.0,
        sc_target=float('inf')
    )
    print("Time-optimized:")
    print(f"  tc_weight: {time_score.tc_weight}")
    print(f"  sc_weight: {time_score.sc_weight} (ignore memory)")
    print(f"  sc_target: {time_score.sc_target} (infinite)")

    print()

    # Space-optimized (ignore time)
    space_score = ScoreFunction(
        tc_weight=0.0,
        sc_weight=1.0,
        rw_weight=0.0,
        sc_target=25.0
    )
    print("Space-optimized:")
    print(f"  tc_weight: {space_score.tc_weight} (ignore time)")
    print(f"  sc_weight: {space_score.sc_weight}")
    print(f"  sc_target: {space_score.sc_target}")
    print()


def example_5_dynamic_sc_target():
    """Example 5: Dynamically set sc_target based on available memory"""
    print("=" * 60)
    print("Example 5: Dynamic sc_target for Different GPU Sizes")
    print("=" * 60)

    gpu_memory_options = [4, 8, 16, 32, 80]  # GB

    print(f"{'GPU Memory':<15} {'sc_target':<15} {'Max Tensor Size':<20}")
    print("-" * 50)

    for gpu_gb in gpu_memory_options:
        # Calculate sc_target for float32 (4 bytes per element)
        sc_target = math.log2(gpu_gb * 1024**3 / 4)
        max_elements = 2**sc_target

        print(f"{gpu_gb} GB{'':<10} {sc_target:<15.1f} {max_elements/1e9:.2f} billion elements")

    print("\nExample for 8GB GPU:")
    gpu_gb = 8
    sc_target = math.log2(gpu_gb * 1024**3 / 4)
    score = ScoreFunction(
        tc_weight=1.0,
        sc_weight=1.0,
        rw_weight=10.0,     # Experimental: try 1.0 to >20
        sc_target=sc_target
    )
    print(f"  score = ScoreFunction(tc=1.0, sc=1.0, rw=10.0, sc_target={sc_target:.1f})")
    print()


def example_6_comparison_table():
    """Example 6: Compare different configurations"""
    print("=" * 60)
    print("Example 6: Configuration Comparison Table")
    print("=" * 60)

    configs = [
        ("Default", ScoreFunction()),
        ("Time-optimized", ScoreFunction(1.0, 0.0, 0.0, float('inf'))),
        ("Space-optimized", ScoreFunction(0.0, 1.0, 0.0, 25.0)),
        ("GPU (experimental)", ScoreFunction(1.0, 1.0, 10.0, 28.0)),
    ]

    print(f"{'Config':<20} {'tc':<10} {'sc':<10} {'rw':<10} {'target':<10}")
    print("-" * 60)

    for name, score in configs:
        target = "∞" if score.sc_target == float('inf') else f"{score.sc_target:.1f}"
        print(f"{name:<20} {score.tc_weight:<10.1f} {score.sc_weight:<10.1f} {score.rw_weight:<10.1f} {target:<10}")
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
    example_4_score_presets()
    example_5_dynamic_sc_target()
    example_6_comparison_table()

    print("=" * 60)
    print("Examples complete!")
    print("See docs/score_function_guide.md for detailed explanations")
    print("=" * 60)


if __name__ == "__main__":
    main()
