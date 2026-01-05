"""
omeco - Tensor network contraction order optimization.

This package provides tools for optimizing tensor network contraction orders,
helping minimize computational cost (time and memory) when contracting tensors.

Classes:
    ScoreFunction: Score function for evaluating contraction quality.
        ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.0, sc_target=20.0)

    GreedyMethod: Fast greedy optimizer.
        GreedyMethod(alpha=0.0, temperature=0.0)

    TreeSA: Simulated annealing optimizer.
        TreeSA(ntrials=10, niters=50, betas=None, score=None)
        TreeSA.fast(score=None)

    TreeSASlicer: Slicing optimizer for reducing space complexity.
        TreeSASlicer(ntrials=10, niters=10, betas=None, fixed_slices=None,
                     optimization_ratio=2.0, score=None)
        TreeSASlicer.fast(fixed_slices=None, score=None)

Example:
    >>> from omeco import optimize_code, contraction_complexity, TreeSA, ScoreFunction
    >>>
    >>> # Matrix chain: A[0,1] × B[1,2] × C[2,3] → D[0,3]
    >>> ixs = [[0, 1], [1, 2], [2, 3]]
    >>> out = [0, 3]
    >>> sizes = {0: 100, 1: 200, 2: 50, 3: 100}
    >>>
    >>> # With custom score function
    >>> score = ScoreFunction(sc_target=15.0, sc_weight=2.0)
    >>> tree = optimize_code(ixs, out, sizes, TreeSA(ntrials=5, score=score))
    >>> complexity = contraction_complexity(tree, ixs, sizes)
    >>> print(f"Time: 2^{complexity.tc:.2f}, Space: 2^{complexity.sc:.2f}")

Slicing to reduce memory:
    >>> from omeco import slice_code, sliced_complexity, TreeSASlicer, ScoreFunction
    >>> score = ScoreFunction(sc_target=10.0)
    >>> sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast(score=score))
    >>> print(f"Sliced indices: {sliced.slicing()}")
    >>>
    >>> # With fixed slices (indices that must be sliced)
    >>> slicer = TreeSASlicer(fixed_slices=[1, 2], score=ScoreFunction(sc_target=10.0))
    >>> sliced = slice_code(tree, ixs, sizes, slicer)

Using with PyTorch:
    >>> tree_dict = tree.to_dict()  # Convert to dict for traversal
    >>> # tree_dict structure:
    >>> # - Leaf: {"tensor_index": int}
    >>> # - Node: {"args": [...], "eins": {"ixs": [[int]], "iy": [int]}}
    >>>
    >>> # See examples/pytorch_tensor_network_example.py for complete usage
"""

from omeco._core import (
    # Classes
    NestedEinsum,
    SlicedEinsum,
    ContractionComplexity,
    ScoreFunction,
    GreedyMethod,
    TreeSA,
    TreeSASlicer,
    # Functions
    optimize_code,
    optimize_greedy,
    optimize_treesa,
    contraction_complexity,
    sliced_complexity,
    slice_code,
    uniform_size_dict,
)

__version__ = "0.2.0"
__all__ = [
    "NestedEinsum",
    "SlicedEinsum",
    "ContractionComplexity",
    "ScoreFunction",
    "GreedyMethod",
    "TreeSA",
    "TreeSASlicer",
    "optimize_code",
    "optimize_greedy",
    "optimize_treesa",
    "contraction_complexity",
    "sliced_complexity",
    "slice_code",
    "uniform_size_dict",
]

