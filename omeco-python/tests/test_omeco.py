"""Tests for omeco Python bindings."""

import pytest
from omeco import (
    GreedyMethod,
    TreeSA,
    TreeSASlicer,
    ScoreFunction,
    optimize_code,
    optimize_greedy,
    optimize_treesa,
    contraction_complexity,
    sliced_complexity,
    slice_code,
    SlicedEinsum,
    uniform_size_dict,
)


def test_optimize_greedy_basic():
    """Test basic greedy optimization."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    assert tree is not None
    assert tree.is_binary()
    assert tree.leaf_count() == 2


def test_optimize_greedy_chain():
    """Test greedy optimization on a chain."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = {0: 10, 1: 20, 2: 20, 3: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    assert tree.leaf_count() == 3
    assert tree.depth() >= 1


def test_optimize_treesa():
    """Test TreeSA optimization."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_treesa(ixs, out, sizes, TreeSA.fast())
    assert tree is not None
    assert tree.is_binary()


def test_contraction_complexity():
    """Test complexity computation."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    complexity = contraction_complexity(tree, ixs, sizes)
    
    assert complexity.tc > 0
    assert complexity.sc > 0
    assert complexity.flops() > 0
    assert complexity.peak_memory() > 0


def test_sliced_einsum():
    """Test sliced einsum."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    sliced = SlicedEinsum([1], tree)
    
    assert sliced.num_slices() == 1
    assert 1 in sliced.slicing()
    
    complexity = sliced_complexity(sliced, ixs, sizes)
    assert complexity.sc > 0


def test_uniform_size_dict():
    """Test uniform size dictionary creation."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    
    sizes = uniform_size_dict(ixs, out, 16)
    assert sizes[0] == 16
    assert sizes[1] == 16
    assert sizes[2] == 16


def test_greedy_method_params():
    """Test GreedyMethod with parameters."""
    opt = GreedyMethod(alpha=0.5, temperature=1.0)
    
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes, opt)
    assert tree is not None


def test_treesa_config():
    """Test TreeSA configuration with ScoreFunction."""
    score = ScoreFunction(sc_target=10.0)
    opt = TreeSA(ntrials=2, score=score)

    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 4, 1: 8, 2: 4}

    tree = optimize_treesa(ixs, out, sizes, opt)
    assert tree is not None


def test_to_dict_leaf():
    """Test to_dict for a single tensor (leaf node)."""
    ixs = [[0, 1]]
    out = [0, 1]
    sizes = {0: 10, 1: 20}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Single tensor should be a leaf
    assert "tensor_index" in d
    assert d["tensor_index"] == 0


def test_to_dict_binary():
    """Test to_dict for a binary contraction."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Should be a node with args and eins
    assert "args" in d
    assert "eins" in d
    assert len(d["args"]) == 2
    
    # Check eins structure
    assert "ixs" in d["eins"]
    assert "iy" in d["eins"]
    assert len(d["eins"]["ixs"]) == 2
    
    # Children should be leaves
    for arg in d["args"]:
        assert "tensor_index" in arg


def test_to_dict_chain():
    """Test to_dict for a chain of contractions."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = {0: 10, 1: 20, 2: 20, 3: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Should be a node
    assert "args" in d
    assert "eins" in d
    
    # Count leaves by recursion
    def count_leaves(node):
        if "tensor_index" in node:
            return 1
        return sum(count_leaves(arg) for arg in node["args"])
    
    assert count_leaves(d) == 3


def test_to_dict_indices():
    """Test that to_dict preserves correct indices."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Output should match
    assert d["eins"]["iy"] == out
    
    # Input indices should be the original tensor indices
    input_ixs = d["eins"]["ixs"]
    assert input_ixs == ixs


def test_optimize_code_default():
    """Test optimize_code with default optimizer (GreedyMethod)."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_code(ixs, out, sizes)
    assert tree is not None
    assert tree.is_binary()
    assert tree.leaf_count() == 2


def test_optimize_code_greedy():
    """Test optimize_code with explicit GreedyMethod."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = {0: 10, 1: 20, 2: 20, 3: 10}
    
    tree = optimize_code(ixs, out, sizes, GreedyMethod())
    assert tree.leaf_count() == 3


def test_optimize_code_treesa():
    """Test optimize_code with TreeSA."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_code(ixs, out, sizes, TreeSA.fast())
    assert tree is not None
    assert tree.is_binary()


def test_optimize_code_treesa_configured():
    """Test optimize_code with configured TreeSA."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = {0: 10, 1: 20, 2: 20, 3: 10}

    opt = TreeSA().with_ntrials(2).with_niters(10)
    tree = optimize_code(ixs, out, sizes, opt)
    assert tree.leaf_count() == 3


def test_slice_code_basic():
    """Test basic slice_code functionality."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = uniform_size_dict(ixs, out, 64)

    tree = optimize_code(ixs, out, sizes, GreedyMethod())
    slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=10.0))

    sliced = slice_code(tree, ixs, sizes, slicer)
    assert sliced is not None
    assert sliced.num_slices() >= 0


def test_slice_code_reduces_space():
    """Test that slice_code reduces space complexity."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = uniform_size_dict(ixs, out, 64)

    tree = optimize_code(ixs, out, sizes, GreedyMethod())
    original = contraction_complexity(tree, ixs, sizes)

    slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=8.0))
    sliced = slice_code(tree, ixs, sizes, slicer)

    sliced_comp = sliced_complexity(sliced, ixs, sizes)

    # Space complexity should be reduced or at least not increased
    assert sliced_comp.sc <= original.sc + 1.0


def test_slice_code_default_slicer():
    """Test slice_code with default slicer."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 16, 1: 32, 2: 16}

    tree = optimize_code(ixs, out, sizes, GreedyMethod())
    sliced = slice_code(tree, ixs, sizes)

    assert sliced is not None


def test_treesaslicer_config():
    """Test TreeSASlicer configuration with ScoreFunction."""
    score = ScoreFunction(sc_target=15.0)
    slicer = TreeSASlicer(ntrials=5, niters=8, score=score)

    # Test getters
    assert slicer.ntrials == 5
    assert slicer.niters == 8
    assert slicer.score.sc_target == 15.0

    # Test repr
    repr_str = repr(slicer)
    assert "TreeSASlicer" in repr_str


def test_treesaslicer_fast():
    """Test TreeSASlicer.fast() static method."""
    slicer = TreeSASlicer.fast()

    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 8, 1: 16, 2: 8}

    tree = optimize_code(ixs, out, sizes, GreedyMethod())
    sliced = slice_code(tree, ixs, sizes, slicer)

    assert sliced is not None


# ============== New tests for ScoreFunction ==============


def test_score_function_default():
    """Test ScoreFunction with default parameters."""
    score = ScoreFunction()
    assert score.tc_weight == 1.0
    assert score.sc_weight == 1.0
    assert score.rw_weight == 0.0
    assert score.sc_target == 20.0


def test_score_function_custom():
    """Test ScoreFunction with custom parameters."""
    score = ScoreFunction(tc_weight=2.0, sc_weight=0.5, rw_weight=0.1, sc_target=15.0)
    assert score.tc_weight == 2.0
    assert score.sc_weight == 0.5
    assert score.rw_weight == 0.1
    assert score.sc_target == 15.0


def test_score_function_repr():
    """Test ScoreFunction __repr__."""
    score = ScoreFunction(sc_target=10.0)
    repr_str = repr(score)
    assert "ScoreFunction" in repr_str
    assert "sc_target=10" in repr_str


def test_score_function_with_treesa():
    """Test ScoreFunction with TreeSA optimizer."""
    score = ScoreFunction(tc_weight=1.0, sc_weight=2.0, sc_target=10.0)
    opt = TreeSA(ntrials=2, niters=10, score=score)

    assert opt.score.tc_weight == 1.0
    assert opt.score.sc_weight == 2.0
    assert opt.score.sc_target == 10.0

    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 4, 1: 8, 2: 4}

    tree = optimize_code(ixs, out, sizes, opt)
    assert tree is not None


def test_score_function_with_treesaslicer():
    """Test ScoreFunction with TreeSASlicer."""
    score = ScoreFunction(sc_target=10.0, sc_weight=2.0)
    slicer = TreeSASlicer(ntrials=2, niters=5, score=score)

    assert slicer.score.sc_target == 10.0
    assert slicer.score.sc_weight == 2.0


# ============== New tests for GreedyMethod getters ==============


def test_greedy_method_getters():
    """Test GreedyMethod getter properties."""
    opt = GreedyMethod(alpha=0.5, temperature=1.0)
    assert opt.alpha == 0.5
    assert opt.temperature == 1.0


def test_greedy_method_default():
    """Test GreedyMethod default values."""
    opt = GreedyMethod()
    assert opt.alpha == 0.0
    assert opt.temperature == 0.0


# ============== New tests for TreeSA constructor ==============


def test_treesa_constructor():
    """Test TreeSA constructor with all parameters."""
    score = ScoreFunction(sc_target=15.0)
    betas = [0.1, 0.5, 1.0, 2.0]
    opt = TreeSA(ntrials=5, niters=20, betas=betas, score=score)

    assert opt.ntrials == 5
    assert opt.niters == 20
    assert opt.betas == betas
    assert opt.score.sc_target == 15.0


def test_treesa_fast_with_score():
    """Test TreeSA.fast() with custom score."""
    score = ScoreFunction(sc_target=10.0)
    opt = TreeSA.fast(score=score)

    assert opt.score.sc_target == 10.0


def test_treesa_getters():
    """Test TreeSA getter properties."""
    opt = TreeSA(ntrials=3, niters=15)
    assert opt.ntrials == 3
    assert opt.niters == 15
    assert len(opt.betas) > 0


# ============== New tests for TreeSASlicer constructor ==============


def test_treesaslicer_constructor():
    """Test TreeSASlicer constructor with all parameters."""
    score = ScoreFunction(sc_target=20.0)
    betas = [14.0, 14.5, 15.0]
    slicer = TreeSASlicer(
        ntrials=5,
        niters=8,
        betas=betas,
        fixed_slices=[1, 2],
        optimization_ratio=3.0,
        score=score,
    )

    assert slicer.ntrials == 5
    assert slicer.niters == 8
    assert slicer.betas == betas
    assert slicer.fixed_slices == [1, 2]
    assert slicer.optimization_ratio == 3.0
    assert slicer.score.sc_target == 20.0


def test_treesaslicer_getters():
    """Test TreeSASlicer getter properties."""
    slicer = TreeSASlicer(ntrials=3, niters=6, optimization_ratio=2.5)
    assert slicer.ntrials == 3
    assert slicer.niters == 6
    assert slicer.optimization_ratio == 2.5
    assert len(slicer.betas) > 0
    assert slicer.fixed_slices == []


# ============== New tests for fixed_slices ==============


def test_fixed_slices_basic():
    """Test TreeSASlicer with fixed_slices."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = uniform_size_dict(ixs, out, 16)

    tree = optimize_code(ixs, out, sizes, GreedyMethod())

    # Specify that index 1 must be sliced
    slicer = TreeSASlicer.fast(fixed_slices=[1])
    sliced = slice_code(tree, ixs, sizes, slicer)

    assert sliced is not None
    # Index 1 should be in the slicing
    assert 1 in sliced.slicing()


def test_fixed_slices_multiple():
    """Test TreeSASlicer with multiple fixed_slices."""
    ixs = [[0, 1], [1, 2], [2, 3], [3, 4]]
    out = [0, 4]
    sizes = uniform_size_dict(ixs, out, 16)

    tree = optimize_code(ixs, out, sizes, GreedyMethod())

    # Specify that indices 1 and 2 must be sliced
    slicer = TreeSASlicer.fast(fixed_slices=[1, 2])
    sliced = slice_code(tree, ixs, sizes, slicer)

    assert sliced is not None
    slicing = sliced.slicing()
    assert 1 in slicing
    assert 2 in slicing


def test_fixed_slices_with_score():
    """Test fixed_slices combined with ScoreFunction."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = uniform_size_dict(ixs, out, 32)

    tree = optimize_code(ixs, out, sizes, GreedyMethod())

    score = ScoreFunction(sc_target=8.0)
    slicer = TreeSASlicer(
        ntrials=2,
        niters=5,
        fixed_slices=[1],
        score=score,
    )
    sliced = slice_code(tree, ixs, sizes, slicer)

    assert sliced is not None
    assert 1 in sliced.slicing()
