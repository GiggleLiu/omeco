from omeco import (
    TreeSA, GreedyMethod, optimize_code, contraction_complexity,
    simplify_then_optimize, waist_refine,
)

CHAIN_IXS = [[0, 1], [1, 2], [2, 3], [3, 4]]
CHAIN_OUT = [0, 4]
CHAIN_SIZES = {i: 2 for i in range(5)}


def test_treesa_pipeline_defaults():
    opt = TreeSA()
    assert opt.preprocess is True
    assert opt.surgery_budget == 0.0
    opt2 = TreeSA(preprocess=False, surgery_budget=1.5)
    assert opt2.preprocess is False
    assert opt2.surgery_budget == 1.5


def test_treesa_default_keeps_all_leaves():
    tree = optimize_code(CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, TreeSA(ntrials=1, niters=5))
    assert tree.leaf_count() == 4


def test_simplify_then_optimize_reports_shrink():
    tree, report = simplify_then_optimize(CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, GreedyMethod())
    assert tree.leaf_count() == 4
    assert report.n_original == 4
    assert report.n_reduced <= report.n_original


def test_waist_refine_never_worse():
    seed = optimize_code(CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, GreedyMethod())
    seed_tc = contraction_complexity(seed, CHAIN_IXS, CHAIN_SIZES).tc
    refined, report = waist_refine(seed, CHAIN_IXS, CHAIN_OUT, CHAIN_SIZES, 0.5)
    tc = contraction_complexity(refined, CHAIN_IXS, CHAIN_SIZES).tc
    assert tc <= seed_tc + 1e-9
    assert report.surgery_calls >= 0
