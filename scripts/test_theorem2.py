"""Validation of Theorem-2's combinatorial core.

Checks, on random binary trees over n leaves:
  (A) larger-child descent: s_0=n > s_1 > ... > 1, strictly decreasing;
  (B) halving: s_{i+1} >= ceil(s_i/2)  (larger child of an s-node has >= ceil(s/2));
  (C) window coverage: for every j in 1..floor(log2 n)-1 the path visits >=1 node
      with size in W_j=(n/2^{j+1}, n/2^j]; those nodes are DISTINCT;
  (D) path-DP is a valid relaxation: path_dp(b) <= (larger-child path sum of ANY
      tree) for random profiles b; and dyadic_lb(b) <= path_dp(b);
  (E) path-DP vs brute force over ALL binary-tree shapes for small n.
"""
import math, random, functools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_theorem2 import windows, dyadic_lb, path_dp


def random_tree(n, rng):
    """Random full binary tree as nested tuples of leaf-count; returns root.
    Node = ('leaf',) or ('int', left, right, size)."""
    # build by randomly splitting sizes
    def build(k):
        if k == 1:
            return ('leaf', 1)
        j = rng.randint(1, k - 1)
        L = build(j); R = build(k - j)
        return ('int', L, R, k)
    return build(n)


def node_size(node):
    return node[1] if node[0] == 'leaf' else node[3]


def larger_child_path_sizes(root):
    sizes = []
    node = root
    while node[0] == 'int':
        _, L, R, k = node
        sizes.append(k)
        node = L if node_size(L) >= node_size(R) else R
    return sizes  # sizes of internal nodes on path (root..)


def larger_child_pathsum(root, b):
    return sum(2.0 ** b[s] for s in larger_child_path_sizes(root))


def test_coverage_and_halving():
    rng = random.Random(0)
    for trial in range(3000):
        n = rng.randint(4, 400)
        root = random_tree(n, rng)
        sizes = larger_child_path_sizes(root)
        # (A) strictly decreasing, starts at n, all >=2 (internal)
        assert sizes[0] == n
        for a, b in zip(sizes, sizes[1:]):
            assert b < a, (a, b)
            assert b >= (a + 1) // 2, (a, b)  # (B) halving: >= ceil(a/2)
        # add the terminal leaf size 1 for coverage crossing
        seq = sizes + [1]
        # (C) coverage: each window hit by a DISTINCT path node (by size range)
        wins = windows(n)
        hit_sizes = []
        for j, klo, khi in wins:
            got = [s for s in sizes if klo <= s <= khi]
            assert got, (n, j, klo, khi, sizes)
            hit_sizes.append((j, got[0]))
        # windows disjoint => the chosen sizes are distinct internal nodes
        chosen = [s for _, s in hit_sizes]
        assert len(set(chosen)) == len(chosen)
    print("test_coverage_and_halving: OK (3000 random trees; A,B,C verified)")


def test_pathdp_is_lower_bound():
    rng = random.Random(1)
    for trial in range(2000):
        n = rng.randint(4, 200)
        b = [0] * (n + 1)
        for k in range(2, n):
            b[k] = rng.randint(0, 8)
        b[n] = 0
        # path-DP value
        pdp_log, _ = path_dp(b, n)
        Ppd = 2.0 ** pdp_log
        # dyadic <= path-DP
        dy_log, _ = dyadic_lb(b, n)
        assert dy_log <= pdp_log + 1e-9, (n, dy_log, pdp_log)
        # path-DP <= larger-child path sum of many random trees
        for _ in range(5):
            root = random_tree(n, rng)
            ps = larger_child_pathsum(root, b)
            assert Ppd <= ps + 1e-6, (n, Ppd, ps)
    print("test_pathdp_is_lower_bound: OK (2000 profiles; dyadic<=pathDP<=tree path sum)")


def test_pathdp_vs_bruteforce():
    """For small n, min over ALL tree shapes of the larger-child path sum equals
    path_dp (path-DP minimizes over descent sequences; every reachable sequence
    is realized by some tree, and every tree's path is a valid sequence)."""
    rng = random.Random(2)
    for n in [4, 5, 6, 7, 8, 9]:
        for rep in range(30):
            b = [0] * (n + 1)
            for k in range(2, n):
                b[k] = rng.randint(0, 6)
            b[n] = 0

            @functools.lru_cache(maxsize=None)
            def best_pathsum(k):
                # min over splits of: 2^{b[k]} + best_pathsum(larger child)
                if k == 1:
                    return 0.0
                best = math.inf
                for j in range(1, k):
                    larger = max(j, k - j)
                    best = min(best, best_pathsum(larger))
                return 2.0 ** b[k] + best
            bf = best_pathsum(n)
            best_pathsum.cache_clear()
            pdp_log, _ = path_dp(b, n)
            assert abs(math.log2(bf) - pdp_log) < 1e-9, (n, bf, 2 ** pdp_log)
    print("test_pathdp_vs_bruteforce: OK (n=4..9, path-DP == min over all tree shapes)")


if __name__ == "__main__":
    test_coverage_and_halving()
    test_pathdp_is_lower_bound()
    test_pathdp_vs_bruteforce()
    print("ALL THEOREM-2 SANITY TESTS PASSED")
