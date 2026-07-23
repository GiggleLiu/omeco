"""Sanity checks for the DP and boundary math (run: python3 test_sanity.py)."""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp import dp_exact_int, dp_float  # noqa
from graphlib import boundary, build_adj_list  # noqa


def test_dp_cycle():
    # Cycle C_n: every proper nonempty contiguous arc has boundary exactly 2,
    # and b(k)=2 for all 1<=k<=n-1 (min boundary of any k-subset is 2, achieved
    # by a contiguous arc). So DP: f(1)=0, f(k)=2^2 + min split. With b(k)=2 for
    # 2<=k<=n-1 and b(n)=0. There are n-1 internal nodes; the cheapest tree puts
    # n-2 nodes at boundary 2 and the root at boundary 0.
    # Lower bound value = 4*(n-2) + 1.
    for n in [4, 5, 8, 16]:
        bvals = [0] + [2] * (n - 1) + []
        bvals = [0] * (n + 1)
        for k in range(2, n):
            bvals[k] = 2
        bvals[1] = 0
        bvals[n] = 0
        F, log2F, f = dp_exact_int(bvals)
        expect = 4 * (n - 2) + 1  # (n-2) internal nodes at 2^2, root at 2^0
        assert F == expect, (n, F, expect)
    print("test_dp_cycle: OK")


def test_dp_star_split():
    # b(k)=0 for all k: F should equal number of internal nodes = n-1.
    for n in [3, 7, 10]:
        bvals = [0] * (n + 1)
        F, log2F, f = dp_exact_int(bvals)
        assert F == (n - 1), (n, F)
    print("test_dp_star_split: OK")


def test_dp_matches_bruteforce():
    # brute force min over all binary tree shapes of sum 2^{b(size)} for small n
    import functools
    for n in [5, 6, 7]:
        rng = np.random.default_rng(0)
        bvals = [0] + list(rng.integers(0, 5, size=n))
        bvals[0] = 0
        bvals = [int(x) for x in bvals]
        # brute: partitions
        @functools.lru_cache(maxsize=None)
        def brute(k):
            if k == 1:
                return 0
            best = None
            for j in range(1, k):
                c = brute(j) + brute(k - j)
                if best is None or c < best:
                    best = c
            return (1 << bvals[k]) + best
        F, _, _ = dp_exact_int(bvals)
        assert F == brute(n), (n, F, brute(n), bvals)
    print("test_dp_matches_bruteforce: OK")


def test_spectral_inequality():
    # verify |partial S| >= lambda2 * k(n-k)/n on random subsets of both graphs
    from graphlib import load_graph, fiedler
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for gf in ["reg3_250.json", "sycamore_m20.json"]:
        p = os.path.join(ROOT, "research/benchmark/targets", gf)
        n, edges, ixs, iy = load_graph(p)
        lam2, vec, lam1, _ = fiedler(n, edges)
        rng = np.random.default_rng(1)
        viol = 0
        for _ in range(2000):
            k = rng.integers(1, n)
            sset = rng.choice(n, size=k, replace=False)
            inset = np.zeros(n, dtype=bool)
            inset[sset] = True
            b = boundary(edges, inset)
            lb = lam2 * k * (n - k) / n
            if b < lb - 1e-9:
                viol += 1
        assert viol == 0, (gf, viol)
        print(f"test_spectral_inequality[{gf}]: OK (2000 random subsets, 0 violations)")


if __name__ == "__main__":
    test_dp_cycle()
    test_dp_star_split()
    test_dp_matches_bruteforce()
    test_spectral_inequality()
    print("ALL SANITY TESTS PASSED")
