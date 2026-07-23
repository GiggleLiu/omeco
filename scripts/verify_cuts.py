"""Independently verify that reported b_emp(k) values are ACHIEVABLE (genuine
cuts): reconstruct a concrete size-k set with the heuristic, then recount its
boundary two independent ways (edge iteration vs sparse adjacency matvec)."""
import os
import sys
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from graphlib import load_graph, fiedler, build_adj_list, boundary, adjacency  # noqa
from empirical_profile import _grow_to_size, swap_refine  # noqa
from tree_profile import parse_tree  # noqa
import random


def boundary_matrix(A, inS):
    """|partial S| via adjacency matrix: sum over edges (u<v) with membership
    differing = (1/2) * x^T (D - A) x where x is +-1? Use: cut = sum_{u in S}
    (deg(u)) - 2*edges_within(S). edges_within = (1/2) sum_{u,v in S} A[u,v]."""
    x = inS.astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    within = 0.5 * (x @ (A @ x))
    volS = deg[inS].sum()
    return int(round(volS - 2 * within))


def main():
    for gf, tf in [("reg3_250.json", "reg3_tree.json"),
                   ("sycamore_m20.json", "sycamore_tree.json")]:
        p = os.path.join(ROOT, "research/benchmark/targets", gf)
        n, edges, ixs, iy = load_graph(p)
        adj = build_adj_list(n, edges)
        A = adjacency(n, edges)
        lam2, vec, lam1, _ = fiedler(n, edges)
        rng = random.Random(7)
        print(f"\n=== {gf}: verifying achievable cuts (two independent counts) ===")
        # 1) verify the frontier tree's OWN nodes are genuine cuts
        _, nodes, leafsets = parse_tree(os.path.join(ROOT, "data", tf), want_sets=True)
        maxb = 0
        okcount = 0
        for size, sset in leafsets:
            inS = np.zeros(n, dtype=bool)
            for v in sset:
                inS[v] = True
            b1 = boundary(edges, inS)
            b2 = boundary_matrix(A, inS)
            assert b1 == b2, (size, b1, b2)
            okcount += 1
            maxb = max(maxb, b1)
        print(f"  frontier tree: {okcount} internal-node cuts, two counts agree; "
              f"max boundary = {maxb}")
        # 2) reconstruct balanced cuts and double-count
        for k in [n // 4, n // 3, n // 2, 2 * n // 3]:
            inS = _grow_to_size(n, adj, k, int(np.argmin([len(a) for a in adj])), rng)
            inS, _ = swap_refine(n, adj, inS, passes=40, cand=250)
            b1 = boundary(edges, inS)
            b2 = boundary_matrix(A, inS)
            spec = lam2 * inS.sum() * (n - inS.sum()) / n
            assert b1 == b2, (k, b1, b2)
            assert b1 >= spec - 1e-9
            print(f"  k={int(inS.sum()):4d}: achievable |partial S|={b1:3d} "
                  f"(both counts agree; certified spectral LB={spec:.2f})")


if __name__ == "__main__":
    main()
