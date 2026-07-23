"""Main driver: compute all bound variants for both instances and emit
data files + a gap table. Run from the scripts/ directory."""
import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphlib import load_graph, fiedler, boundary  # noqa
from dp import dp_float, dp_exact_int  # noqa
from empirical_profile import build_profile, window_min  # noqa
from tree_profile import parse_tree, tc_from_nodes, sumform_from_nodes_boundary  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "research/benchmark/targets")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

INSTANCES = [
    ("reg3_250", "reg3_250.json", "reg3_tree.json", 39.95, 34),
    ("sycamore_m20", "sycamore_m20.json", "sycamore_tree.json", 61.544, 53),
]


def maxform_balanced(bvals):
    """Rigorous max-form (balanced-separator) bound: every binary tree has an
    internal node with n/3 <= |S| <= 2n/3, whose width >= b(|S|). So
    sc >= min_{n/3<=k<=2n/3} b(k). Returns (value, argmin k)."""
    n = len(bvals) - 1
    lo, hi = (n + 2) // 3, (2 * n) // 3
    vals = [(bvals[k], k) for k in range(lo, hi + 1)]
    v, k = min(vals)
    return v, k


def profile_peak(bvals):
    n = len(bvals) - 1
    m = max(bvals[k] for k in range(1, n))
    return m


def main():
    results = {}
    for name, gfile, tfile, frontier_tc, frontier_sc in INSTANCES:
        print(f"\n===== {name} =====")
        n, edges, ixs, iy = load_graph(os.path.join(TARGETS, gfile))
        m = len(edges)
        print(f"n={n} vertices, m={m} edges")

        # ---- frontier tree's OWN observed profile (parse first, to harvest cuts) ----
        tpath = os.path.join(DATA, tfile)
        total_leaves, nodes, leafsets = parse_tree(tpath, want_sets=True)
        assert total_leaves == n, (total_leaves, n)
        harvest = [(sz, sset) for sz, sset in leafsets]

        # ---- spectral (certified) ----
        lam2, vec, lam1, _ = fiedler(n, edges)
        print(f"lambda1={lam1:.6e} (should be ~0), lambda2(algebraic conn.)={lam2:.6f}")
        b_spec = np.zeros(n + 1)
        for k in range(0, n + 1):
            b_spec[k] = lam2 * k * (n - k) / n  # certified lower bound on b(k)
        F_spec, lb_spec, _ = dp_float(b_spec)
        print(f"[CERTIFIED] spectral sum-form LB(tc) = {lb_spec:.4f}")
        mf_spec_val, mf_spec_k = maxform_balanced(b_spec)
        print(f"[CERTIFIED] spectral max-form (balanced-sep) = {mf_spec_val:.4f} at k={mf_spec_k}")

        # ---- empirical (high-confidence, not certified) ----
        b_raw = build_profile(n, edges, vec, n_seeds=500, refine_grid=60,
                              harvest_sets=harvest)
        b_emp = window_min(b_raw, w=2)
        # exact integer DP on the conservative empirical profile
        F_emp, lb_emp, _ = dp_exact_int([int(x) for x in b_emp])
        print(f"[EMPIRICAL] sum-form structural LB(tc) = {lb_emp:.4f}")
        mf_emp_val, mf_emp_k = maxform_balanced(b_emp)
        peak_emp = int(profile_peak(b_emp))
        print(f"[EMPIRICAL] max-form (balanced-sep) = {mf_emp_val} at k={mf_emp_k}; profile peak={peak_emp}")
        # also raw (non-window) empirical for reference
        F_raw, lb_raw, _ = dp_exact_int([int(x) for x in b_raw])
        tc_tree = tc_from_nodes(nodes)
        sumform_tree = sumform_from_nodes_boundary(nodes)
        maxform_tree = max(nd["boundary"] for nd in nodes)  # max observed width
        maxcost_tree = max(nd["cost"] for nd in nodes)      # sc of the tree
        print(f"tree: tc={tc_tree:.4f} (frontier target {frontier_tc}), "
              f"sc(max cost)={maxcost_tree} (target {frontier_sc})")
        print(f"tree sum-form on OWN boundary profile = {sumform_tree:.4f}")
        print(f"tree max observed boundary width = {maxform_tree}")

        results[name] = {
            "n": n, "m": m, "lambda2": lam2, "lambda1": lam1,
            "frontier_tc": frontier_tc, "frontier_sc": frontier_sc,
            "lb_spec_sumform": lb_spec,
            "spec_maxform_balanced": mf_spec_val, "spec_maxform_k": mf_spec_k,
            "lb_emp_sumform": lb_emp,
            "emp_maxform_balanced": mf_emp_val, "emp_maxform_k": mf_emp_k,
            "emp_profile_peak": peak_emp,
            "lb_emp_raw_sumform": lb_raw,
            "tree_tc": tc_tree, "tree_sc": maxcost_tree,
            "tree_sumform_own": sumform_tree,
            "tree_max_boundary": maxform_tree,
            "b_spec": b_spec.tolist(),
            "b_emp": [int(x) for x in b_emp],
            "b_raw": [int(x) for x in b_raw],
            "tree_nodes": nodes,
        }
        # write per-instance profile CSV
        with open(os.path.join(DATA, f"{name}_profile.csv"), "w") as fp:
            fp.write("k,b_spec,b_emp_raw,b_emp_windowmin\n")
            for k in range(0, n + 1):
                fp.write(f"{k},{b_spec[k]:.6f},{int(b_raw[k])},{int(b_emp[k])}\n")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    with open(os.path.join(DATA, "results.json"), "w") as fp:
        json.dump(_clean(results), fp, indent=1)

    # ---- gap table ----
    print("\n\n========== GAP TABLE ==========")
    hdr = ("instance", "maxform-spec(CERT)", "maxform-emp(HC)",
           "sumform-spec(CERT)", "sumform-emp(HC)", "frontier sc", "frontier tc")
    print("{:<14} {:>18} {:>16} {:>18} {:>16} {:>12} {:>12}".format(*hdr))
    for name, *_ in INSTANCES:
        r = results[name]
        print("{:<14} {:>18.3f} {:>16d} {:>18.3f} {:>16.3f} {:>12d} {:>12.3f}".format(
            name, r["spec_maxform_balanced"], r["emp_maxform_balanced"],
            r["lb_spec_sumform"], r["lb_emp_sumform"], r["frontier_sc"],
            r["frontier_tc"]))
    print("\nwrote", os.path.join(DATA, "results.json"))


if __name__ == "__main__":
    main()
