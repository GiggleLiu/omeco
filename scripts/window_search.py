"""Dedicated per-window minimum-boundary search + cut verification.

For each dyadic window W_j = (n/2^{j+1}, n/2^j], we want the TIGHTEST empirical
estimate of b_min(W_j) = min_{k in W_j} b(k). Because heuristic cuts are
achievable, any cut we find at size k gives b(k) <= (that boundary), so the
smaller the cut we find, the LOWER (more honest) our estimate of b_min(W_j).

Sources per window:
  1. 026's harvested profile (region-grow + fiedler + FM + frontier-tree cuts)
  2. Fresh multi-start region growing to each target size in the window
  3. FM swap refinement at fixed size (size-preserving)
  4. Frontier-tree nested cuts whose size falls in the window
  5. METIS-style recursive bisection projections (fiedler prefix at window sizes)

Every reported cut is VERIFIED by two independent boundary counts
(edge-iteration vs adjacency matvec x^T(D-A)x). We record an explicit vertex
set (a certificate) achieving the window minimum.
"""
import os, sys, math, json, random
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphlib import load_graph, fiedler, build_adj_list, boundary, adjacency  # noqa
from empirical_profile import grow_profile, fiedler_sweep, swap_refine, _grow_to_size  # noqa
from tree_profile import parse_tree  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "research/benchmark/targets")


def windows(n):
    jmax = int(math.floor(math.log2(n))) - 1
    out = []
    for j in range(1, jmax + 1):
        lo = n / 2 ** (j + 1)
        hi = n / 2 ** j
        out.append((j, int(math.floor(lo)) + 1, int(math.floor(hi))))
    return out


def boundary_matvec(A, inS):
    x = inS.astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    within = 0.5 * (x @ (A @ x))
    volS = deg[inS].sum()
    return int(round(volS - 2 * within))


def verify(edges, A, inS):
    b1 = boundary(edges, inS)
    b2 = boundary_matvec(A, inS)
    assert b1 == b2, (b1, b2, int(inS.sum()))
    return b1


def main():
    out_all = {}
    for name, gfile, tfile in [("reg3_250", "reg3_250.json", "reg3_tree.json"),
                               ("sycamore_m20", "sycamore_m20.json", "sycamore_tree.json")]:
        n, edges, ixs, iy = load_graph(os.path.join(TARGETS, gfile))
        adj = build_adj_list(n, edges)
        A = adjacency(n, edges)
        lam2, vec, lam1, _ = fiedler(n, edges)
        rng = random.Random(2027)
        deg = np.array([len(a) for a in adj])

        # best[k] = (boundary, membership tuple) achieving smallest boundary at size k
        best_b = np.full(n + 1, 1 << 30, dtype=np.int64)
        best_set = [None] * (n + 1)

        def consider(inS):
            k = int(inS.sum())
            if k <= 0 or k >= n:
                return
            b = verify(edges, A, inS)
            if b < best_b[k]:
                best_b[k] = b
                best_set[k] = inS.copy()
            # complement gives size n-k, same boundary
            kc = n - k
            if b < best_b[kc]:
                best_b[kc] = b
                best_set[kc] = (~inS).copy()

        # --- source 1: fiedler sweep prefixes ---
        order = np.argsort(vec)
        inS = np.zeros(n, dtype=bool)
        for i, v in enumerate(order):
            inS[v] = True
            if 1 <= i + 1 <= n - 1:
                consider(inS.copy())

        # --- source 2: region growing to sampled target sizes, many seeds, then FM.
        # b_min over a window sits at its cheap (off-center) end for expander-like
        # profiles, so we sample densely near each window's LEFT edge and a few
        # interior points, rather than every size.
        wins = windows(n)
        target_sizes = set()
        for j, klo, khi in wins:
            width = khi - klo + 1
            # left edge densely, then a coarse sweep across the window
            for k in range(klo, min(klo + 6, khi + 1)):
                target_sizes.add(k)
            for frac in (0.25, 0.5, 0.75, 1.0):
                target_sizes.add(min(khi, klo + int(frac * (width - 1))))
        low_deg_seeds = list(np.argsort(deg)[:12])
        rand_seeds = [rng.randrange(n) for _ in range(18)]
        grow_seeds = low_deg_seeds + rand_seeds
        for k in sorted(target_sizes):
            for s in grow_seeds:
                inS = _grow_to_size(n, adj, k, int(s), rng)
                consider(inS)
            # FM refine only the best-so-far at this size (cheaper, effective)
            if best_set[k] is not None:
                inS2, _ = swap_refine(n, adj, best_set[k].copy(), passes=30, cand=200)
                consider(inS2)

        # --- source 3: frontier-tree nested cuts ---
        _, nodes, leafsets = parse_tree(os.path.join(ROOT, "data", tfile), want_sets=True)
        for size, sset in leafsets:
            inS = np.zeros(n, dtype=bool)
            for v in sset:
                inS[v] = True
            consider(inS)

        # --- source 4: FM refine the window-left-edge best sets harder ---
        for j, klo, khi in wins:
            for k in list(range(klo, min(klo + 6, khi + 1))):
                if best_set[k] is not None:
                    inS2, _ = swap_refine(n, adj, best_set[k].copy(), passes=50, cand=300)
                    consider(inS2)

        # compute per-window minima with certificates.
        # Only sizes actually searched are reliable; restrict argmin to those.
        searched = sorted(target_sizes)
        win_report = []
        for j, klo, khi in wins:
            ks = [k for k in searched if klo <= k <= khi]
            if not ks:
                ks = [klo]
            kbest = min(ks, key=lambda k: best_b[k])
            bmin = int(best_b[kbest])
            cert = best_set[kbest]
            cert_vertices = sorted(int(i) for i in np.where(cert)[0]) if cert is not None else None
            # spectral certified min over window
            spec_min = min(lam2 * k * (n - k) / n for k in ks)
            spec_k = min(ks, key=lambda k: lam2 * k * (n - k) / n)
            win_report.append({
                "j": j, "k_lo": klo, "k_hi": khi,
                "b_emp_min": bmin, "argmin_k": int(kbest),
                "b_spec_min": spec_min, "spec_argmin_k": int(spec_k),
                "cert_size": int(cert.sum()) if cert is not None else None,
                "cert_boundary_recount": int(verify(edges, A, cert)) if cert is not None else None,
                "cert_vertices": cert_vertices,
            })

        # dyadic LB values
        emp_terms = [w["b_emp_min"] for w in win_report]
        spec_terms = [w["b_spec_min"] for w in win_report]
        lb_emp = math.log2(sum(2.0 ** t for t in emp_terms))
        lb_spec = math.log2(sum(2.0 ** t for t in spec_terms))

        print(f"\n===== {name}  n={n}  lambda2={lam2:.6f} =====")
        print(f"{'j':>2} {'k_lo':>5} {'k_hi':>5} | {'b_emp_min':>9} {'@k':>5} {'certsz':>6} {'recount':>7} | {'b_spec_min':>10} {'@k':>5}")
        for w in win_report:
            print(f"{w['j']:>2} {w['k_lo']:>5} {w['k_hi']:>5} | {w['b_emp_min']:>9d} {w['argmin_k']:>5} "
                  f"{w['cert_size']:>6} {w['cert_boundary_recount']:>7} | {w['b_spec_min']:>10.3f} {w['spec_argmin_k']:>5}")
        print(f"[CERTIFIED spectral] dyadic-window LB(tc) = {lb_spec:.4f}")
        print(f"[HC empirical]       dyadic-window LB(tc) = {lb_emp:.4f}")

        out_all[name] = {
            "n": n, "m": len(edges), "lambda2": lam2,
            "windows": win_report,
            "lb_dyadic_spec": lb_spec, "lb_dyadic_emp": lb_emp,
            "full_profile_emp": [int(x) if x < (1 << 29) else None for x in best_b],
        }

    with open(os.path.join(ROOT, "data", "window_search.json"), "w") as fp:
        json.dump(out_all, fp, indent=1)
    print("\nwrote data/window_search.json")


if __name__ == "__main__":
    main()
