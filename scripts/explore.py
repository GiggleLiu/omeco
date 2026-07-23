"""Quick exploration: dyadic-window bound from 026's existing profiles.
Reads the *_profile.csv (k, b_spec, b_emp_raw, b_emp_windowmin) and computes
    LB = log2 sum_j 2^{b_min(W_j)},  W_j = (n/2^{j+1}, n/2^j], j=1..floor(log2 n)-1
for both the certified spectral profile and the HC empirical raw profile.
Also prints per-window k-ranges and argmin. Also computes the frontier tree's
larger-child descent path and its own boundary profile for sanity.
"""
import os, sys, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphlib import load_graph, fiedler, build_adj_list, boundary  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "research/benchmark/targets")


def windows(n):
    """Guaranteed dyadic windows W_j = (n/2^{j+1}, n/2^j], j=1..floor(log2 n)-1.
    Returns list of (j, k_lo, k_hi) integer inclusive ranges."""
    jmax = int(math.floor(math.log2(n))) - 1
    out = []
    for j in range(1, jmax + 1):
        lo = n / 2 ** (j + 1)
        hi = n / 2 ** j
        k_lo = int(math.floor(lo)) + 1      # smallest integer > lo
        k_hi = int(math.floor(hi))          # largest integer <= hi
        out.append((j, k_lo, k_hi))
    return out


def dyadic_lb(bvals, n):
    """log2 sum_j 2^{b_min(W_j)}, plus per-window detail."""
    terms = []
    detail = []
    for j, klo, khi in windows(n):
        ks = range(klo, khi + 1)
        bmin = min(bvals[k] for k in ks)
        kstar = min(ks, key=lambda k: bvals[k])
        terms.append(bmin)
        detail.append((j, klo, khi, bmin, kstar))
    S = sum(2.0 ** t for t in terms)
    return math.log2(S), detail


def larger_child_path(tree_json, n):
    """Descent path always to child with more leaves. Returns list of
    (size, boundary) for internal nodes on the path (root..down)."""
    d = json.load(open(tree_json))
    tree = d["tree"]

    def count_leaves(node):
        if node.get("isleaf"):
            return 1
        a, b = node["args"]
        return count_leaves(a) + count_leaves(b)

    path = []
    node = tree
    while not node.get("isleaf"):
        eins = node["eins"]
        iy = eins["iy"]
        # size of this node
        # boundary = len(iy)
        a, b = node["args"]
        # recompute sizes (cache would be nicer but fine)
        na = count_leaves(a)
        nb = count_leaves(b)
        size = na + nb
        path.append((size, len(iy)))
        node = a if na >= nb else b
    return path


def main():
    for name, gfile, tfile in [("reg3_250", "reg3_250.json", "reg3_tree.json"),
                               ("sycamore_m20", "sycamore_m20.json", "sycamore_tree.json")]:
        n, edges, ixs, iy = load_graph(os.path.join(TARGETS, gfile))
        # load profiles
        csv = os.path.join(ROOT, "data", f"{name}_profile.csv")
        b_spec = np.zeros(n + 1)
        b_emp = np.zeros(n + 1, dtype=np.int64)
        with open(csv) as fp:
            next(fp)
            for line in fp:
                k, bs, be_raw, be_wm = line.strip().split(",")
                k = int(k)
                b_spec[k] = float(bs)
                b_emp[k] = int(be_raw)
        print(f"\n===== {name}  n={n}  =====")
        print(f"windows (j, k_lo..k_hi):")
        lb_spec, det_spec = dyadic_lb(b_spec, n)
        lb_emp, det_emp = dyadic_lb(b_emp, n)
        print(f"{'j':>2} {'k_lo':>5} {'k_hi':>5} | {'b_spec_min':>10} {'@k':>5} | {'b_emp_min':>9} {'@k':>5}")
        for (j, klo, khi, bmin_s, ks_s), (_, _, _, bmin_e, ks_e) in zip(det_spec, det_emp):
            print(f"{j:>2} {klo:>5} {khi:>5} | {bmin_s:>10.3f} {ks_s:>5} | {bmin_e:>9d} {ks_e:>5}")
        print(f"[CERTIFIED spectral] dyadic-window LB(tc) = {lb_spec:.4f}")
        print(f"[HC empirical(raw)]  dyadic-window LB(tc) = {lb_emp:.4f}")

        # frontier tree descent path sanity
        path = larger_child_path(os.path.join(ROOT, "data", tfile), n)
        Spath = sum(2.0 ** bnd for size, bnd in path)
        print(f"frontier larger-child path: {len(path)} internal nodes; "
              f"log2 sum 2^boundary(path) = {math.log2(Spath):.4f}")
        # show which path node lands in each window and its boundary
        print(f"  path node boundaries by window:")
        for j, klo, khi in windows(n):
            innodes = [(sz, bd) for sz, bd in path if klo <= sz <= khi]
            if innodes:
                bmax = max(bd for sz, bd in innodes)
                print(f"    W{j}[{klo}..{khi}]: path sizes/bnd {innodes[:4]}"
                      f"{'...' if len(innodes)>4 else ''} maxbnd={bmax}")


if __name__ == "__main__":
    main()
