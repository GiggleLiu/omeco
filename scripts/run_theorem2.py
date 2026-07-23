"""Theorem-2 driver: dyadic-window LB, path-DP strengthening, gap table,
frontier-tree sanity. Reads the profiles produced by window_search.py plus the
026 profiles, and the frontier trees in data/.

Bounds computed (all in log2 bits, closed dim-2 network):
  * DYADIC-WINDOW    : log2 sum_j 2^{b_min(W_j)}, one guaranteed node per window.
  * PATH-DP          : min over larger-child descent sequences (s_{i+1} in
                       [ceil(s_i/2), s_i-1]) of sum_i 2^{b(s_i)} -- the exact
                       strengthening (respects that the balanced band cannot be
                       skipped). >= dyadic-window, still a valid tc lower bound.
  * MAXFORM balanced : min_{n/3<=k<=2n/3} b(k)  (max-form cap, from attempt-022/026).
Variants: CERTIFIED spectral b_spec(k)=lam2 k(n-k)/n; HIGH-CONF empirical
b_emp(k) from window_search.py (achievable cuts, double-count verified).
"""
import os, sys, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphlib import load_graph, fiedler  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "research/benchmark/targets")

# max-form cap per instance (proven / established in attempt-022 & 026):
#   sycamore: 53 (carving-width LB, attempt-022); reg3: ~30 (balanced cut, 026).
MAXFORM_CAP = {"reg3_250": 30, "sycamore_m20": 53}
FRONTIER = {"reg3_250": (39.95, 34), "sycamore_m20": (61.544, 53)}


def windows(n):
    jmax = int(math.floor(math.log2(n))) - 1
    return [(j, int(math.floor(n / 2 ** (j + 1))) + 1, int(math.floor(n / 2 ** j)))
            for j in range(1, jmax + 1)]


def dyadic_lb(bvals, n):
    terms, detail = [], []
    for j, klo, khi in windows(n):
        ks = range(klo, khi + 1)
        kstar = min(ks, key=lambda k: bvals[k])
        terms.append(bvals[kstar]); detail.append((j, klo, khi, bvals[kstar], kstar))
    return math.log2(sum(2.0 ** t for t in terms)), detail


def path_dp(bvals, n):
    """min over descent sequences s_0=n> s_1> ... with s_{i+1} in [ceil(s/2), s-1]
    of sum 2^{b(s_i)}. P[1]=0; P[s]=2^{b[s]}+min_{ceil(s/2)<=s'<=s-1} P[s'].
    Returns (log2 P[n], argmin path sizes)."""
    P = [math.inf] * (n + 1)
    nxt = [None] * (n + 1)
    P[1] = 0.0
    for s in range(2, n + 1):
        lo = (s + 1) // 2  # ceil(s/2)
        best, bs = math.inf, None
        for sp in range(lo, s):
            if P[sp] < best:
                best, bs = P[sp], sp
        P[s] = 2.0 ** bvals[s] + best
        nxt[s] = bs
    # reconstruct path
    path = []
    s = n
    while s is not None and s >= 2:
        path.append(s); s = nxt[s]
    return math.log2(P[n]), path


def maxform_balanced(bvals, n):
    lo, hi = (n + 2) // 3, (2 * n) // 3
    kstar = min(range(lo, hi + 1), key=lambda k: bvals[k])
    return bvals[kstar], kstar


def load_emp_profile(name, n):
    """empirical profile from window_search.json (best_b over searched sizes),
    filled with 026's raw profile elsewhere, symmetrized."""
    ws = json.load(open(os.path.join(ROOT, "data", "window_search.json")))[name]
    prof = ws["full_profile_emp"]  # length n+1, None where unsearched
    b = np.full(n + 1, 1 << 30, dtype=np.int64)
    for k in range(n + 1):
        if prof[k] is not None:
            b[k] = prof[k]
    # fill unsearched sizes from 026 raw profile
    csv = os.path.join(ROOT, "data", f"{name}_profile.csv")
    with open(csv) as fp:
        next(fp)
        for line in fp:
            k, bs, be_raw, be_wm = line.strip().split(",")
            k = int(k); b[k] = min(b[k], int(be_raw))
    b[0] = 0; b[n] = 0
    for k in range(1, n):
        b[k] = min(b[k], b[n - k])
    return b


def larger_child_path(tree_json):
    d = json.load(open(tree_json)); tree = d["tree"]

    def cl(node):
        if node.get("isleaf"):
            return 1
        a, b = node["args"]; return cl(a) + cl(b)
    path = []; node = tree
    while not node.get("isleaf"):
        a, b = node["args"]; na, nb = cl(a), cl(b)
        path.append((na + nb, len(node["eins"]["iy"])))
        node = a if na >= nb else b
    return path


def main():
    rows = []
    detail_out = {}
    for name, gfile, tfile in [("reg3_250", "reg3_250.json", "reg3_tree.json"),
                               ("sycamore_m20", "sycamore_m20.json", "sycamore_tree.json")]:
        n, edges, ixs, iy = load_graph(os.path.join(TARGETS, gfile))
        lam2, vec, lam1, _ = fiedler(n, edges)
        b_spec = np.array([lam2 * k * (n - k) / n for k in range(n + 1)])
        b_emp = load_emp_profile(name, n)

        dy_spec, det_spec = dyadic_lb(b_spec, n)
        dy_emp, det_emp = dyadic_lb(b_emp, n)
        pdp_spec, path_spec = path_dp(b_spec, n)
        pdp_emp, path_emp = path_dp(b_emp, n)
        mf_spec, mfk_spec = maxform_balanced(b_spec, n)
        mf_emp, mfk_emp = maxform_balanced(b_emp, n)

        # frontier tree sanity: larger-child path own boundary sum, and total tc
        path = larger_child_path(os.path.join(ROOT, "data", tfile))
        frontier_pathsum = math.log2(sum(2.0 ** bd for _, bd in path))
        ftc, fsc = FRONTIER[name]

        # bound must be <= frontier tc
        ok_spec = dy_spec <= ftc and pdp_spec <= ftc
        ok_emp = dy_emp <= ftc and pdp_emp <= ftc

        print(f"\n===== {name}  n={n}  lambda2={lam2:.6f}  cap={MAXFORM_CAP[name]} =====")
        print(f"  DYADIC-WINDOW  LB:  spectral(CERT) {dy_spec:7.4f}   empirical(HC) {dy_emp:7.4f}")
        print(f"  PATH-DP        LB:  spectral(CERT) {pdp_spec:7.4f}   empirical(HC) {pdp_emp:7.4f}")
        print(f"  MAXFORM (bal)  LB:  spectral(CERT) {mf_spec:7.4f}@k{mfk_spec}  empirical(HC) {mf_emp:7.4f}@k{mfk_emp}")
        print(f"  frontier: tc={ftc}, sc={fsc}; larger-child path own-boundary sum={frontier_pathsum:.4f}")
        print(f"  cap exceeded? HC dyadic {dy_emp:.3f} vs cap {MAXFORM_CAP[name]}: "
              f"{'YES' if dy_emp > MAXFORM_CAP[name] else 'NO'}; "
              f"HC path-DP {pdp_emp:.3f} vs cap: {'YES' if pdp_emp > MAXFORM_CAP[name] else 'NO'}")
        print(f"  sanity bound<=frontier_tc: spectral {ok_spec}, empirical {ok_emp}")

        rows.append((name, mf_spec, mf_emp, dy_spec, dy_emp, pdp_spec, pdp_emp,
                     MAXFORM_CAP[name], fsc, ftc))
        detail_out[name] = {
            "n": n, "lambda2": lam2,
            "dyadic_spec": dy_spec, "dyadic_emp": dy_emp,
            "pathdp_spec": pdp_spec, "pathdp_emp": pdp_emp,
            "maxform_spec": float(mf_spec), "maxform_spec_k": int(mfk_spec),
            "maxform_emp": int(mf_emp), "maxform_emp_k": int(mfk_emp),
            "cap": MAXFORM_CAP[name], "frontier_tc": ftc, "frontier_sc": fsc,
            "frontier_pathsum": frontier_pathsum,
            "dyadic_detail_emp": [[j, klo, khi, int(bm), int(ks)] for j, klo, khi, bm, ks in det_emp],
            "pathdp_path_emp": path_emp,
            "cap_exceeded_dyadic_emp": bool(dy_emp > MAXFORM_CAP[name]),
            "cap_exceeded_pathdp_emp": bool(pdp_emp > MAXFORM_CAP[name]),
        }

    print("\n\n================= GAP TABLE (log2 bits) =================")
    hdr = ("instance", "maxform-CERT", "maxform-HC", "dyadic-CERT", "dyadic-HC",
           "pathDP-CERT", "pathDP-HC", "cap", "frontier-sc", "frontier-tc")
    print("{:<14} {:>11} {:>10} {:>11} {:>9} {:>11} {:>9} {:>5} {:>11} {:>11}".format(*hdr))
    for r in rows:
        name, mfs, mfe, dys, dye, pds, pde, cap, fsc, ftc = r
        print("{:<14} {:>11.2f} {:>10d} {:>11.2f} {:>9.2f} {:>11.2f} {:>9.2f} {:>5d} {:>11d} {:>11.2f}".format(
            name, mfs, int(mfe), dys, dye, pds, pde, cap, fsc, ftc))

    with open(os.path.join(ROOT, "data", "theorem2_results.json"), "w") as fp:
        json.dump(detail_out, fp, indent=1)
    print("\nwrote data/theorem2_results.json")


if __name__ == "__main__":
    main()
