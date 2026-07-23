"""Diagnostics: reconstruct the DP-optimal tree shape and its dominant terms;
characterize the frontier tree's own boundary profile."""
import os
import sys
import math
import json
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "data/results.json")))


def dp_with_shape(bvals):
    n = len(bvals) - 1
    f = [0] * (n + 1)
    split = [None] * (n + 1)
    for k in range(2, n + 1):
        best, bj = None, None
        for j in range(1, k // 2 + 1):
            c = f[j] + f[k - j]
            if best is None or c < best:
                best, bj = c, j
        f[k] = (1 << int(bvals[k])) + best
        split[k] = bj
    # reconstruct size multiset of internal nodes
    sizes = []
    stack = [n]
    while stack:
        k = stack.pop()
        if k == 1:
            continue
        sizes.append(k)
        j = split[k]
        stack.append(j)
        stack.append(k - j)
    return f[n], sizes


for name in ["reg3_250", "sycamore_m20"]:
    d = r[name]
    n = d["n"]
    b_emp = d["b_emp"]
    F, sizes = dp_with_shape(b_emp)
    log2F = math.log2(F) if F.bit_length() < 1000 else None
    # dominant terms: 2^{b(size)} for each internal node size
    terms = sorted((b_emp[s] for s in sizes), reverse=True)
    print(f"\n=== {name}: DP-optimal empirical shape ===")
    print(f"  log2 F = {log2F:.4f};  #internal nodes = {len(sizes)}")
    print(f"  top-10 node boundaries in DP tree: {terms[:10]}")
    # how much does the single largest term contribute
    S = sum(2.0 ** t for t in terms)
    print(f"  largest-term share of the sum: {2.0**terms[0]/S:.4f}"
          f"  (log2 of the rest = {math.log2(S - 2.0**terms[0]) if S>2.0**terms[0] else float('-inf'):.3f})")
    # size distribution near balance
    balanced = [s for s in sizes if n/3 <= s <= 2*n/3]
    print(f"  #internal nodes in balanced window [n/3,2n/3]: {len(balanced)}"
          f"  their boundaries: {sorted(b_emp[s] for s in balanced)}")

    # frontier tree's OWN profile
    nodes = d["tree_nodes"]
    bnd = sorted((nd["boundary"] for nd in nodes), reverse=True)
    Sown = sum(2.0 ** nd["boundary"] for nd in nodes)
    print(f"  --- frontier tree OWN boundary profile ---")
    print(f"  tree sum-form(own) = {math.log2(Sown):.4f}; tree tc = {d['tree_tc']:.4f}")
    print(f"  top-15 tree node boundaries: {bnd[:15]}")
    # count nodes with boundary within 3 of the peak
    peak = bnd[0]
    near = sum(1 for x in bnd if x >= peak - 3)
    print(f"  peak boundary = {peak}; #nodes within 3 of peak = {near}")
    # histogram of tree node boundaries >= peak-8
    c = Counter(x for x in bnd if x >= peak - 10)
    print(f"  histogram (boundary:count) for boundary>=peak-10: {dict(sorted(c.items(), reverse=True))}")
