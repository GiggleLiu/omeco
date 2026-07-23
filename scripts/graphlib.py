"""Shared utilities for the isoperimetric-profile bound analysis (attempt-026).

A tensor-network JSON with an empty output (iy=[]) whose every index appears on
exactly two tensors is an ordinary (multi)graph G: vertices = tensors, edges =
indices. All index dimensions are 2, so a contraction's log2-cost equals the
number of distinct indices involved, and the boundary |partial S| of a vertex
subset S equals the number of edges leaving S (each dim-2 index = 1 bit).
"""
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import laplacian


def load_graph(path):
    """Return (n, edges, ixs, iy). edges is a list of (u, v) vertex pairs, one
    per index that appears on exactly two tensors. Verifies the network is
    closed and dim-2."""
    d = json.load(open(path))
    ixs = d["ixs"]
    iy = d["iy"]
    sizes = d.get("sizes", {})
    if sizes:
        dims = set(int(v) for v in sizes.values())
        assert dims == {2}, f"non dim-2 indices: {dims}"
    n = len(ixs)
    # map each index label -> list of tensor ids carrying it
    occ = {}
    for t, labels in enumerate(ixs):
        for lab in labels:
            occ.setdefault(lab, []).append(t)
    for lab in iy:
        occ.setdefault(lab, []).append(-1)  # -1 = external/output
    edges = []
    for lab, ts in occ.items():
        assert len(ts) == 2, f"index {lab} appears {len(ts)} times (not closed)"
        u, v = ts
        assert u != -1 and v != -1 and u != v, f"index {lab} open or self-loop: {ts}"
        edges.append((u, v))
    return n, edges, ixs, iy


def adjacency(n, edges):
    """Sparse symmetric adjacency (integer multiplicity for multi-edges)."""
    rows, cols, vals = [], [], []
    for u, v in edges:
        rows += [u, v]
        cols += [v, u]
        vals += [1, 1]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def fiedler(n, edges):
    """Return (lambda2, fiedler_vector) computed from the graph Laplacian.

    lambda2 is the algebraic connectivity (second-smallest Laplacian
    eigenvalue). Computed densely for exactness when n is small; the value is
    certified in the sense that it is a numerically-converged eigenvalue of the
    exact integer Laplacian."""
    A = adjacency(n, edges)
    L = laplacian(A).astype(float)
    # dense symmetric eigdecomposition -- exact & fully converged for our sizes
    Ld = L.toarray()
    w, V = np.linalg.eigh(Ld)
    # w sorted ascending; w[0] ~ 0 (connected). lambda2 = w[1]
    lam2 = float(w[1])
    vec = V[:, 1]
    lam1 = float(w[0])
    return lam2, vec, lam1, Ld


def boundary(edges, in_set):
    """Number of edges with exactly one endpoint in the boolean membership
    array in_set (indexed by vertex id)."""
    b = 0
    for u, v in edges:
        if in_set[u] != in_set[v]:
            b += 1
    return b


def build_adj_list(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj
