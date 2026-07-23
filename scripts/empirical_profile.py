"""Heuristic estimate of the isoperimetric profile b(k) = min_{|S|=k} |partial S|.

Because heuristic cuts are ACHIEVABLE, they upper-bound the true minimum:
b_heur(k) >= b_true(k). We drive b_heur down toward b_true with many restarts
and refinement, then apply a conservative window-min so the estimate leans
toward b_true. The resulting profile bound is high-confidence, NOT certified.
"""
import heapq
import random
import numpy as np
from graphlib import build_adj_list


def grow_profile(n, adj, seed_vertex, rng):
    """Min-boundary greedy region growing from seed_vertex.
    Returns array bnd[1..n] where bnd[k] = boundary of the size-k set grown.
    Adding vertex v changes boundary by deg(v) - 2*inS(v)."""
    inS = np.zeros(n, dtype=bool)
    inS_count = np.zeros(n, dtype=np.int32)  # neighbors already in S
    deg = np.array([len(a) for a in adj])
    bnd = np.zeros(n + 1, dtype=np.int64)
    # heap of (delta, tiebreak, vertex) for frontier candidates
    heap = []
    boundary = 0
    order = 0

    def add_vertex(v):
        nonlocal boundary, order
        delta = deg[v] - 2 * inS_count[v]
        boundary += delta
        inS[v] = True
        for w in adj[v]:
            if not inS[w]:
                inS_count[w] += 1
                d = deg[w] - 2 * inS_count[w]
                order += 1
                heapq.heappush(heap, (d, rng.random(), w))

    add_vertex(seed_vertex)
    bnd[1] = boundary
    size = 1
    while size < n:
        # pop until we get a fresh, still-outside vertex with up-to-date delta
        cand = None
        while heap:
            d, tb, w = heapq.heappop(heap)
            if inS[w]:
                continue
            cur = deg[w] - 2 * inS_count[w]
            if cur != d:  # stale key, reinsert with current value
                heapq.heappush(heap, (cur, tb, w))
                continue
            cand = w
            break
        if cand is None:
            # disconnected: pick any outside vertex (adds its full degree)
            outside = np.where(~inS)[0]
            cand = int(outside[0])
        add_vertex(cand)
        size += 1
        bnd[size] = boundary
    return bnd


def fiedler_sweep(n, edges, vec):
    """Boundary of prefix sets under the Fiedler-vector ordering."""
    order = np.argsort(vec)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    bnd = np.zeros(n + 1, dtype=np.int64)
    # sweep: track boundary incrementally by adding vertices in order
    inS = np.zeros(n, dtype=bool)
    adj = None
    boundary = 0
    from graphlib import build_adj_list as _bal
    adj = _bal(n, edges)
    deg = np.array([len(a) for a in adj])
    inS_count = np.zeros(n, dtype=np.int32)
    for i, v in enumerate(order, start=1):
        delta = deg[v] - 2 * inS_count[v]
        boundary += delta
        inS[v] = True
        for w in adj[v]:
            if not inS[w]:
                inS_count[w] += 1
        bnd[i] = boundary
    return bnd


def swap_refine(n, adj, in_set, passes=30, cand=200):
    """Size-preserving steepest-descent swap local search with incrementally
    maintained external-minus-internal degree d[v].

    Removing a in S reduces cut by d[a] = out(a) - in(a).
    Adding  w notin S reduces cut by -d[w] = in(w) - out(w).
    Swap (remove a, add w): cut reduction = d[a] - d[w] - 2*[a~w].
    Returns (membership bool array, boundary)."""
    inS = in_set.copy()
    adjset = [set(a) for a in adj]

    def compute_d():
        d = np.zeros(n, dtype=np.int64)
        for v in range(n):
            insd = sum(1 for w in adj[v] if inS[w])
            d[v] = (len(adj[v]) - insd) - insd
        return d

    def boundary():
        return sum(1 for u in range(n) if inS[u] for w in adj[u] if not inS[w])

    b = boundary()
    for _ in range(passes):
        d = compute_d()
        ins = np.where(inS)[0]
        outs = np.where(~inS)[0]
        ca = ins[np.argsort(-d[ins])[:cand]]
        cw = outs[np.argsort(d[outs])[:cand]]
        best_gain = 0
        move = None
        for a in ca:
            da = int(d[a])
            for w in cw:
                gain = da - int(d[w]) - (2 if w in adjset[a] else 0)
                if gain > best_gain:
                    best_gain = gain
                    move = (int(a), int(w))
        if move is None:
            break
        a, w = move
        inS[a] = False
        inS[w] = True
        b -= best_gain
    return inS, int(b)


def build_profile(n, edges, vec, n_seeds=400, refine_grid=40, seed=12345,
                  harvest_sets=None):
    """Return raw heuristic profile b_raw[1..n] (elementwise best over all
    heuristic sources). harvest_sets: optional list of (size, python-set) of
    ACTUAL cuts (e.g. from a good contraction tree) that seed refinement --
    these give real achievable boundaries at balanced sizes."""
    rng = random.Random(seed)
    adj = build_adj_list(n, edges)
    best = np.full(n + 1, 1 << 30, dtype=np.int64)
    best[0] = 0
    best[1] = min(len(a) for a in adj)  # min degree = best size-1 boundary
    # source 1: fiedler sweep
    fs = fiedler_sweep(n, edges, vec)
    best = np.minimum(best, fs)
    # source 2: many min-boundary region growths from random + low-degree seeds
    deg = np.array([len(a) for a in adj])
    low_deg = list(np.argsort(deg)[: min(n, 60)])
    seeds = low_deg + [rng.randrange(n) for _ in range(n_seeds)]
    for s in seeds:
        bnd = grow_profile(n, adj, int(s), rng)
        best = np.minimum(best, bnd)
    best[n] = 0  # whole set: boundary 0

    refine_seeds = {}  # size -> boolean membership array to refine

    # source 3: harvest real cuts from a provided contraction tree
    if harvest_sets:
        for size, sset in harvest_sets:
            inS = np.zeros(n, dtype=bool)
            for v in sset:
                inS[v] = True
            bnd = int(boundary_of(adj, inS))
            if bnd < best[size]:
                best[size] = bnd
            refine_seeds.setdefault(size, inS)
            # complement (same boundary, size n-size)
            refine_seeds.setdefault(n - size, ~inS)

    # source 4: grid seeds from region growth
    ks = np.unique(np.linspace(2, n - 2, refine_grid).astype(int))
    for k in ks:
        if k not in refine_seeds:
            refine_seeds[k] = _grow_to_size(n, adj, int(k), int(low_deg[0]), rng)

    # refine every seeded size with steepest-descent swaps
    for k, inS in refine_seeds.items():
        _, b = swap_refine(n, adj, inS.copy(), passes=40, cand=250)
        if b < best[k]:
            best[k] = b

    # enforce symmetry b(k)=b(n-k)
    for k in range(1, n + 1):
        best[k] = min(best[k], best[n - k])
    return best


def boundary_of(adj, inS):
    return sum(1 for u in range(len(adj)) if inS[u] for w in adj[u] if not inS[w])


def _grow_to_size(n, adj, k, seed_vertex, rng):
    inS = np.zeros(n, dtype=bool)
    inS[seed_vertex] = True
    frontier = set(adj[seed_vertex])
    inS_count = np.zeros(n, dtype=np.int32)
    for w in adj[seed_vertex]:
        inS_count[w] += 1
    deg = np.array([len(a) for a in adj])
    size = 1
    while size < k:
        if not frontier:
            outside = np.where(~inS)[0]
            v = int(outside[0])
        else:
            v = min(frontier, key=lambda x: deg[x] - 2 * inS_count[x])
        inS[v] = True
        frontier.discard(v)
        for w in adj[v]:
            if not inS[w]:
                inS_count[w] += 1
                frontier.add(w)
        size += 1
    return inS


def window_min(b_raw, w=2):
    """Conservative window-min: b_conservative(k) = min_{|j-k|<=w} b_raw(j),
    but never below the trivially-valid endpoints. Pushes the estimate toward
    b_true (down), leaning conservative for use as a lower bound."""
    n = len(b_raw) - 1
    out = np.zeros_like(b_raw)
    for k in range(0, n + 1):
        lo = max(0, k - w)
        hi = min(n, k + w)
        out[k] = b_raw[lo:hi + 1].min()
    out[0] = 0
    out[n] = 0
    return out
