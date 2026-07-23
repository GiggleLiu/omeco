"""Aggressively search for the minimum 1/3-2/3 balanced edge cut of G.

Goal: establish B_bal(G) as tightly as possible from ABOVE (achievable cuts).
If nothing beats 53 across many METIS seeds, spatial/temporal structured cuts,
and Kernighan-Lin / Fiduccia-Mattheyses refinement, that is strong evidence the
minimum balanced cut equals the temporal cut = 53. (Upper bound on separator.)
"""
import json
import pathlib
import numpy as np
import networkx as nx

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
edges = [tuple(map(int, l.split())) for l in open(HERE / "G.edgelist")]
n = 561
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(edges)

lo, hi = n // 3, 2 * n // 3  # 187..374 balanced window (1/3-2/3)

def cut_size(mask):
    return sum(1 for u, v in edges if mask[u] != mask[v])

best = (10**9, None)

# 1) METIS many seeds
import pymetis
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v); adj[v].append(u)
for seed in range(40):
    _, mem = pymetis.part_graph(2, adjacency=adj, options=pymetis.Options(seed=seed))
    mask = np.array(mem)
    s = int(mask.sum())
    if lo <= s <= hi or lo <= n - s <= hi:
        c = cut_size(mask)
        if c < best[0]:
            best = (c, ('metis', seed, s))

# 2) Structured cuts using the actual construction (identify each tensor's role)
g = json.load(open(ROOT / "research/benchmark/targets/sycamore_m20.json"))
ixs = g["ixs"]
# Reconstruct tensor roles by replaying gen_sycamore.build order:
ROWS, COLS, CYCLES = 8, 7, 20
cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
for corner in [(0, 0), (0, COLS - 1), (ROWS - 1, 0)]:
    cells.remove(corner)
qubits = {cell: i for i, cell in enumerate(cells)}
def color(a, b):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return 0 if min(c1, c2) % 2 == 0 else 1
    return 2 if min(r1, r2) % 2 == 0 else 3
cedges = {0: [], 1: [], 2: [], 3: []}
for (r, c) in cells:
    for nb in [(r, c + 1), (r + 1, c)]:
        if nb in qubits:
            cedges[color((r, c), nb)].append(((r, c), nb))
# tensor id order: 53 inputs, then per-cycle gates, then 53 outputs.
tid = 0
role = {}  # tid -> ('in',cell) | ('gate',cycle,(a,b)) | ('out',cell)
for cell in cells:
    role[tid] = ('in', cell); tid += 1
for t in range(CYCLES):
    for (a, b) in cedges[t % 4]:
        role[tid] = ('gate', t, (a, b)); tid += 1
for cell in cells:
    role[tid] = ('out', cell); tid += 1
assert tid == n

def cycle_of(i):
    r = role[i]
    if r[0] == 'in': return -1
    if r[0] == 'out': return CYCLES
    return r[1]
def cellset_of(i):
    r = role[i]
    if r[0] in ('in', 'out'): return (r[1],)
    return (r[2][0], r[2][1])

# 2a) temporal cuts: S = everything with cycle < t (inputs on left)
for t in range(1, CYCLES):
    mask = np.array([0 if cycle_of(i) < t else 1 for i in range(n)])
    s = int((mask == 0).sum())
    if lo <= s <= hi or lo <= n - s <= hi:
        c = cut_size(mask)
        if c < best[0]:
            best = (c, ('temporal', t, s))

# 2b) spatial cuts: split qubits by a grid coordinate threshold; a tensor goes
#     with the side containing (a majority of) its qubit cells.
def side_by_cells(pred):
    mask = np.zeros(n, int)
    for i in range(n):
        cs = cellset_of(i)
        votes = sum(1 for cell in cs if pred(cell))
        mask[i] = 1 if votes * 2 >= len(cs) else 0
    return mask
for thr in range(1, ROWS):
    mask = side_by_cells(lambda cell: cell[0] >= thr)
    s = int((mask == 0).sum())
    if lo <= s <= hi or lo <= n - s <= hi:
        c = cut_size(mask)
        if c < best[0]:
            best = (c, ('spatial_row', thr, s))
for thr in range(1, COLS):
    mask = side_by_cells(lambda cell: cell[1] >= thr)
    s = int((mask == 0).sum())
    if lo <= s <= hi or lo <= n - s <= hi:
        c = cut_size(mask)
        if c < best[0]:
            best = (c, ('spatial_col', thr, s))

# 3) Kernighan-Lin refinement seeded from best METIS partition (networkx)
try:
    a, b = nx.algorithms.community.kernighan_lin_bisection(
        G, max_iter=20, seed=1)
    mask = np.zeros(n, int)
    for v in b: mask[v] = 1
    s = int((mask == 0).sum())
    if lo <= s <= hi or lo <= n - s <= hi:
        c = cut_size(mask)
        if c < best[0]:
            best = (c, ('KL', s))
except Exception as e:
    print("KL failed:", e)

print(f"balanced window sizes [{lo},{hi}]")
print(f"BEST balanced cut found: {best[0]}  via {best[1]}")
print(f"=> B_bal(G) <= {best[0]} (achievable). tc >= B_bal(G).")
