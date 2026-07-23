"""Build the tensor graph G and its line graph L(G) from the instance JSON.

G: vertices = tensors (561), edges = indices (963). Since every index in the
instance appears in EXACTLY two tensors (verified: index-occurrence
distribution is {2: 963}), the tensor network is an ordinary graph, not a
hypergraph. All bond dims are 2, so a contraction cost equals the number of
distinct indices in the union of the two operands' index sets.

Writes:
  scratch/G.edgelist       tensor graph, one "u v" per line (0-indexed vertices)
  scratch/LG.gr            line graph in PACE .gr format (1-indexed) for tw tools
  scratch/LG.edgelist      line graph edge list (0-indexed)
"""
import json
import itertools
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
g = json.load(open(ROOT / "research/benchmark/targets/sycamore_m20.json"))
ixs = g["ixs"]
n = len(ixs)  # 561 tensors

# Map each index label -> list of tensor ids that carry it.
idx_to_tensors = {}
for tid, ix in enumerate(ixs):
    for lab in ix:
        idx_to_tensors.setdefault(lab, []).append(tid)

# Sanity: every index in exactly 2 tensors (closed network, degree-2 indices).
bad = {lab: ts for lab, ts in idx_to_tensors.items() if len(ts) != 2}
assert not bad, f"non-degree-2 indices present: {len(bad)}"
labels = sorted(idx_to_tensors)  # 963 indices == edges of G
lab_to_eid = {lab: e for e, lab in enumerate(labels)}

# --- G: tensor graph ---
G_edges = []  # (u,v) per index
for lab in labels:
    u, v = idx_to_tensors[lab]
    G_edges.append((u, v))
assert len(G_edges) == len(labels)

# --- L(G): line graph. Vertices = indices (963). Two indices adjacent iff they
# share a tensor. For a tensor of degree d, all C(d,2) pairs of its indices are
# adjacent in L(G). ---
LG_adj = set()
for tid, ix in enumerate(ixs):
    eids = [lab_to_eid[lab] for lab in ix]
    for a, b in itertools.combinations(sorted(eids), 2):
        LG_adj.add((a, b))
LG_edges = sorted(LG_adj)
nL = len(labels)

# Write outputs
(HERE / "G.edgelist").write_text(
    "\n".join(f"{u} {v}" for u, v in G_edges) + "\n")
(HERE / "LG.edgelist").write_text(
    "\n".join(f"{a} {b}" for a, b in LG_edges) + "\n")

# PACE .gr format for LG: header "p tw <nverts> <nedges>", then 1-indexed edges.
with open(HERE / "LG.gr", "w") as f:
    f.write(f"p tw {nL} {len(LG_edges)}\n")
    for a, b in LG_edges:
        f.write(f"{a+1} {b+1}\n")

# Degree stats
from collections import Counter
degG = Counter()
for u, v in G_edges:
    degG[u] += 1
    degG[v] += 1
print("G: |V|=", n, "|E|=", len(G_edges),
      "deg dist:", dict(Counter(degG.values())))
degL = Counter()
for a, b in LG_edges:
    degL[a] += 1
    degL[b] += 1
print("L(G): |V|=", nL, "|E|=", len(LG_edges),
      "min/max deg:", min(degL.values()), max(degL.values()))
