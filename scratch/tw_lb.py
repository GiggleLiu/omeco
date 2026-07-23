"""Certified treewidth LOWER bounds for L(G).

All three quantities below are proven lower bounds on treewidth:

  1. delta-degeneracy: tw(H) >= degeneracy(H) = max over subgraphs of min degree.
     (Every subgraph is a minor; min degree of any minor lower-bounds tw.)
  2. MMD  (Minor-Min-Width / MMW+): tw(H) >= max min-degree encountered while
     repeatedly contracting a minimum-degree vertex into a neighbour. Contracting
     edges yields minors; min degree of a minor lower-bounds tw. Ref: Gogate &
     Dechter 2004, "A complete anytime algorithm for treewidth".
  3. MMD+(least-c): same, but the contracted neighbour is the one of MINIMUM
     degree (the standard "least-c" tie/neighbour rule), often a stronger LB.

Each returns an integer L with the guarantee tw(L(G)) >= L, hence (Markov-Shi)
the optimal contraction width w satisfies w = tw(L(G)) + 1 >= L + 1.
"""
import pathlib
HERE = pathlib.Path(__file__).resolve().parent

def load_adj(path):
    adj = {}
    for line in open(path):
        a, b = line.split()
        a, b = int(a), int(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj

def degeneracy(adj):
    adj = {u: set(v) for u, v in adj.items()}
    best = 0
    # peel min-degree vertex repeatedly; degeneracy = max min-degree at removal
    import heapq
    deg = {u: len(v) for u, v in adj.items()}
    while adj:
        u = min(adj, key=lambda x: len(adj[x]))
        best = max(best, len(adj[u]))
        for w in adj[u]:
            adj[w].discard(u)
        del adj[u]
    return best

def mmd(adj, least_c=False):
    """Minor-min-width. If least_c, contract min-degree vertex into its
    minimum-degree neighbour (MMD+ least-c); else into an arbitrary neighbour."""
    adj = {u: set(v) for u, v in adj.items()}
    best = 0
    while len(adj) > 1:
        # min-degree vertex
        u = min(adj, key=lambda x: len(adj[x]))
        d = len(adj[u])
        best = max(best, d)
        if d == 0:
            del adj[u]
            continue
        # choose neighbour
        if least_c:
            v = min(adj[u], key=lambda x: len(adj[x]))
        else:
            v = next(iter(adj[u]))
        # contract u into v (merge u's neighbourhood into v)
        for w in adj[u]:
            if w != v:
                adj[w].discard(u)
                adj[w].add(v)
                adj[v].add(w)
        adj[v].discard(u)
        adj[v].discard(v)
        del adj[u]
    return best

if __name__ == "__main__":
    adj = load_adj(HERE / "LG.edgelist")
    dgn = degeneracy(adj)
    m1 = mmd(adj, least_c=False)
    m2 = mmd(adj, least_c=True)
    print(f"L(G): |V|={len(adj)}")
    print(f"degeneracy (delta-LB)         tw >= {dgn}")
    print(f"MMD  (minor-min-width)        tw >= {m1}")
    print(f"MMD+ (least-c)                tw >= {m2}")
    best = max(dgn, m1, m2)
    print(f"=> certified tw(L(G)) >= {best}")
    print(f"=> contraction width w = tw(L(G))+1 >= {best+1}")
    print(f"=> tc >= {best+1}  (single-contraction / width floor)")
