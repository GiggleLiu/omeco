"""Parse a treesa_tuned output tree and compute, per internal node v:
  - |S_v|        : number of leaf tensors under v
  - |partial S_v|: boundary edges of S_v in G (= number of labels on the
                   intermediate tensor produced at v, since dims are 2 and the
                   network is closed) -- read directly from the einsum 'iy'
  - cost_v       : log2-cost = |labels(A) union labels(B)| = |ixs[0] U ixs[1]|
The identity cost_v = |partial S_v| + |E(A,B)| is verified numerically.
"""
import json


def parse_tree(path, want_sets=False):
    d = json.load(open(path))
    tree = d["tree"]
    nodes = []  # list of dicts: size, boundary, cost, eab
    leafsets = []  # list of (size, frozenset(tensor ids)) per internal node

    def rec(node):
        if node.get("isleaf"):
            return frozenset([node["tensorindex"]])
        a, b = node["args"]
        sa = rec(a)
        sb = rec(b)
        eins = node["eins"]
        ixs = eins["ixs"]
        iy = eins["iy"]
        labA = set(ixs[0])
        labB = set(ixs[1])
        union = labA | labB
        cost = len(union)          # log2-cost (dims=2)
        boundary = len(iy)         # |partial S_v|
        eab = len(labA & labB)     # edges contracted here = |E(A,B)|
        s = sa | sb
        size = len(s)
        nodes.append({"size": size, "boundary": boundary, "cost": cost,
                      "eab": eab})
        if want_sets:
            leafsets.append((size, s))
        # sanity: cost == boundary + eab
        assert cost == boundary + eab, (cost, boundary, eab)
        return s

    root = rec(tree)
    total = len(root)
    if want_sets:
        return total, nodes, leafsets
    return total, nodes


def tc_from_nodes(nodes):
    import math
    s = sum(2.0 ** nd["cost"] for nd in nodes)
    return math.log2(s)


def sumform_from_nodes_boundary(nodes):
    """log2 sum_v 2^{|partial S_v|} -- the sum-form value using the tree's OWN
    observed boundary profile (a valid lower bound on that tree's tc)."""
    import math
    s = sum(2.0 ** nd["boundary"] for nd in nodes)
    return math.log2(s)
