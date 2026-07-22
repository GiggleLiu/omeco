"""Independent contraction-tree scorer for the autoresearch validator.

Recomputes tc/sc from the tree TOPOLOGY and the original einsum spec only.
Candidate-supplied `eins` metadata inside the tree JSON is never trusted:
each node's input/output index sets are rederived bottom-up, so a tree that
lies about its per-node einsums scores exactly as its topology deserves.

Matches omeco's complexity conventions (log2 scale):
  - per-node tc = sum of log2(size) over the union of both children's labels
  - total tc    = log2(sum of 2^node_tc)  (log-sum-exp base 2)
  - sc          = max over all tensors (leaves and intermediates) of
                  sum of log2(size) over the tensor's labels
"""

import json
import math


class TreeError(ValueError):
    """Raised when the candidate tree violates the contract."""


def _log2sumexp2(vals):
    if not vals:
        return float("-inf")
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log2(sum(2.0 ** (v - m) for v in vals))


def _collect_leaves(node, leaves):
    if node.get("isleaf"):
        leaves.append(node["tensorindex"])
        return
    args = node.get("args")
    if not isinstance(args, list) or len(args) != 2:
        raise TreeError(
            f"internal node must have exactly 2 children, got "
            f"{len(args) if isinstance(args, list) else type(args).__name__}"
        )
    for a in args:
        _collect_leaves(a, leaves)


def score_tree(graph, tree_doc):
    """Validate tree_doc against graph and return (tc, sc, n_nodes).

    graph: dict with 'ixs' (list of label lists), 'iy', 'sizes'
           ({label-as-string: int}).
    tree_doc: parsed writejson output; only 'tree' topology is used.
    """
    ixs = [list(ix) for ix in graph["ixs"]]
    iy = list(graph["iy"])
    log2s = {int(k): math.log2(v) for k, v in graph["sizes"].items()}
    n = len(ixs)

    tree = tree_doc.get("tree")
    if tree is None:
        raise TreeError("tree JSON has no 'tree' key")

    leaves = []
    _collect_leaves(tree, leaves)
    if sorted(leaves) != list(range(n)):
        raise TreeError(
            f"leaves must be a permutation of 0..{n - 1}: got {len(leaves)} "
            f"leaves, missing={sorted(set(range(n)) - set(leaves))[:5]}, "
            f"dup={sorted({x for x in leaves if leaves.count(x) > 1})[:5]}"
        )

    out_labels = set(iy)
    node_tcs = []
    scs = []

    def walk(node, outside_counts):
        """Return this subtree's output label multiset, computing costs.

        outside_counts: for each label, number of occurrences in leaves
        OUTSIDE this subtree (final output counts as one occurrence).
        """
        if node.get("isleaf"):
            labels = ixs[node["tensorindex"]]
            scs.append(sum(log2s.get(l, 0.0) for l in set(labels)))
            return set(labels)

        left, right = node["args"]

        def leaf_label_counts(nd, counts):
            if nd.get("isleaf"):
                for l in set(ixs[nd["tensorindex"]]):
                    counts[l] = counts.get(l, 0) + 1
            else:
                for a in nd["args"]:
                    leaf_label_counts(a, counts)
            return counts

        lcounts = leaf_label_counts(left, {})
        rcounts = leaf_label_counts(right, {})

        lout = walk(left, _merged(outside_counts, rcounts))
        rout = walk(right, _merged(outside_counts, lcounts))

        union = lout | rout
        node_tcs.append(sum(log2s.get(l, 0.0) for l in union))
        out = {l for l in union if outside_counts.get(l, 0) > 0 or l in out_labels}
        scs.append(sum(log2s.get(l, 0.0) for l in out))
        return out

    def _merged(a, b):
        m = dict(a)
        for k, v in b.items():
            m[k] = m.get(k, 0) + v
        return m

    root_out = walk(tree, {})
    if root_out != out_labels:
        raise TreeError(
            f"root output labels {sorted(root_out)[:10]} != requested "
            f"output {sorted(out_labels)[:10]}"
        )

    tc = _log2sumexp2(node_tcs)
    sc = max(scs) if scs else 0.0
    return tc, sc, len(node_tcs)


def score_file(graph_path, tree_path):
    with open(graph_path) as f:
        graph = json.load(f)
    with open(tree_path) as f:
        tree_doc = json.load(f)
    return score_tree(graph, tree_doc)


if __name__ == "__main__":
    import sys

    tc, sc, nodes = score_file(sys.argv[1], sys.argv[2])
    print(json.dumps({"tc": tc, "sc": sc, "nodes": nodes}))
