"""Cotengra reference runner, attempt-contract compatible.

usage: run_cotengra.py <graph.json> <budget_ms> <out.json> [sa]

Runs cotengra's HyperOptimizer (greedy + kahypar) under the wall-clock
budget; with the optional `sa` flag, spends the second half refining the
best tree with cotengra's subtree simulated annealing. Emits the omeco
writejson format (eins bodies are dummies — the validator recomputes from
topology). Trees above the instance sc cap are never emitted; both
'flops' and 'combo' objectives are tried to find the best cap-feasible
tree.
"""

import json
import os
import sys
import time

t0 = time.monotonic()
graph_path, budget_ms, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
do_sa = len(sys.argv) > 4 and sys.argv[4] == "sa"
budget_s = budget_ms / 1000.0

with open(graph_path) as f:
    g = json.load(f)
inputs = [tuple(map(int, ix)) for ix in g["ixs"]]
output = tuple(map(int, g["iy"]))
size_dict = {int(k): v for k, v in g["sizes"].items()}
cap = float("inf")  # v2.1 pure-tc objective: sc unbounded

import cotengra as ctg  # noqa: E402


def emit(tree):
    """Serialize a cotengra ContractionTree to omeco writejson format.

    Handles both node schemes: int ids (leaves 0..N-1, internal fresh ids,
    root at tree.root) and frozenset-of-leaves keys."""
    node_map = {}
    n = len(inputs)
    for i in range(n):
        node_map[i] = {"isleaf": True, "tensorindex": i}
        node_map[frozenset([i])] = node_map[i]
    root_key = None
    for parent, left, right in tree.traverse():
        node = {"isleaf": False,
                "args": [node_map[left], node_map[right]],
                "eins": {"ixs": [[], []], "iy": []}}
        node_map[parent] = node
        root_key = parent
    if hasattr(tree, "root") and tree.root in node_map:
        root_key = tree.root
    elif frozenset(range(n)) in node_map:
        root_key = frozenset(range(n))
    doc = {"label-type": "Int64", "inputs": [list(ix) for ix in inputs],
           "output": list(output), "tree": node_map[root_key]}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, out_path)


best = None  # (tc_log2, tree)


def consider(tree):
    global best
    w = tree.contraction_width()  # log2 largest intermediate == sc
    if w > cap:
        return
    tc = float(tree.contraction_cost(log=2))
    if best is None or tc < best[0]:
        best = (tc, tree)
        emit(tree)


def remaining():
    return budget_s * 0.92 - (time.monotonic() - t0)


def main():
    # safety net: random-greedy sampling until something cap-feasible exists
    consider(ctg.array_contract_tree(inputs, output, size_dict, optimize="greedy"))
    rg = ctg.HyperOptimizer(methods=["greedy"], minimize="combo",
                            max_time=5, max_repeats=256, parallel=False,
                            progbar=False, on_trial_error="ignore")
    try:
        consider(rg.search(inputs, output, size_dict))
    except Exception as e:  # noqa: BLE001
        print(f"greedy warmup failed: {e}", file=sys.stderr)

    modes = ["flops", "combo"] if not do_sa else ["flops"]
    phase1 = remaining() * (0.45 if do_sa else 0.95)
    for minimize in modes:
        if remaining() <= 1:
            break
        opt = ctg.HyperOptimizer(
            methods=["greedy", "kahypar"], minimize=minimize,
            max_time=max(1.0, phase1 / len(modes)),
            max_repeats=4096, parallel=False, progbar=False,
            on_trial_error="ignore")
        try:
            consider(opt.search(inputs, output, size_dict))
        except Exception as e:  # noqa: BLE001
            print(f"hyper({minimize}) failed: {e}", file=sys.stderr)

    if do_sa and best is not None and remaining() > 2:
        from cotengra.pathfinders.path_simulated_annealing import (
            simulated_anneal_tree,
        )
        tree = best[1].copy()
        while remaining() > 2:
            try:
                tree = simulated_anneal_tree(
                    tree, minimize="flops", tfinal=0.05, tstart=2.0,
                    tsteps=40, numiter=8, progbar=False)
                consider(tree)
            except Exception as e:  # noqa: BLE001
                print(f"sa failed: {e}", file=sys.stderr)
                break

    if best is None:
        print("no cap-feasible tree found", file=sys.stderr)
        sys.exit(1)
    print(f"best tc={best[0]:.3f} width={best[1].contraction_width():.1f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
