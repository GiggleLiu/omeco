"""Generate the sealed holdout instance set for beat-existing-optimizers.

Run once from the repo root. Seeds come from os.urandom and are stored ONLY
in research/benchmark/private/seeds.json (gitignored) — attempts can read
this generator's source but cannot reproduce the instances without seeds.

Families mirror the dev suite at sizes absent from it:
  reg3_60, reg3_140, reg3_300 (random 3-regular, pairing model),
  grid_7x7, chain_30. All bond dimensions 2; einsum format matches
  benchmarks/graphs/*.json.
"""

import json
import os
import pathlib
import random

PRIV = pathlib.Path("research/benchmark/private")


def gen_reg3(n, rng):
    """Random 3-regular multigraph via pairing model, retried until simple."""
    while True:
        stubs = [v for v in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            return sorted(edges)


def edges_to_einsum(name, desc, n, edges):
    incident = {v: [] for v in range(n)}
    for idx, (a, b) in enumerate(edges):
        label = n + idx
        incident[a].append(label)
        incident[b].append(label)
    ixs = [incident[v] for v in range(n)]
    sizes = {str(lbl): 2 for v in range(n) for lbl in incident[v]}
    return {"name": name, "description": desc, "ixs": ixs, "iy": [],
            "sizes": sizes}


def gen_grid(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return edges


def gen_chain(n):
    """Open chain of n matrices: tensor i has labels [i, i+1]; iy = ends."""
    ixs = [[i, i + 1] for i in range(n)]
    sizes = {str(i): 2 for i in range(n + 1)}
    return {"name": f"chain_{n}", "description": f"Matrix chain of {n} matrices",
            "ixs": ixs, "iy": [0, n], "sizes": sizes}


def main():
    PRIV.mkdir(parents=True, exist_ok=True)
    seeds = {f"reg3_{n}": int.from_bytes(os.urandom(4), "big")
             for n in (60, 140, 300)}
    with open(PRIV / "seeds.json", "w") as f:
        json.dump(seeds, f)

    graphs = []
    for n in (60, 140, 300):
        rng = random.Random(seeds[f"reg3_{n}"])
        edges = gen_reg3(n, rng)
        graphs.append(edges_to_einsum(
            f"reg3_{n}_h", f"holdout random 3-regular ({n} vertices)", n, edges))
    rows = cols = 7
    edges = gen_grid(rows, cols)
    g = edges_to_einsum("grid_7x7_h", "holdout 7x7 grid", rows * cols, edges)
    graphs.append(g)
    graphs.append(gen_chain(30))

    for g in graphs:
        with open(PRIV / f"{g['name']}.json", "w") as f:
            json.dump(g, f)
        print("wrote", g["name"], len(g["ixs"]), "tensors")


if __name__ == "__main__":
    main()
