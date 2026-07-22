"""Generate a Sycamore-m20-scale RQC tensor network (single-amplitude).

Proxy construction: 53 qubits on an 8x7 grid minus 3 corners; grid edges
4-colored (right/even, right/odd, down/even, down/odd); cycle t applies
2-qubit gates on all edges of color t%4, for m=20 cycles. Single-qubit gates
are absorbed into the 2-qubit gate tensors (standard practice). The network
is <input vectors> - <rank-4 gate tensors> - <output vectors>, contracted to
a scalar amplitude (iy=[]), all bond dimensions 2.

Deterministic (no seed): the structure is fixed by the layout; there is no
randomness to hide. Writes research/benchmark/targets/sycamore_m20.json.
"""

import json
import pathlib

ROWS, COLS, CYCLES = 8, 7, 20


def build():
    cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
    # drop 3 corners to reach 53 qubits
    for corner in [(0, 0), (0, COLS - 1), (ROWS - 1, 0)]:
        cells.remove(corner)
    assert len(cells) == 53
    qubits = {cell: i for i, cell in enumerate(cells)}

    def color(a, b):
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:  # horizontal
            return 0 if min(c1, c2) % 2 == 0 else 1
        return 2 if min(r1, r2) % 2 == 0 else 3

    edges = {0: [], 1: [], 2: [], 3: []}
    for (r, c) in cells:
        for nb in [(r, c + 1), (r + 1, c)]:
            if nb in qubits:
                edges[color((r, c), nb)].append(((r, c), nb))

    next_label = 0

    def fresh():
        nonlocal next_label
        next_label += 1
        return next_label - 1

    wire = {}  # qubit cell -> current open label
    ixs = []
    for cell in cells:  # input state vectors, rank 1
        wire[cell] = fresh()
        ixs.append([wire[cell]])
    for t in range(CYCLES):
        for a, b in edges[t % 4]:
            ia, ib = wire[a], wire[b]
            oa, ob = fresh(), fresh()
            ixs.append([ia, ib, oa, ob])
            wire[a], wire[b] = oa, ob
    for cell in cells:  # output projectors, rank 1
        ixs.append([wire[cell]])

    sizes = {str(l): 2 for l in range(next_label)}
    return {
        "name": "sycamore_m20",
        "description": f"Sycamore-m20-scale RQC proxy: 53 qubits (8x7 grid "
                       f"minus 3 corners), {CYCLES} cycles, ABCD edge "
                       f"coloring, single-amplitude, {len(ixs)} tensors",
        "ixs": ixs, "iy": [], "sizes": sizes,
    }


if __name__ == "__main__":
    g = build()
    out = pathlib.Path("research/benchmark/targets/sycamore_m20.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(g, open(out, "w"))
    print(f"wrote {out}: {len(g['ixs'])} tensors, {len(g['sizes'])} indices")
