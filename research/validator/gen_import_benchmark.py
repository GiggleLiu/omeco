"""Import instances from OMEinsumContractionOrdersBenchmark into targets/.

Source: /Users/liujinguo/jcode/OMEinsumContractionOrdersBenchmark (local clone;
schema {"einsum": {"ixs": [[...]], "iy": [...]}, "size": {label: dim}}, Julia
1-based integer labels). Converted to the validator target schema
{"name", "description", "ixs", "iy", "sizes"} with labels kept as ints
(permute_instance relabels per run anyway).

Provenance: user-directed 2026-07-24 ("add some harder instances from
OMEinsumContractionOrdersBenchmark"); instances are the published benchmark
set of OMEinsumContractionOrders.jl v1.0.0, deterministic files checked into
that repo (no hidden seeds on our side).
"""

import json
import pathlib

SRC = pathlib.Path("/Users/liujinguo/jcode/OMEinsumContractionOrdersBenchmark/examples")
DST = pathlib.Path(__file__).resolve().parent.parent / "benchmark" / "targets"

IMPORTS = {
    "sycamore_53_20_0": ("quantumcircuit/codes/sycamore_53_20_0.json",
                         "REAL Sycamore 53q m=20 circuit (3369 tensors) from "
                         "OMEinsumContractionOrdersBenchmark"),
    "surfacecode_d21": ("qec/codes/surfacecode_d=21.json",
                        "surface code d=21 QEC tensor network (2203 tensors)"),
    "ksg": ("independentset/codes/ksg.json",
            "independent-set counting, king's subgraph (5197 tensors)"),
    "nqueens_28": ("nqueens/codes/nqueens_n=28.json",
                   "28-queens counting network (4252 tensors)"),
    "dbn_13": ("inference/codes/DBN_13.json",
               "deep belief network inference, 572 tensors over 44 shared "
               "labels (dense hyperedges)"),
    "qft_27": ("einsumorg/codes/qc_qft_27.json",
               "QFT-27 circuit with 27 open indices (405 tensors)"),
}


def main():
    for name, (rel, desc) in IMPORTS.items():
        src = json.load(open(SRC / rel))
        e, size = src["einsum"], src["size"]
        out = {
            "name": name,
            "description": desc + f" [source: {rel}]",
            "ixs": [[int(l) for l in ix] for ix in e["ixs"]],
            "iy": [int(l) for l in e["iy"]],
            "sizes": {str(int(k)): int(v) for k, v in size.items()},
        }
        # sanity: every used label has a size; leaf count sane
        used = {l for ix in out["ixs"] for l in ix} | set(out["iy"])
        missing = used - {int(k) for k in out["sizes"]}
        assert not missing, (name, sorted(missing)[:5])
        path = DST / f"{name}.json"
        json.dump(out, open(path, "w"))
        print(f"{name}: {len(out['ixs'])} tensors, {len(out['sizes'])} labels, "
              f"|iy|={len(out['iy'])} -> {path.name}")


if __name__ == "__main__":
    main()
