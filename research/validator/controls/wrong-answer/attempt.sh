#!/bin/bash
/usr/bin/python3 - "$1" "$3" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
n = len(g["ixs"])
# left-comb tree over tensors 0..n-2 — omits the last tensor entirely
node = {"isleaf": True, "tensorindex": 0}
for i in range(1, n - 1):
    node = {"isleaf": False,
            "args": [node, {"isleaf": True, "tensorindex": i}],
            "eins": {"ixs": [[], []], "iy": []}}
doc = {"label-type": "Int64", "inputs": g["ixs"], "output": g["iy"], "tree": node}
json.dump(doc, open(sys.argv[2], "w"))
PY
exit 0
