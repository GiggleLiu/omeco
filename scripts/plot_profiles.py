"""Plot isoperimetric-profile curves and the tree's own boundary profile."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "data/results.json")))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, name in zip(axes, ["reg3_250", "sycamore_m20"]):
    d = r[name]
    n = d["n"]
    ks = np.arange(n + 1)
    b_spec = np.array(d["b_spec"])
    b_emp = np.array(d["b_emp"])
    ax.plot(ks, b_emp, color="C0", lw=1.8,
            label="empirical profile b_emp(k) (achievable cuts)")
    ax.plot(ks, b_spec, color="C3", lw=1.6, ls="--",
            label=r"certified spectral $\lambda_2 k(n-k)/n$")
    # frontier tree node boundaries as scatter
    sizes = [nd["size"] for nd in d["tree_nodes"]]
    bnds = [nd["boundary"] for nd in d["tree_nodes"]]
    ax.scatter(sizes, bnds, s=9, color="C1", alpha=0.45,
               label="frontier-tree node |∂S_v|")
    ax.axhline(d["frontier_sc"], color="gray", ls=":", lw=1,
               label=f"frontier sc = {d['frontier_sc']}")
    ax.set_title(f"{name}  (n={n})")
    ax.set_xlabel("subset size k")
    ax.set_ylabel(r"boundary $|\partial S|$  (= log2 width)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(ROOT, "data", "profiles.png")
fig.savefig(out, dpi=140)
print("wrote", out)
