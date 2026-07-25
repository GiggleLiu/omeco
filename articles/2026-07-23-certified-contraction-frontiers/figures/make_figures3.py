#!/usr/bin/env python3
"""Figures for the huawei same-machine campaign (F11) and the UAI-2014
inference batch (F12). Same Martinis conventions as make_figures{,2}.py:
line weight >= 2, saturated colors + distinct markers (greyscale-safe),
no arbitrary units — all axes in log2 flops relative to a named anchor."""

import json
import pathlib
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = pathlib.Path("/Users/liujinguo/rcode/omeco/articles/2026-07-23-certified-contraction-frontiers")
FIG = ART / "figures"
DATA = ART / "data"

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9, "lines.linewidth": 2.0,
    "axes.linewidth": 1.0, "figure.dpi": 150, "savefig.bbox": "tight",
})
BLACK, BLUE, RED, ORANGE, VIOLET = "#000000", "#0033cc", "#cc0000", "#cc6600", "#660099"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------- F11 matched-budget campaign
def fig11_campaign():
    camp = json.load(open(DATA / "huawei_campaign.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))

    # (a) 15-rep distributions at 90 s, centered per instance on ref median
    ax = axes[0]
    order = ["sycamore_53_20_0", "surfacecode_d21", "ksg", "reg3_1000",
             "dbn_13", "rqc_97_m24"]
    labs = ["Sycamore\n53q", "surface\nd=21", "king\ngraph", "reg3\n1000",
            "DBN", "RQC\n97q"]
    for i, inst in enumerate(order):
        d = camp["p3_distributions"][inst]
        ref = d["ref"]
        best_m = [m for m in d if m != "ref"][0]
        c0 = st.median(ref)
        xs = [i - 0.16 + 0.02 * k for k in range(len(ref))]
        ax.scatter(xs, [v - c0 for v in ref], marker="o", s=22,
                   facecolors="none", edgecolors=BLACK, linewidths=1.3)
        xs = [i + 0.16 - 0.02 * k for k in range(len(d[best_m]))]
        ax.scatter(xs, [v - c0 for v in d[best_m]], marker="s", s=22,
                   facecolors="none", edgecolors=RED, linewidths=1.3)
    ax.axhline(0.0, color=BLACK, lw=1.0, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel(r"tc $-$ tuned TreeSA median  [log$_2$]")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors=BLACK,
               label="tuned TreeSA (15 reps)")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors=RED,
               label="best attempt (15 reps)")
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.05, 0.06, "(a)", transform=ax.transAxes, fontsize=11)

    # (b) surface-code family: both methods' reps vs d, anchored per d
    ax = axes[1]
    ds = [9, 13, 17, 21]
    for j, d_ in enumerate(ds):
        reps = camp["p4_family"][str(d_)]
        anchor = st.median(reps["a054"])
        ax.scatter([j - 0.1] * len(reps["ref"]),
                   [v - anchor for v in reps["ref"]], marker="o", s=26,
                   facecolors="none", edgecolors=BLACK, linewidths=1.4)
        ax.scatter([j + 0.1] * len(reps["a054"]),
                   [v - anchor for v in reps["a054"]], marker="s", s=26,
                   facecolors="none", edgecolors=RED, linewidths=1.4)
    ax.axhline(0.0, color=BLACK, lw=1.0, ls=":")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([f"d={d_}" for d_ in ds])
    ax.set_xlabel("surface-code distance")
    ax.set_ylabel(r"tc $-$ surgery median  [log$_2$]")
    ax.text(0.03, 0.92, "circles: tuned TreeSA\nsquares: waist surgery",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.text(0.05, 0.06, "(b)", transform=ax.transAxes, fontsize=11)

    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig11_campaign")


# ------------------------------------------------- F12 inference two regimes
def fig12_inference():
    inf = json.load(open(DATA / "uai_inference.json"))["instances"]
    order = ["uai_DBN_12", "uai_DBN_14", "uai_DBN_16",
             "uai_linkage_13", "uai_linkage_15", "uai_linkage_17",
             "uai_linkage_23", "uai_CSP_11", "uai_Grids_15", "uai_Promedus_14"]
    labs = ["DBN\n12", "DBN\n14", "DBN\n16", "link.\n13", "link.\n15",
            "link.\n17", "link.\n23", "CSP\n11", "Grids\n15", "Prom.\n14"]

    series = []  # (label, marker, color, values above per-instance frontier)
    rows = {}
    for inst in order:
        e = inf[inst]
        jl = e["julia"]
        vals = {
            "default": jl.get("TI-default-Greedy"),
            "elim": min(v for k, v in jl.items()
                        if k in ("HyperND", "Treewidth-MF") and v != "crash"),
            "treesa": min(v for k, v in jl.items()
                          if k.startswith("TreeSA-") and v != "crash"),
            "ours": min(m["median"] for m in e["ours"].values()),
        }
        frontier = min(vals.values())
        rows[inst] = {k: v - frontier for k, v in vals.items()}

    fig, ax = plt.subplots(figsize=(9, 3.8))
    specs = [("default", "TensorInference default (GreedyMethod)", "v", BLACK),
             ("treesa", "best tuned TreeSA (Julia ladder)", "s", BLUE),
             ("elim", "best elimination (HyperND / Treewidth-MF)", "^", ORANGE),
             ("ours", "this work (best method, median of 5)", "*", RED)]
    for key, lab, mk, c in specs:
        ys = [rows[i][key] for i in order]
        ax.scatter(range(len(order)), ys, marker=mk, s=110 if mk == "*" else 60,
                   facecolors="none", edgecolors=c, linewidths=2.0, label=lab,
                   zorder=3)
    ax.axhline(0.0, color=BLACK, lw=2.0)
    ax.text(6.6, 1.0, "per-instance frontier", fontsize=8, ha="left")
    ax.axvspan(-0.5, 2.5, color="0.93", zorder=0)
    ax.axvspan(2.5, 6.5, color="0.985", zorder=0)
    ax.text(1.0, 33.0, "dense DBN:\nelimination wins", fontsize=8, ha="center")
    ax.text(4.5, 33.0, "pedigree linkage:\nannealing wins", fontsize=8,
            ha="center")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(r"tc above frontier  [log$_2$ flops]")
    ax.set_ylim(-1.2, 36)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8)
    leg.get_frame().set_edgecolor("none")
    save(fig, "fig12_inference")





# --------------------------------------- F13 JOSS-style Pareto (time vs tc)
def fig13_pareto():
    """Mimics the presentation of the OMEinsumContractionOrders JOSS paper
    (per-configuration scatter, log10 wall-time vs log2 flops, dashed
    Pareto front), on same-machine data."""
    import math
    data = json.load(open(DATA / "pareto_points.json"))["instances"]
    FAMS = [
        ("julia-treesa", "TreeSA (Julia, 6 configs)", "o", VIOLET, True),
        ("hypernd", "HyperND (Julia)", "s", BLUE, True),
        ("treewidth", "Treewidth MF/AMF/MMD (Julia)", "^", ORANGE, True),
        ("cotengra", "cotengra hyper+SA", "P", BLACK, True),
        ("ours-treesa", "tuned TreeSA (this work)", "D", "#9933cc", False),
        ("ours-simplify", "simplify+anneal (this work)", "v", BLUE, False),
        ("ours-surgery", "waist surgery (this work)", "*", RED, False),
    ]
    titles = {"sycamore_53_20_0": "Sycamore 53q, m=20 (3369 tensors)",
              "surfacecode_d21": "surface code d=21 (2203 tensors)"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, inst in zip(axes, ["sycamore_53_20_0", "surfacecode_d21"]):
        pts = data[inst]
        for fam, lab, mk, c, filled in FAMS:
            xs = [p["tc"] for p in pts if p["family"] == fam]
            ys = [math.log10(max(p["t"], 0.02)) for p in pts if p["family"] == fam]
            if not xs:
                continue
            size = 150 if mk == "*" else 55
            ax.scatter(xs, ys, marker=mk, s=size,
                       facecolors=c if filled else "none",
                       edgecolors=c, linewidths=1.8,
                       label=lab if ax is axes[0] else None, zorder=3)
        # Pareto front: non-dominated in (tc, t)
        srt = sorted(pts, key=lambda p: (p["tc"], p["t"]))
        front, best_t = [], float("inf")
        for p in srt:
            if p["t"] < best_t:
                front.append(p)
                best_t = p["t"]
        front = sorted(front, key=lambda p: p["tc"])
        ax.plot([p["tc"] for p in front],
                [math.log10(max(p["t"], 0.02)) for p in front],
                ls="--", color=BLACK, lw=1.6, zorder=2)
        mid = front[min(1, len(front) - 1)]
        ax.annotate("Pareto front", (mid["tc"], math.log10(max(mid["t"], 0.02))),
                    textcoords="offset points", xytext=(14, 6), fontsize=9)
        ax.set_xlabel(r"$\log_2$ FLOPs (time complexity tc)")
        ax.set_title(titles[inst], fontsize=10)
    axes[0].set_ylabel(r"$\log_{10}$ wall-clock time  [s]")
    leg = axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    axes[0].text(0.04, 0.05, "(a)", transform=axes[0].transAxes, fontsize=11)
    axes[1].text(0.04, 0.05, "(b)", transform=axes[1].transAxes, fontsize=11)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig13_pareto")


if __name__ == "__main__":
    fig11_campaign()
    fig12_inference()
    fig13_pareto()
