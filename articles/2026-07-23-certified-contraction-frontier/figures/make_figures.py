#!/usr/bin/env python3
"""Draft figures for the certified-contraction-frontier paper.

All numbers come from the autoresearch audit trail:
- .worktrees/attempt-026/data/results.json  (bounds, profiles, tree nodes)
- research/validator/leaderboard.json        (reference rows, records)
- docs/discussion/*.md                       (attempt scores, budget scaling)
- .worktrees/attempt-027/LOG.md              (profile-conservation histograms)

Style: paper figure rulebook — saturated colors + distinct line styles
(greyscale-safe), line weight >= 2, readable text, no arbitrary units.
"""
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/Users/liujinguo/rcode/omeco")
OUT = ROOT / "articles/2026-07-23-certified-contraction-frontier/figures"
R26 = json.load(open(ROOT / ".worktrees/attempt-026/data/results.json"))

BLACK, BLUE, RED, ORANGE, VIOLET = "#000000", "#1a52b0", "#c11b1b", "#c66a00", "#6b2fa0"
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11.5,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 2, "axes.linewidth": 1.0, "figure.dpi": 150,
    "savefig.bbox": "tight",
})

FRONTIER = {"reg3_250": 39.950, "sycamore_m20": 61.544}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1
# Certification ladder: bounds vs the achieved frontier, per instance.
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    data = {
        "reg3_250": [
            # (value, label, kind)  kind: cert | struct | achieved
            (13.14, "spectral sum-form (Thm 1)", "cert"),
            (18.0, "treewidth minor certificate", "cert"),
            (30.81, "isoperimetric sum-form", "struct"),
            (34.0, "achieved width (sc)", "achieved"),
            (39.95, "frontier tc", "frontier"),
        ],
        "sycamore_m20": [
            (9.84, "spectral sum-form (Thm 1)", "cert"),
            (22.0, "treewidth minor certificate", "cert"),
            (47.17, "isoperimetric sum-form", "struct"),
            (53.0, "balanced temporal cut = width floor", "struct"),
            (61.544, "frontier tc", "frontier"),
        ],
    }
    styles = {
        "cert": dict(color=BLACK, ls="-"),
        "struct": dict(color=BLUE, ls="--"),
        "achieved": dict(color=ORANGE, ls="-."),
        "frontier": dict(color=RED, ls="-"),
    }
    titles = {"reg3_250": "reg3_250  (n = 250)",
              "sycamore_m20": "sycamore_m20  (n = 561)"}
    for ax, inst in zip(axes, data):
        for v, label, kind in data[inst]:
            st = styles[kind]
            ax.hlines(v, 0.05, 0.52, colors=st["color"], linestyles=st["ls"], lw=2.4)
            ax.annotate(f"{label}  ({v:g})", xy=(0.55, v), va="center",
                        fontsize=9, color=st["color"])
        if inst == "sycamore_m20":
            ax.axhspan(53.0, 61.544, xmin=0.05, xmax=0.52, color=RED, alpha=0.10)
            ax.annotate("optimum in [53, 61.5];\ngap 8.5 ≈ log₂ n", xy=(0.10, 55.6),
                        fontsize=9, color=RED)
        else:
            ax.axhspan(34.0, 39.95, xmin=0.05, xmax=0.52, color=RED, alpha=0.10)
            ax.annotate("residual ≈ log-count\nof near-max nodes",
                        xy=(0.10, 35.9), fontsize=9, color=RED)
        ax.set_title(titles[inst])
        ax.set_xlim(0, 2.6)
        ax.set_xticks([])
        ax.set_ylim(0, max(v for v, _, _ in data[inst]) * 1.10)
        ax.set_ylabel("log₂ cost  (bits)")
    handles = [plt.Line2D([], [], **{**styles[k], "lw": 2.4},
                          label={"cert": "certified lower bound",
                                 "struct": "structural bound (high-conf.)",
                                 "achieved": "achieved width",
                                 "frontier": "achieved frontier tc"}[k])
               for k in ["cert", "struct", "achieved", "frontier"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.12)
    save(fig, "fig1_certification_ladder")


# ---------------------------------------------------------------- fig 2
# Convergence: every optimizer lands on one frontier; hyper-opt alone lags.
ATTEMPTS = {  # scored tc per (reg3_250, sycamore_m20), cycles 2-5
    "011": (40.16, 61.73), "012": (40.29, 62.05), "015": (39.95, 61.80),
    "017": (39.95, 61.57), "018": (40.02, 61.51), "019": (39.996, 61.619),
    "020": (40.110, 61.635), "023": (40.024, 61.571), "024": (39.949, 61.607),
    "027": (40.114, 61.598),
}
REFS = {  # pure-tc reference rows (leaderboard)
    "treesa-inf": (39.950, 61.544), "cotengra-sa": (40.002, 61.679),
    "cotengra-hyper": (41.678, 63.971),
    "treesa default sc_target": (44.285, 71.171),
}


def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cats = ["in-house mechanisms\n(10 scored, cycles 2–5)", "TreeSA tuned\n(reference)",
            "cotengra SA", "cotengra hyper\n(no SA refinement)", "TreeSA default\nsc_target = 20"]
    xs = {c: i for i, c in enumerate(cats)}
    for name, (r, s) in ATTEMPTS.items():
        ax.plot(xs[cats[0]] - 0.08, r - FRONTIER["reg3_250"], "o", color=BLUE,
                ms=7, mfc="none", mew=2)
        ax.plot(xs[cats[0]] + 0.08, s - FRONTIER["sycamore_m20"], "^", color=RED,
                ms=7, mfc="none", mew=2)
    ref_x = {"treesa-inf": 1, "cotengra-sa": 2, "cotengra-hyper": 3,
             "treesa default sc_target": 4}
    for name, (r, s) in REFS.items():
        x = ref_x[name]
        ax.plot(x - 0.08, r - FRONTIER["reg3_250"], "o", color=BLUE, ms=9)
        ax.plot(x + 0.08, s - FRONTIER["sycamore_m20"], "^", color=RED, ms=9)
    ax.axhline(0, color=BLACK, lw=1.2, ls=":")
    ax.set_xticks(range(len(cats)), cats, fontsize=9)
    ax.set_ylabel("Δtc above frontier  (bits)")
    ax.plot([], [], "o", color=BLUE, mfc="none", mew=2, label="reg3_250")
    ax.plot([], [], "^", color=RED, mfc="none", mew=2, label="sycamore_m20")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.5, 11)
    save(fig, "fig2_frontier_convergence")


# ---------------------------------------------------------------- fig 3
# The masquerade: novel-mechanism "records" fully explained by baseline tuning.
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=True)
    stages = ["baseline\nreference", "schedule\nhygiene", "8 novel\nmechanisms",
              "tuned\nreference"]
    reg = {"base": 45.008, "hyg": 44.285,
           "novel": [40.16, 40.29, 43.45, 45.04, 39.95, 45.12, 39.95, 40.02],
           "tuned": 39.905}
    syc = {"base": 72.585, "hyg": 71.171,
           "novel": [61.73, 62.05, 72.00, 66.06, 61.80, 67.08, 61.57, 61.51],
           "tuned": 61.527}
    for ax, d, inst in zip(axes, [reg, syc], ["reg3_250", "sycamore_m20"]):
        ax.plot(0, d["base"], "s", color=BLACK, ms=9)
        ax.plot(1, d["hyg"], "s", color=BLACK, ms=9)
        ax.plot([2] * len(d["novel"]), d["novel"], "o", color=BLUE, ms=7,
                mfc="none", mew=2)
        ax.plot(3, d["tuned"], "*", color=RED, ms=16)
        ax.axhline(d["tuned"], color=RED, lw=1.4, ls="--")
        ax.set_xticks(range(4), stages, fontsize=9)
        ax.set_title(inst)
        ax.set_ylabel("scored tc  (bits)")
    axes[1].annotate("apparent 800× gain…", xy=(2, 65.5), fontsize=9,
                     color=BLUE, ha="center")
    axes[1].annotate("…is one reference\nparameter (sc_target)", xy=(3, 64.3),
                     fontsize=9, color=RED, ha="center")
    save(fig, "fig3_sc_target_masquerade")


# ---------------------------------------------------------------- fig 4
# Isoperimetric profiles vs what a real tree pays.
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, inst in zip(axes, ["reg3_250", "sycamore_m20"]):
        d = R26[inst]
        n = d["n"]
        ks = list(range(n + 1))
        ax.plot(ks, d["b_emp"], "-", color=BLUE, lw=2.2,
                label="isoperimetric profile b(k) (empirical)")
        ax.plot(ks, d["b_spec"], "-.", color=VIOLET, lw=2,
                label="spectral bound (certified)")
        sizes = [t["size"] for t in d["tree_nodes"]]
        bounds = [t["boundary"] for t in d["tree_nodes"]]
        ax.plot(sizes, bounds, "o", color=RED, ms=3.5, alpha=0.55,
                label="frontier-tree nodes (|S|, |∂S|)", ls="none")
        peak = d["emp_profile_peak"]
        ax.axhline(peak, color=ORANGE, lw=1.6, ls="--")
        ax.annotate(f"width cap {peak}", xy=(0.02 * n, peak + 0.8),
                    fontsize=9, color=ORANGE)
        bal = d["emp_maxform_balanced"]
        ax.axhline(bal, color=BLACK, lw=1.4, ls=":")
        if inst == "reg3_250":
            ax.annotate(f"bisection width {bal} — ceiling of every\nprofile-only bound (Thms 1–2)",
                        xy=(0.30 * n, bal - 7.5), fontsize=8.5, color=BLACK)
        else:
            ax.annotate(f"bisection width {bal} — ceiling of every\nprofile-only bound (Thms 1–2)",
                        xy=(210, 10.5), fontsize=8.5, color=BLACK)
        ax.set_xlabel("subset size k  (tensors)")
        ax.set_ylabel("boundary  (bits)")
        ax.set_title(inst)
    if True:  # b(141) <= 40 certificate (attempt-031) on sycamore
        axes[1].plot([141], [40], "D", color=BLACK, ms=8, mfc="none", mew=2)
        axes[1].annotate("b(141) ≤ 40:\noff-center dips\nbeat temporal slabs",
                         xy=(141, 40), xytext=(200, 27), fontsize=8.5, color=BLACK,
                         arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2))
    axes[0].legend(loc="lower center", fontsize=8.2, framealpha=0.95)
    save(fig, "fig4_profiles")


# ---------------------------------------------------------------- fig 5
# Profile conservation: lowering the peak fattens the near-max shelf.
def fig5():
    # Two measures with different units -> small multiples, never a dual axis.
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6), sharex="col")
    # (label, peak, within-1-count) at equal-or-better tc, from attempt-027
    reg = [("TreeSA-inf tree\n(tc 40.02)", 37, 7), ("width-reduced tree\n(tc 39.88)", 36, 19)]
    syc = [("TreeSA-inf tree\n(tc 63.58)", 61, 5), ("width-reduced tree\n(tc 61.51)", 57, 17)]
    for col, (rows, inst) in enumerate(zip([reg, syc], ["reg3_250", "sycamore_m20"])):
        x = [0, 1]
        peaks = [r[1] for r in rows]
        counts = [r[2] for r in rows]
        axp, axc = axes[0][col], axes[1][col]
        axp.plot(x, peaks, "s-", color=BLUE, ms=9)
        for xi, p in zip(x, peaks):
            axp.annotate(str(p), xy=(xi, p + 0.3), ha="center", fontsize=9, color=BLUE)
        axp.set_title(inst)
        axp.set_ylim(min(peaks) - 1.6, max(peaks) + 1.6)
        axc.plot(x, counts, "o--", color=RED, ms=9)
        for xi, c in zip(x, counts):
            axc.annotate(str(c), xy=(xi, c + 1.0), ha="center", fontsize=9, color=RED)
        axc.set_ylim(0, max(counts) * 1.5)
        axc.set_xticks(x, [r[0] for r in rows], fontsize=8.5)
        axc.set_xlim(-0.45, 1.45)
        if col == 0:
            axp.set_ylabel("peak node cost\n(bits)")
            axc.set_ylabel("nodes within 1 bit\nof the peak")
    save(fig, "fig5_profile_conservation")


# ---------------------------------------------------------------- fig 6
# Budget scaling: the frontier is budget-independent at these sizes.
def fig6():
    budgets = [90, 300, 900]
    reg = [39.94, 39.88, 39.95]
    syc = [61.61, 61.58, 61.51]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.semilogx(budgets, [v - FRONTIER["reg3_250"] for v in reg], "o-",
                color=BLUE, ms=8, label="reg3_250")
    ax.semilogx(budgets, [v - FRONTIER["sycamore_m20"] for v in syc], "^--",
                color=RED, ms=8, label="sycamore_m20")
    ax.axhspan(-0.15, 0.15, color="0.85", zorder=0)
    ax.annotate("single-run noise band", xy=(95, 0.17), fontsize=8.5, color="0.35")
    ax.set_xlabel("optimizer budget  (s, single thread)")
    ax.set_ylabel("Δtc vs 90 s frontier  (bits)")
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.legend()
    ax.set_ylim(-0.6, 0.6)
    save(fig, "fig6_budget_scaling")


fig1(), fig2(), fig3(), fig4(), fig5(), fig6()
