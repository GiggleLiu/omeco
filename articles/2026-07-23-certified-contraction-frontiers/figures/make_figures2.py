#!/usr/bin/env python3
"""New figures for the story-A' restructure (record board, simplification,
waist surgery). Same Martinis conventions as make_figures.py: line weight
>= 2, saturated colors + distinct markers/line styles (greyscale-safe),
data as points / references as lines, no arbitrary units."""

import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/Users/liujinguo/rcode/omeco")
ART = ROOT / "articles/2026-07-23-certified-contraction-frontiers"
FIG = ART / "figures"
DATA = ART / "data"

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9, "lines.linewidth": 2.0,
    "axes.linewidth": 1.0, "figure.dpi": 150, "savefig.bbox": "tight",
})
BLACK, BLUE, RED, ORANGE, VIOLET = "#000000", "#0033cc", "#cc0000", "#cc6600", "#660099"
BROWN = "#5c3a1e"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------ F2' record board
def fig8_record_board():
    board = json.load(open(DATA / "record_board.json"))
    refs = board["reference_rows"]
    recs = board["records"]
    # instance -> (reference tc = treesa-inf row, record tc, holder, mechanism color)
    mech = {
        "attempt-039": ("simplify-then-anneal", BLUE, "o"),
        "attempt-050": ("simplify-then-anneal", BLUE, "o"),
        "attempt-047": ("composite", VIOLET, "D"),
        "attempt-054": ("waist surgery", RED, "s"),
        "attempt-038": ("VE seed", ORANGE, "^"),
    }
    instances = ["sycamore_53_20_0", "surfacecode_d21", "ksg", "reg3_1000",
                 "dbn_13", "rqc_97_m24", "nqueens_28", "qft_27"]
    labels = ["Sycamore\n53q m=20\n(3369)", "surface code\nd=21\n(2203)",
              "king graph\nIS (5197)", "random\n3-regular\n(1000)",
              "DBN\ninference\n(572)", "RQC 97q\nm=24\n(1238)",
              "28-queens\n(4252)", "QFT-27\n(405)"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    seen = set()
    for i, inst in enumerate(instances):
        ref = refs[inst]["treesa-inf"]["tc"]
        rec = recs[inst]
        delta = ref - rec["tc"]
        by = rec["by"]
        if by.startswith("ref:"):
            ax.scatter([i], [0.0], marker="_", s=420, color=BLACK, linewidths=3.0)
        else:
            name, c, m = mech.get(by, ("other", BROWN, "v"))
            lab = name if name not in seen else None
            seen.add(name)
            ax.scatter([i], [delta], marker=m, s=70, facecolors="none",
                       edgecolors=c, linewidths=2.2, label=lab, zorder=3)
            ax.vlines(i, 0, delta, color=c, lw=1.2, ls=":", zorder=2)
    ax.axhline(0.0, color=BLACK, lw=2.0)
    ax.text(7.45, 0.06, "tuned TreeSA\n(reference)", fontsize=9, ha="right",
            va="bottom", color=BLACK)
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"record improvement  $\Delta$tc  [log$_2$ flops]")
    ax.set_ylim(-0.15, 3.1)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    save(fig, "fig8_record_board")


# ------------------------------------------------- F3' simplification pillar
def fig9_simplification():
    att = json.load(open(DATA / "mechanism_attribution.json"))
    sh = att["simplification"]["shrink"]
    march = att["simplification"]["sycamore_record_march"]
    ab = att["simplification"]["matched_budget_ab_sycamore_53"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    ax = axes[0]
    insts = ["sycamore_53_20_0", "surfacecode_d21", "nqueens_28",
             "dbn_13", "qft_27", "reg3_1000"]
    labs = ["Sycamore\n53q", "surface\ncode d21", "28-queens", "DBN", "QFT-27",
            "reg3\n1000"]
    fracs = [sh[i]["frac"] for i in insts]
    ax.bar(range(len(insts)), fracs, color="none", edgecolor=BLUE, linewidth=2.2,
           hatch="//")
    ax.set_xticks(range(len(insts)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("tensors removed by\nfree contractions  [fraction]")
    ax.set_ylim(0, 1.02)
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.03, f"{f:.0%}", ha="center", fontsize=8)
    ax.text(0.86, 0.90, "(a)", transform=ax.transAxes, fontsize=11)

    ax = axes[1]
    xs = range(len(march))
    ys = [m["tc"] for m in march]
    ax.plot(xs, ys, color=BLUE, lw=2.0, ls="-", marker="o", markersize=8,
            markerfacecolor="none", markeredgewidth=2.2)
    for x, m in zip(xs, march):
        ax.annotate(m["by"].replace("ref:treesa-inf", "tuned TreeSA")
                    .replace("attempt-", "attempt "),
                    (x, m["tc"]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8)
    ax.set_ylim(59.4, 61.2)
    ax.set_xticks(list(xs))
    ax.set_xlim(-0.4, 3.6)
    ax.set_xticklabels(["reference", "cycle 8", "cycle 9a", "cycle 9b"], fontsize=9)
    ax.set_ylabel(r"tc  [log$_2$ flops]")
    ax.text(0.5, 0.14, "Sycamore 53q, m=20", transform=ax.transAxes, fontsize=9,
            ha="center", bbox=dict(fc="white", ec=BLACK, lw=0.8))
    ax.annotate(r"same anneal without simplification: 76.0 (off scale $\uparrow$)",
                xy=(0.03, 0.93), xycoords="axes fraction", color=RED, fontsize=8)
    ax.text(0.9, 0.05, "(b)", transform=ax.transAxes, fontsize=11)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig9_simplification")


# ---------------------------------------------------- F4' waist surgery pillar
def _parse_trace(path):
    waist, traj = [], []
    for line in open(path):
        m = re.match(r"t=(\d+)ms iter=(\d+) WAIST cost=([\d.]+) best_alt=([\d.]+) "
                     r"gap=([\d.]+) tried=\d+ sep_labels=\d+ clique_frac=([\d.]+)", line)
        if m:
            waist.append({"t": int(m.group(1)) / 1e3, "cost": float(m.group(3)),
                          "alt": float(m.group(4)), "clique": float(m.group(6))})
            continue
        m = re.match(r"t=(\d+)ms iter=(\d+) REBUILD (ACCEPT|reject) new_tc=([\d.]+)", line)
        if m:
            traj.append({"t": int(m.group(1)) / 1e3, "accept": m.group(3) == "ACCEPT",
                         "tc": float(m.group(4))})
            continue
        m = re.match(r"t=(\d+)ms iter=(\d+) incumb=([\d.]+)", line)
        if m:
            traj.append({"t": int(m.group(1)) / 1e3, "accept": None,
                         "tc": float(m.group(3))})
    return waist, traj


def fig10_waist_surgery():
    # panel (a) pools waist-vs-alternative calls from all fresh runs; panel (b)
    # shows run 2 (final 47.44, representative of the official median 47.377;
    # runs ended 47.90/47.44/47.46 -- stated in the caption).
    w_sc, _ = _parse_trace(DATA / "waist_trace_surfacecode.log")
    for extra in ["waist_trace_surfacecode2.log", "waist_trace_surfacecode3.log"]:
        w, _ = _parse_trace(DATA / extra)
        w_sc += w
    _, tr_sc = _parse_trace(DATA / "waist_trace_surfacecode2.log")
    w_ksg, _ = _parse_trace(DATA / "waist_trace_ksg.log")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    ax.scatter([w["cost"] for w in w_sc], [w["alt"] for w in w_sc], marker="o",
               s=34, facecolors="none", edgecolors=BLUE, linewidths=1.6,
               label=f"surface code d=21 ({len(w_sc)} calls)")
    ax.scatter([w["cost"] for w in w_ksg], [w["alt"] for w in w_ksg], marker="s",
               s=30, facecolors="none", edgecolors=RED, linewidths=1.6,
               label=f"king graph IS ({len(w_ksg)} calls)")
    lo = min(min(w["alt"] for w in w_sc + w_ksg) - 2,
             min(w["cost"] for w in w_sc + w_ksg) - 2)
    hi = max(w["cost"] for w in w_sc + w_ksg) + 2
    ax.plot([lo, hi], [lo, hi], color=BLACK, lw=2.0, ls="--")
    ax.text(hi - 1, hi - 3.4, "waist not\nimprovable", fontsize=8, ha="right")
    ax.set_xlabel(r"incumbent waist cut weight  [log$_2$]")
    ax.set_ylabel(r"best cut found by FM  [log$_2$]")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.9, 0.06, "(a)", transform=ax.transAxes, fontsize=11)

    ax = axes[1]
    inc = [(p["t"], p["tc"]) for p in tr_sc if p["accept"] is None]
    acc = [(p["t"], p["tc"]) for p in tr_sc if p["accept"] is True]
    ax.plot([t for t, _ in inc], [c for _, c in inc], color=BLUE, lw=2.0,
            label="incumbent tc")
    ax.scatter([t for t, _ in acc], [c for _, c in acc], marker="*", s=140,
               color=RED, zorder=3, label="accepted waist rebuild")
    ax.axhline(47.824, color=BLACK, lw=2.0, ls=":")
    ax.text(30, 47.85, "pre-surgery record 47.824", fontsize=8)
    ax.axhline(47.377, color=RED, lw=1.6, ls="-.")
    ax.text(2, 47.31, "official median with surgery 47.377", fontsize=8, color=RED)
    ax.set_ylim(47.2, None)
    ax.set_xlabel("wall-clock time  [s]")
    ax.set_ylabel(r"surface code d=21   tc  [log$_2$ flops]")
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.03, 0.06, "(b)", transform=ax.transAxes, fontsize=11)
    save(fig, "fig10_waist_surgery")


if __name__ == "__main__":
    fig8_record_board()
    fig9_simplification()
    fig10_waist_surgery()
