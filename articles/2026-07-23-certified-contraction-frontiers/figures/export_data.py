#!/usr/bin/env python3
"""Render every figure's plot-ready data to JSON under data/fig/.

One JSON per figure; the Typst sources in figures/*.typ read ONLY these
files. Raw provenance (jsonl, csv, logs, campaign score dirs) is reduced
here, once, so the .typ files stay pure presentation.
"""

import collections
import csv
import json
import math
import pathlib
import re
import statistics as st

ROOT = pathlib.Path("/Users/liujinguo/rcode/omeco")
ART = ROOT / "articles/2026-07-23-certified-contraction-frontiers"
DATA = ART / "data"
OUT = DATA / "fig"
OUT.mkdir(exist_ok=True)


def dump(name, obj):
    json.dump(obj, open(OUT / f"{name}.json", "w"), indent=1)
    print("wrote", name)


# ------------------------------------------------------------------ fig1
def fig1():
    runs = collections.defaultdict(list)
    for line in open(DATA / "sweep_sctarget.jsonl"):
        d = json.loads(line)
        runs[(d["instance"], d["sc_target"])].append(d["result"]["tc"])
    order = {"reg3_250": ["10", "15", "20", "25", "30", "34", "40", "50", "inf"],
             "sycamore_m20": ["20", "30", "40", "45", "50", "53", "60", "70", "inf"]}
    out = {"frontier": {"reg3_250": 39.95, "sycamore_m20": 61.544},
           "natural": {"reg3_250": "34", "sycamore_m20": "53"},
           "shipped_default": "20", "instances": {}}
    for inst, targets in order.items():
        rows = []
        for v in targets:
            vals = sorted(runs.get((inst, v), []))
            if not vals:
                continue
            rows.append({"sc_target": v, "median": vals[len(vals) // 2],
                         "lo": vals[0], "hi": vals[-1]})
        out["instances"][inst] = rows
    dump("fig1_sweep", out)


# ------------------------------------------------------------------ fig2
def fig2():
    out = {
        "frontier": {"reg3_250": 39.95, "sycamore_m20": 61.544},
        "references": [
            {"label": "TreeSA default sc_target=20",
             "reg3_250": 44.285, "sycamore_m20": 71.171},
            {"label": "cotengra hyper",
             "reg3_250": 41.678, "sycamore_m20": 63.971},
            {"label": "cotengra SA",
             "reg3_250": 40.002, "sycamore_m20": 61.679},
            {"label": "TreeSA sc_target=inf (frontier ref)",
             "reg3_250": 39.950, "sycamore_m20": 61.544},
        ],
        "attempts": {
            "reg3_250": [40.159, 40.286, 43.454, 45.037, 39.949, 45.118,
                         39.950, 40.024, 39.996, 40.110, 40.024, 39.949, 40.114],
            "sycamore_m20": [61.726, 62.049, 72.002, 66.056, 61.800, 67.084,
                             61.568, 61.505, 61.619, 61.635, 61.571, 61.607, 61.598],
        },
    }
    dump("fig2_frontier", out)


# ------------------------------------------------------------------ fig3
def fig3():
    out = {
        "rows": [
            {"label": "spectral balanced cut", "reg3_250": 11.36,
             "sycamore_m20": 5.76, "certified": True},
            {"label": "line-graph tw (minor cert.)", "reg3_250": 18.0,
             "sycamore_m20": 22.0, "certified": True},
            {"label": "Thm 1 sum-form (spectral)", "reg3_250": 13.14,
             "sycamore_m20": 9.84, "certified": True},
            {"label": "Thm 1 sum-form (empirical)", "reg3_250": 30.81,
             "sycamore_m20": 47.17, "certified": False},
            {"label": "Thm 2 dyadic-window (emp.)", "reg3_250": 27.0,
             "sycamore_m20": 40.0, "certified": False},
            {"label": "Thm 2 path-DP (emp.)", "reg3_250": 30.32,
             "sycamore_m20": 47.01, "certified": False},
            {"label": "balanced/carving cut", "reg3_250": 30.0,
             "sycamore_m20": 53.0, "certified": False},
        ],
        "caps": {"reg3_250": 30.0, "sycamore_m20": 47.0},
        "frontier": {"reg3_250": 39.95, "sycamore_m20": 61.544},
    }
    dump("fig3_bounds", out)


# ------------------------------------------------------------------ fig4
def fig4():
    out = {"meta": {"reg3_250": {"n": 250, "bisection": 30.0, "cert": None},
                    "sycamore_m20": {"n": 561, "bisection": 47.0,
                                     "cert": {"k": 141, "b": 40}}},
           "profiles": {}}
    for inst in ["reg3_250", "sycamore_m20"]:
        rows = []
        with open(ROOT / f".worktrees/attempt-031/data/{inst}_profile.csv") as f:
            for row in csv.DictReader(f):
                k = int(row["k"])
                if k < 1:
                    continue
                rows.append({"k": k, "spec": float(row["b_spec"]),
                             "emp": float(row["b_emp_windowmin"])})
        out["profiles"][inst] = rows
    dump("fig4_profiles", out)


# ------------------------------------------------------------------ fig5
def fig5():
    out = {
        "reg3_250": [
            {"label": "TreeSA-inf", "tc": 40.024, "peak": 37, "near1": 7, "near2": 17},
            {"label": "plain-tc SA", "tc": 39.882, "peak": 36, "near1": 19, "near2": 26},
            {"label": "profile-aware SA", "tc": 39.883, "peak": 36, "near1": 18, "near2": 25},
        ],
        "sycamore_m20": [
            {"label": "TreeSA-inf", "tc": 63.576, "peak": 61, "near1": 5, "near2": 6},
            {"label": "plain-tc SA", "tc": 61.703, "peak": 58, "near1": 4, "near2": 20},
            {"label": "profile-aware SA", "tc": 61.508, "peak": 57, "near1": 17, "near2": 65},
        ],
    }
    dump("fig5_conservation", out)


# ------------------------------------------------------------------ fig7
def fig7():
    out = {"events": [
        {"cycle": 1, "reg3": 47.02, "syc": 73.61,
         "note": "v1 gate: speed bar met\nby schedule hygiene"},
        {"cycle": 2, "reg3": 39.95, "syc": 61.51,
         "note": "8 mechanisms 'beat'\nreferences 20-800x"},
        {"cycle": 2.5, "reg3": 39.90, "syc": 61.53,
         "note": "tuned reference\nreclaims records"},
        {"cycle": 3, "reg3": 39.95, "syc": 61.54,
         "note": "cotengra rows;\npure-tc objective"},
        {"cycle": 4, "reg3": 39.95, "syc": 61.54,
         "note": "certified bounds;\nhierarchical nulls"},
        {"cycle": 5, "reg3": 39.95, "syc": 61.54,
         "note": "profile program;\nscorer bias fixed"},
        {"cycle": 6, "reg3": 39.95, "syc": 61.54,
         "note": "Thm 2: profile bounds\ncapped (impossibility)"},
    ], "certified_lb_sycamore": 53.0}
    dump("fig7_timeline", out)


# ------------------------------------------------------------------ fig8
def fig8():
    board = json.load(open(DATA / "record_board.json"))
    mech = {"attempt-039": "simplify-then-anneal",
            "attempt-050": "simplify-then-anneal",
            "attempt-047": "composite", "attempt-054": "waist surgery",
            "attempt-038": "VE seed"}
    instances = ["sycamore_53_20_0", "surfacecode_d21", "ksg", "reg3_1000",
                 "dbn_13", "rqc_97_m24", "nqueens_28", "qft_27"]
    labels = {"sycamore_53_20_0": "Sycamore\n53q m=20\n(3369)",
              "surfacecode_d21": "surface code\nd=21\n(2203)",
              "ksg": "king graph\nIS (5197)",
              "reg3_1000": "random\n3-regular\n(1000)",
              "dbn_13": "DBN\ninference\n(572)",
              "rqc_97_m24": "RQC 97q\nm=24\n(1238)",
              "nqueens_28": "28-queens\n(4252)", "qft_27": "QFT-27\n(405)"}
    rows = []
    for inst in instances:
        ref = board["reference_rows"][inst]["treesa-inf"]["tc"]
        rec = board["records"][inst]
        by = rec["by"]
        rows.append({"instance": inst, "label": labels[inst],
                     "delta": round(ref - rec["tc"], 4),
                     "mechanism": "reference" if by.startswith("ref:")
                     else mech.get(by, "other"),
                     "record_tc": round(rec["tc"], 3), "sc": rec["sc"]})
    dump("fig8_board", {"rows": rows})


# ------------------------------------------------------------------ fig9
def fig9():
    att = json.load(open(DATA / "mechanism_attribution.json"))
    s = att["simplification"]
    insts = ["sycamore_53_20_0", "surfacecode_d21", "nqueens_28",
             "dbn_13", "qft_27", "reg3_1000"]
    labs = ["Sycamore\n53q", "surface\ncode d21", "28-queens", "DBN",
            "QFT-27", "reg3\n1000"]
    out = {
        "shrink": [{"instance": i, "label": l, "frac": s["shrink"][i]["frac"]}
                   for i, l in zip(insts, labs)],
        "march": s["sycamore_record_march"],
        "ab_without_simplify": s["matched_budget_ab_sycamore_53"]["without_simplify_tc"],
    }
    dump("fig9_simplify", out)


# ------------------------------------------------------------------ fig10
def _parse_trace(path):
    waist, traj = [], []
    for line in open(path):
        m = re.match(r"t=(\d+)ms iter=(\d+) WAIST cost=([\d.]+) best_alt=([\d.]+) "
                     r"gap=([\d.]+) tried=\d+ sep_labels=\d+ clique_frac=([\d.]+)", line)
        if m:
            waist.append({"t": int(m.group(1)) / 1e3, "cost": float(m.group(3)),
                          "alt": float(m.group(4))})
            continue
        m = re.match(r"t=(\d+)ms iter=(\d+) REBUILD (ACCEPT|reject) new_tc=([\d.]+)", line)
        if m:
            traj.append({"t": int(m.group(1)) / 1e3,
                         "accept": m.group(3) == "ACCEPT",
                         "tc": float(m.group(4))})
            continue
        m = re.match(r"t=(\d+)ms iter=(\d+) incumb=([\d.]+)", line)
        if m:
            traj.append({"t": int(m.group(1)) / 1e3, "accept": None,
                         "tc": float(m.group(3))})
    return waist, traj


def fig10():
    w_sc = []
    for f in ["waist_trace_surfacecode.log", "waist_trace_surfacecode2.log",
              "waist_trace_surfacecode3.log"]:
        w, _ = _parse_trace(DATA / f)
        w_sc += w
    _, tr = _parse_trace(DATA / "waist_trace_surfacecode2.log")
    w_ksg, _ = _parse_trace(DATA / "waist_trace_ksg.log")
    out = {"calls_surfacecode": w_sc, "calls_ksg": w_ksg,
           "trajectory_surfacecode_run2": tr,
           "pre_surgery_record": 47.824, "official_median": 47.377}
    dump("fig10_waist", out)


# ------------------------------------------------------------------ fig11
def fig11():
    camp = json.load(open(DATA / "huawei_campaign.json"))
    p3_order = [("sycamore_53_20_0", "Sycamore\n53q"),
                ("surfacecode_d21", "surface\nd=21"),
                ("ksg", "king\ngraph"), ("reg3_1000", "reg3\n1000"),
                ("dbn_13", "DBN"), ("rqc_97_m24", "RQC\n97q")]
    dist = []
    for inst, lab in p3_order:
        d = camp["p3_distributions"][inst]
        best_m = [m for m in d if m != "ref"][0]
        c0 = st.median(d["ref"])
        dist.append({"instance": inst, "label": lab, "best_method": best_m,
                     "ref": [round(v - c0, 3) for v in d["ref"]],
                     "best": [round(v - c0, 3) for v in d[best_m]]})
    fam = []
    for dd in [9, 13, 17, 21]:
        reps = camp["p4_family"][str(dd)]
        anchor = st.median(reps["a054"])
        fam.append({"d": dd,
                    "ref": [round(v - anchor, 3) for v in reps["ref"]],
                    "surgery": [round(v - anchor, 3) for v in reps["a054"]]})
    dump("fig11_campaign", {"distributions": dist, "family": fam})


# ------------------------------------------------------------------ fig12
def fig12():
    inf = json.load(open(DATA / "uai_inference.json"))["instances"]
    order = ["uai_DBN_12", "uai_DBN_14", "uai_DBN_16",
             "uai_linkage_13", "uai_linkage_15", "uai_linkage_17",
             "uai_linkage_23", "uai_CSP_11", "uai_Grids_15", "uai_Promedus_14"]
    labs = ["DBN\n12", "DBN\n14", "DBN\n16", "link.\n13", "link.\n15",
            "link.\n17", "link.\n23", "CSP\n11", "Grids\n15", "Prom.\n14"]
    rows = []
    for inst, lab in zip(order, labs):
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
        rows.append({"instance": inst, "label": lab,
                     **{k: round(v - frontier, 3) for k, v in vals.items()}})
    dump("fig12_inference", {"rows": rows,
                             "regimes": {"dbn": [0, 2], "linkage": [3, 6]}})


# ------------------------------------------------------------------ fig13
def fig13():
    src = json.load(open(DATA / "pareto_points.json"))
    out = {"note": src["note"], "instances": {}}
    for inst, pts in src["instances"].items():
        srt = sorted(pts, key=lambda p: (p["tc"], p["t"]))
        front, best_t = [], float("inf")
        for p in srt:
            if p["t"] < best_t:
                front.append({"tc": p["tc"], "t": p["t"]})
                best_t = p["t"]
        out["instances"][inst] = {
            "points": pts,
            "front": sorted(front, key=lambda p: p["tc"]),
        }
    dump("fig13_pareto", out)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig7(); fig8(); fig9()
    fig10(); fig11(); fig12(); fig13()
