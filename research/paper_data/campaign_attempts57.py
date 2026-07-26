#!/usr/bin/env python3
"""Official runs for attempts 057 (scalable quotient-graph VE seed) and 058
(adaptive width-capped peel ladder), huawei. Two matrices under the standard
protocol (RAYON_NUM_THREADS=1, independent rescoring, failures persisted):
  A. relational family: 5 instances x {a057, a058} x (90s x3 + 300s x3 + 900s x1)
  B. UAI hard ten:     10 instances x {a057, a058} x 90s x 5
= 70 + 100 = 170 jobs.  Results land in relational_runs/ and uai_runs/ next to
the existing arms, same directory naming, so the analysis scripts see one pool."""

import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path.home() / "omeco_campaign"
TARGETS = ROOT / "uai_mar"
SCORER = ROOT / "scorer.py"
WT = ROOT / "worktrees"

BIN = {
    "a057": WT / "attempt-057/target/release/examples/attempt",
    "a058": WT / "attempt-058/target/release/examples/attempt",
}

RELATIONAL = [
    "uai_relational_3", "uai_relational_2", "uai_relational_5",
    "uai_relational_4", "uai_relational_1",
]
UAI_TEN = [
    "uai_DBN_16", "uai_DBN_12", "uai_DBN_14",
    "uai_linkage_15", "uai_linkage_13", "uai_linkage_23", "uai_linkage_17",
    "uai_CSP_11", "uai_Grids_15", "uai_Promedus_14",
]


def jobs():
    out = []
    for inst in RELATIONAL:
        for m in ("a057", "a058"):
            for b, reps in ((90, 3), (300, 3), (900, 1)):
                for r in range(reps):
                    out.append(("relational_runs", m, inst, b, f"r{r}"))
    for inst in UAI_TEN:
        for m in ("a057", "a058"):
            for r in range(5):
                out.append(("uai_runs", m, inst, 90, f"r{r}"))
    return out


def run_job(job):
    outdir, m, inst, budget, tag = job
    g = TARGETS / f"{inst}.json"
    if not g.exists():
        return (job, "missing-instance")
    d = ROOT / outdir / f"{inst}__{m}__{budget}s__{tag}"
    d.mkdir(parents=True, exist_ok=True)
    score_p = d / "score.json"
    if score_p.exists():
        return (job, "done")
    tree = d / "out.json"
    err = d / "stderr.log"
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    t0 = time.monotonic()
    try:
        with open(err, "wb") as ef:
            subprocess.run([str(BIN[m]), str(g), str(int(budget * 1000)),
                            str(tree)], env=env, stdout=subprocess.DEVNULL,
                           stderr=ef, timeout=budget * 1.2 + 180)
    except subprocess.TimeoutExpired:
        pass
    wall = time.monotonic() - t0
    if not tree.exists():
        res = {"error": "no-output", "method": m, "instance": inst,
               "budget_s": budget, "tag": tag, "wall_s": round(wall, 1),
               "host": "huawei-ecs-2core"}
        json.dump(res, open(score_p, "w"))
        return (job, "no-output (persisted)")
    try:
        sc = subprocess.run([sys.executable, str(SCORER), str(g), str(tree)],
                            capture_output=True, text=True, timeout=1800)
        res = json.loads(sc.stdout)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"score-fail: {e}", "method": m, "instance": inst,
               "budget_s": budget, "tag": tag, "wall_s": round(wall, 1),
               "host": "huawei-ecs-2core"}
        json.dump(res, open(score_p, "w"))
        return (job, "score-fail (persisted)")
    res.update({"method": m, "instance": inst, "budget_s": budget, "tag": tag,
                "wall_s": round(wall, 1), "host": "huawei-ecs-2core"})
    json.dump(res, open(score_p, "w"))
    tree.unlink()
    return (job, f"tc={res['tc']:.3f} sc={res['sc']:.1f}")


def main():
    lanes = 2
    if "--lanes" in sys.argv:
        lanes = int(sys.argv[sys.argv.index("--lanes") + 1])
    q = jobs()
    todo = [j for j in q
            if not (ROOT / j[0] / f"{j[2]}__{j[1]}__{j[3]}s__{j[4]}" / "score.json").exists()]
    print(f"{len(q)} jobs total, {len(todo)} to run, lanes={lanes}", flush=True)
    if "--dry" in sys.argv:
        return
    with ThreadPoolExecutor(max_workers=lanes) as ex:
        for job, status in ex.map(run_job, todo):
            print(f"[{time.strftime('%H:%M:%S')}] {job[2]} {job[1]} {job[3]}s {job[4]}: {status}",
                  flush=True)
    print("ATTEMPTS57-CAMPAIGN-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
