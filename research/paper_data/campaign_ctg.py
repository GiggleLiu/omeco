#!/usr/bin/env python3
"""UAI-2014 MAR hard-instance batch (huawei). Same protocol as
campaign_remote.py: 90 s budget, RAYON_NUM_THREADS=1, independent rescoring,
skip-if-done. 10 instances x {ref, a038, a050, a054} x 5 reps = 200 jobs."""

import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path.home() / "omeco_campaign"
OUT = ROOT / "ctg_runs"
TARGETS = ROOT / "targets"
UAI = ROOT / "uai_mar"
VENV_PY = ROOT / "ctg_venv/bin/python"
CTG = ROOT / "run_cotengra.py"
SCORER = ROOT / "scorer.py"
WT = ROOT / "worktrees"

BIN = {
    "ref": WT / "attempt-054/target/release/examples/treesa_tuned",
    "a054": WT / "attempt-054/target/release/examples/attempt",
    "a050": WT / "attempt-050/target/release/examples/attempt",
    "a038": WT / "attempt-038/target/release/examples/attempt",
}

INSTANCES = [
    "sycamore_53_20_0", "surfacecode_d21", "surfacecode_d17", "surfacecode_d13",
    "surfacecode_d9", "ksg", "reg3_1000", "nqueens_28", "dbn_13", "qft_27",
    "rqc_97_m24", "reg3_250", "sycamore_m20",
    "uai_DBN_16", "uai_DBN_12", "uai_DBN_14", "uai_linkage_15",
    "uai_linkage_13", "uai_linkage_23", "uai_linkage_17", "uai_CSP_11",
    "uai_Grids_15", "uai_Promedus_14",
]


def jobs():
    out = []
    for inst in INSTANCES:
        for m in ["ctg-sa"]:
            for r in range(3):
                out.append((m, inst, 90, f"r{r}"))
    return out


def run_job(job):
    m, inst, budget, tag = job
    g = (UAI if inst.startswith("uai_") else TARGETS) / f"{inst}.json"
    if not g.exists():
        return (job, "missing-instance")
    d = OUT / f"{inst}__{m}__{budget}s__{tag}"
    d.mkdir(parents=True, exist_ok=True)
    score_p = d / "score.json"
    if score_p.exists():
        return (job, "done")
    tree = d / "out.json"
    err = d / "stderr.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t0 = time.monotonic()
    try:
        with open(err, "wb") as ef:
            subprocess.run([str(VENV_PY), str(CTG), str(g),
                            str(int(budget * 1000)), str(tree), "sa"],
                           env=env, stdout=subprocess.DEVNULL,
                           stderr=ef, timeout=budget * 1.15 + 120)
    except subprocess.TimeoutExpired:
        pass
    wall = time.monotonic() - t0
    if not tree.exists():
        return (job, "no-output")
    try:
        sc = subprocess.run([sys.executable, str(SCORER), str(g), str(tree)],
                            capture_output=True, text=True, timeout=900)
        res = json.loads(sc.stdout)
    except Exception as e:  # noqa: BLE001
        return (job, f"score-fail: {e}")
    res.update({"method": m, "instance": inst, "budget_s": budget, "tag": tag,
                "wall_s": round(wall, 1), "host": "huawei-ecs-2core"})
    json.dump(res, open(score_p, "w"))
    tree.unlink()
    return (job, f"tc={res['tc']:.3f}")


def main():
    lanes = 2
    if "--lanes" in sys.argv:
        lanes = int(sys.argv[sys.argv.index("--lanes") + 1])
    q = jobs()
    todo = [j for j in q
            if not (OUT / f"{j[1]}__{j[0]}__{j[2]}s__{j[3]}" / "score.json").exists()]
    print(f"{len(q)} jobs total, {len(todo)} to run, lanes={lanes}", flush=True)
    if "--dry" in sys.argv:
        return
    with ThreadPoolExecutor(max_workers=lanes) as ex:
        for job, status in ex.map(run_job, todo):
            print(f"[{time.strftime('%H:%M:%S')}] {job[1]} {job[0]} {job[2]}s {job[3]}: {status}",
                  flush=True)
    print("CTG-CAMPAIGN-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
