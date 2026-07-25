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
OUT = ROOT / "addendum_runs"
TARGETS = ROOT / "targets"
SCORER = ROOT / "scorer.py"
WT = ROOT / "worktrees"

BIN = {
    "ref": WT / "attempt-054/target/release/examples/treesa_tuned",
    "a054": WT / "attempt-054/target/release/examples/attempt",
    "a050": WT / "attempt-050/target/release/examples/attempt",
    "a038": WT / "attempt-038/target/release/examples/attempt",
}

INSTANCES = ["nqueens_28", "qft_27"]


def jobs():
    out = []
    for inst in INSTANCES:
        for m in ["ref", "a054", "a038"]:
            for r in range(5):
                out.append((m, inst, 90, f"r{r}"))
    return out


def run_job(job):
    m, inst, budget, tag = job
    g = TARGETS / f"{inst}.json"
    if not g.exists():
        return (job, "missing-instance")
    d = OUT / f"{inst}__{m}__{budget}s__{tag}"
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
                           stderr=ef, timeout=budget * 1.15 + 60)
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
    print("ADDENDUM-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
