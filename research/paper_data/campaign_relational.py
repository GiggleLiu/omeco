#!/usr/bin/env python3
"""Beyond-the-benchmark batch: the five UAI-2014 relational instances that
TensorInference.jl's own test suite excludes as too large (issue #15;
3k-70k tensors). Same protocol as campaign_uai.py, plus: failed runs are
persisted (score.json with an error field) so resume does not retry them,
and budgets extend to 900 s because 90 s is construction-dominated at this
scale. 5 instances x {ref, a038, a050, a054} x (90s x3 + 300s x3 + 900s x1)
= 140 jobs."""

import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path.home() / "omeco_campaign"
OUT = ROOT / "relational_runs"
TARGETS = ROOT / "uai_mar"
SCORER = ROOT / "scorer.py"
WT = ROOT / "worktrees"

BIN = {
    "ref": WT / "attempt-054/target/release/examples/treesa_tuned",
    "a054": WT / "attempt-054/target/release/examples/attempt",
    "a050": WT / "attempt-050/target/release/examples/attempt",
    "a038": WT / "attempt-038/target/release/examples/attempt",
}

INSTANCES = [
    "uai_relational_3", "uai_relational_2", "uai_relational_5",
    "uai_relational_4", "uai_relational_1",
]


def jobs():
    out = []
    for inst in INSTANCES:
        for m in ["ref", "a038", "a050", "a054"]:
            for b, reps in ((90, 3), (300, 3), (900, 1)):
                for r in range(reps):
                    out.append((m, inst, b, f"r{r}"))
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
            if not (OUT / f"{j[1]}__{j[0]}__{j[2]}s__{j[3]}" / "score.json").exists()]
    print(f"{len(q)} jobs total, {len(todo)} to run, lanes={lanes}", flush=True)
    if "--dry" in sys.argv:
        return
    with ThreadPoolExecutor(max_workers=lanes) as ex:
        for job, status in ex.map(run_job, todo):
            print(f"[{time.strftime('%H:%M:%S')}] {job[1]} {job[0]} {job[2]}s {job[3]}: {status}",
                  flush=True)
    print("RELATIONAL-CAMPAIGN-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
