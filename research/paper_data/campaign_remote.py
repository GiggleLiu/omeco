#!/usr/bin/env python3
"""Paper data campaign, remote edition (huawei ECS, 2 cores).

Same job matrix as research/paper_data/campaign.py but run fresh on ONE quiet
Linux machine so every matched-budget comparison is same-hardware. lanes=2
default (one timed run per core; scorer overlap is brief and symmetric across
methods). Resumable: completed jobs (score.json present) are skipped.
"""

import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path.home() / "omeco_campaign"
OUT = ROOT / "runs"
TARGETS = ROOT / "targets"
SCORER = ROOT / "scorer.py"
WT = ROOT / "worktrees"

BIN = {
    "ref": WT / "attempt-054/target/release/examples/treesa_tuned",
    "a054": WT / "attempt-054/target/release/examples/attempt",
    "a050": WT / "attempt-050/target/release/examples/attempt",
    "a038": WT / "attempt-038/target/release/examples/attempt",
}

BEST = {  # instance -> strongest attempt binary for P3
    "sycamore_53_20_0": "a050", "surfacecode_d21": "a054", "ksg": "a054",
    "reg3_1000": "a054", "dbn_13": "a038", "rqc_97_m24": "a054",
}


def jobs():
    out = []
    # P2 budget scaling
    for inst in ["sycamore_53_20_0", "surfacecode_d21", "reg3_1000"]:
        for b in [90, 300, 900]:
            for m in ["ref", "a054", "a050"]:
                for r in range(3):
                    out.append((m, inst, b, f"p2_r{r}"))
    # P3 distributions
    for inst, best in BEST.items():
        for m in ["ref", best]:
            for r in range(15):
                out.append((m, inst, 90, f"p3_r{r}"))
    # P4 family trend
    for d in [9, 13, 17, 21]:
        inst = f"surfacecode_d{d}"
        for m in ["ref", "a054"]:
            for r in range(5):
                out.append((m, inst, 90, f"p4_r{r}"))
    # P5 waist-minimality traces on converged primaries
    for inst in ["reg3_250", "sycamore_m20"]:
        for r in range(2):
            out.append(("a054", inst, 90, f"p5_r{r}"))
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
    tree.unlink()  # trees are large; the score is the datum
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
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
