#!/usr/bin/env python3
"""Resumable measurement campaign for the paper's data-strengthening plan.

Runs a queue of (binary, instance, budget, seed-tag) jobs with skip-if-done
logic, LANES-way parallel, rescoring every emitted tree with the independent
scorer. Direct measurement runs — never touches the validator leaderboard.

Tracks:
  P2 budget scaling: ref(treesa_tuned) + 054 + 050 at 90/300/900 s on
     sycamore_53_20_0, surfacecode_d21, reg3_1000 (3 reps each).
  P3 distributions: ref + best attempt, 15 reps at 90 s, on the 6 main
     instances.
  P4 family trend: surfacecode d=9/13/17/21, ref + 054, 5 reps at 90 s.
  P5 waist-minimality on converged primaries: 054 on reg3_250/sycamore_m20
     (traces land in stderr logs; 2 reps).

Usage: python3 research/paper_data/campaign.py [--lanes N] [--dry]
Re-run after any kill: completed jobs are skipped (out.json + score present).
"""

import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path("/Users/liujinguo/rcode/omeco")
OUT = ROOT / "research/paper_data/runs"
TARGETS = ROOT / "research/benchmark/targets"
SCORER = ROOT / "research/validator/scorer.py"

BIN = {
    "ref": ROOT / "research/validator/bin/treesa_tuned",
    "a054": ROOT / ".worktrees/attempt-054/target/release/examples/attempt",
    "a050": ROOT / ".worktrees/attempt-050/target/release/examples/attempt",
    "a047": ROOT / ".worktrees/attempt-047/target/release/examples/attempt",
    "a039": ROOT / ".worktrees/attempt-039/target/release/examples/attempt",
    "a038": ROOT / ".worktrees/attempt-038/target/release/examples/attempt",
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
    # P5 waist-minimality on converged primaries (traces via stderr)
    for inst in ["reg3_250", "sycamore_m20"]:
        for r in range(2):
            out.append(("a054", inst, 90, f"p5_r{r}"))
    # dedup (P2/P3 overlap at 90 s is fine to keep separate reps)
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
                           stderr=ef, timeout=budget * 1.15 + 30)
    except subprocess.TimeoutExpired:
        pass
    wall = time.monotonic() - t0
    if not tree.exists():
        return (job, "no-output")
    try:
        sc = subprocess.run([sys.executable, str(SCORER), str(g), str(tree)],
                            capture_output=True, text=True, timeout=600)
        res = json.loads(sc.stdout)
    except Exception as e:  # noqa: BLE001
        return (job, f"score-fail: {e}")
    res.update({"method": m, "instance": inst, "budget_s": budget, "tag": tag,
                "wall_s": round(wall, 1)})
    json.dump(res, open(score_p, "w"))
    tree.unlink()  # trees are large; the score is the datum
    return (job, f"tc={res['tc']:.3f}")


def main():
    lanes = 4
    if "--lanes" in sys.argv:
        lanes = int(sys.argv[sys.argv.index("--lanes") + 1])
    q = jobs()
    todo = [j for j in q if not (OUT / f"{j[1]}__{j[0]}__{j[2]}s__{j[3]}" / "score.json").exists()]
    print(f"{len(q)} jobs total, {len(todo)} to run, lanes={lanes}")
    if "--dry" in sys.argv:
        return
    with ThreadPoolExecutor(max_workers=lanes) as ex:
        for job, status in ex.map(run_job, todo):
            print(f"[{time.strftime('%H:%M:%S')}] {job[1]} {job[0]} {job[2]}s {job[3]}: {status}",
                  flush=True)


if __name__ == "__main__":
    main()
