#!/usr/bin/env python3
"""Caps, relabeling, trace annotation, and stdlib summaries for attempt 065."""

from __future__ import annotations

import json
import os
import random
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn


TRACE_FIELDS = {
    "t_ms",
    "sweep",
    "epoch",
    "epoch_sweep",
    "mode",
    "phase",
    "span",
    "tc",
    "sc",
    "max_node_cost",
    "in_band_accepted_gain",
    "out_band_accepted_gain",
    "band_size",
    "band_jaccard",
    "band_churn",
}


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def kill(process: subprocess.Popen[bytes], process_group: bool) -> None:
    """Immediately kill a command and, when requested, its full process group."""
    try:
        if process_group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    except ProcessLookupError:
        pass


def run_capped(argv: list[str]) -> None:
    if len(argv) < 6 or argv[4] != "--":
        die("usage: summarize_attempt.py exec <seconds> <stdout> <stderr> -- <command...>")
    seconds = float(argv[1])
    with open(argv[2], "wb") as stdout, open(argv[3], "wb") as stderr:
        nested = os.environ.get("ATT_BENCH_GUARDED") == "1"
        process = subprocess.Popen(
            argv[5:], stdout=stdout, stderr=stderr, start_new_session=not nested
        )
        try:
            returncode = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            kill(process, process_group=not nested)
            die(f"command exceeded {seconds:g}s cap: {' '.join(argv[5:])}")
    if returncode:
        die(f"command failed ({returncode}); stderr: {argv[3]}")


def supervise(argv: list[str]) -> None:
    if len(argv) < 4 or argv[2] != "--":
        die("usage: summarize_attempt.py supervise <seconds> -- <command...>")
    seconds = float(argv[1])
    process = subprocess.Popen(argv[3:], env=os.environ.copy(), start_new_session=True)
    try:
        returncode = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        kill(process, process_group=True)
        die(f"benchmark exceeded hard {seconds:g}s total wall cap")
    if returncode:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise SystemExit(returncode)


def relabel(argv: list[str]) -> None:
    if len(argv) != 4:
        die("usage: summarize_attempt.py relabel <input.json> <output.json> <seed>")
    source, destination, seed_text = argv[1:]
    graph = json.loads(Path(source).read_text())
    labels = sorted(
        {int(label) for ix in graph["ixs"] for label in ix}
        | {int(label) for label in graph["iy"]}
        | {int(label) for label in graph["sizes"]}
    )
    shuffled = labels.copy()
    random.Random(int(seed_text)).shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))
    graph["ixs"] = [[mapping[int(label)] for label in ix] for ix in graph["ixs"]]
    graph["iy"] = [mapping[int(label)] for label in graph["iy"]]
    graph["sizes"] = {str(mapping[int(label)]): size for label, size in graph["sizes"].items()}
    Path(destination).write_text(json.dumps(graph, sort_keys=True, separators=(",", ":")))


def annotate(argv: list[str]) -> None:
    if len(argv) != 6:
        die("usage: summarize_attempt.py annotate <instance> <arm> <relabel> <seed> <trace>")
    instance, arm, relabel_text, seed_text, trace_path = argv[1:]
    count = 0
    with open(trace_path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = TRACE_FIELDS - record.keys()
            if missing:
                die(f"{trace_path}:{line_number}: missing fields {sorted(missing)}")
            if record["mode"] != arm:
                die(f"{trace_path}:{line_number}: expected mode={arm}, got {record['mode']}")
            record.update(
                attempt=65,
                instance=instance,
                arm=arm,
                relabel=int(relabel_text),
                relabel_seed=int(seed_text),
            )
            print(json.dumps(record, sort_keys=True, separators=(",", ":")))
            count += 1
    if count == 0:
        die(f"empty trace: {trace_path}")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


@dataclass
class RunSummary:
    min_tc: float = float("inf")
    min_sc: float = float("inf")
    min_max_node_cost: float = float("inf")
    total_in_gain: float = 0.0
    total_out_gain: float = 0.0
    descent_in_gain: float = 0.0
    descent_out_gain: float = 0.0
    first_t_by_sc: dict[float, float] = field(default_factory=dict)
    best_tc_by_sc: dict[float, float] = field(default_factory=dict)
    last_epoch: int | None = None
    jaccard_sum: float = 0.0
    jaccard_count: int = 0
    churn_sum: float = 0.0
    churn_count: int = 0
    band_size_sum: float = 0.0
    epoch_count: int = 0
    sweeps: int = 0

    def add(self, record: dict[str, object]) -> None:
        tc = float(record["tc"])
        sc = round(float(record["sc"]), 6)
        in_gain = float(record["in_band_accepted_gain"])
        out_gain = float(record["out_band_accepted_gain"])
        self.total_in_gain += in_gain
        self.total_out_gain += out_gain
        self.min_sc = min(self.min_sc, sc)
        self.min_max_node_cost = min(
            self.min_max_node_cost, float(record["max_node_cost"])
        )
        self.first_t_by_sc.setdefault(sc, float(record["t_ms"]))
        self.best_tc_by_sc[sc] = min(self.best_tc_by_sc.get(sc, float("inf")), tc)
        if tc < self.min_tc:
            self.min_tc = tc
            self.descent_in_gain = self.total_in_gain
            self.descent_out_gain = self.total_out_gain

        epoch = int(record["epoch"])
        if epoch != self.last_epoch:
            self.last_epoch = epoch
            self.epoch_count += 1
            self.band_size_sum += float(record["band_size"])
            if record["band_jaccard"] is not None:
                self.jaccard_sum += float(record["band_jaccard"])
                self.jaccard_count += 1
            if record["band_churn"] is not None:
                self.churn_sum += float(record["band_churn"])
                self.churn_count += 1
        self.sweeps += 1

    def first_time_at_or_below(self, target: float) -> float:
        return min(time for sc, time in self.first_t_by_sc.items() if sc <= target + 1e-6)


def paired_metrics(
    band: RunSummary, parent: RunSummary
) -> dict[str, float]:
    shared = band.first_t_by_sc.keys() & parent.first_t_by_sc.keys()
    if not shared:
        die("band and parent traces have no shared sc value")
    shared_sc = min(shared)
    return {
        "shared_sc": shared_sc,
        "delta_t_shared_sc_ms": band.first_time_at_or_below(shared_sc)
        - parent.first_time_at_or_below(shared_sc),
        "delta_tc_at_shared_sc": band.best_tc_by_sc[shared_sc]
        - parent.best_tc_by_sc[shared_sc],
        "delta_min_sc": band.min_sc - parent.min_sc,
        "delta_min_max_node_cost": band.min_max_node_cost - parent.min_max_node_cost,
        "band_in_gain_descent": band.descent_in_gain,
        "band_in_gain_late": band.total_in_gain - band.descent_in_gain,
        "band_out_gain_descent": band.descent_out_gain,
        "band_out_gain_late": band.total_out_gain - band.descent_out_gain,
        "band_jaccard": (
            band.jaccard_sum / band.jaccard_count
            if band.jaccard_count
            else float("nan")
        ),
        "band_churn": (
            band.churn_sum / band.churn_count if band.churn_count else float("nan")
        ),
        "band_size": band.band_size_sum / band.epoch_count,
        "band_sweeps": float(band.sweeps),
        "parent_sweeps": float(parent.sweeps),
    }


def report(path: str) -> None:
    groups: dict[tuple[str, int, str], RunSummary] = {}
    with open(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("attempt", -1)) != 65:
                die("trace contains a non-attempt-065 record")
            key = (str(record["instance"]), int(record["relabel"]), str(record["arm"]))
            groups.setdefault(key, RunSummary()).add(record)

    instances = sorted({key[0] for key in groups})
    print(
        "instance          reps shared_sc  d_t_sc_ms  d_tc@sc  d_min_sc d_nodecost "
        "in_gain_desc in_gain_late out_gain_desc jaccard churn band_size sweeps_b/p"
    )
    for instance in instances:
        relabels = sorted({key[1] for key in groups if key[0] == instance})
        metrics = []
        for relabel_index in relabels:
            band = groups.get((instance, relabel_index, "band"))
            parent = groups.get((instance, relabel_index, "parent"))
            if band is None or parent is None:
                die(f"missing paired arm for {instance} relabel {relabel_index}")
            metrics.append(paired_metrics(band, parent))
        if len(metrics) != 3:
            die(f"expected 3 relabelings for {instance}, got {len(metrics)}")

        value = lambda field: median([metric[field] for metric in metrics])
        print(
            f"{instance:<17} {len(metrics):>4} {value('shared_sc'):>9.3f} "
            f"{value('delta_t_shared_sc_ms'):>10.1f} {value('delta_tc_at_shared_sc'):>8.3f} "
            f"{value('delta_min_sc'):>8.3f} {value('delta_min_max_node_cost'):>10.3f} "
            f"{value('band_in_gain_descent'):>12.3f} {value('band_in_gain_late'):>12.3f} "
            f"{value('band_out_gain_descent'):>13.3f} {value('band_jaccard'):>7.3f} "
            f"{value('band_churn'):>5.3f} {value('band_size'):>9.1f} "
            f"{value('band_sweeps'):.0f}/{value('parent_sweeps'):.0f}"
        )
    print("definitions: deltas=band-parent; descent=through trace tc minimum; epoch persistence=mean descendant-set Jaccard")


def main() -> None:
    if len(sys.argv) < 2:
        die("expected subcommand: exec, supervise, relabel, annotate, or report")
    command = sys.argv[1]
    argv = sys.argv[1:]
    if command == "exec":
        run_capped(argv)
    elif command == "supervise":
        supervise(argv)
    elif command == "relabel":
        relabel(argv)
    elif command == "annotate":
        annotate(argv)
    elif command == "report" and len(argv) == 2:
        report(argv[1])
    else:
        die(f"unknown or malformed subcommand: {command}")


if __name__ == "__main__":
    main()
