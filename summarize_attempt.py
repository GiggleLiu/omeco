#!/usr/bin/env python3
"""Caps, relabeling, and stdlib-only tc(t) summaries for attempt 066."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


RECORD_EPS = 39.883325463011175 + 0.05
RECORD_EPS_TEXT = "39.883325463011175 + 0.05 = 39.933325463011175"


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def kill(process: subprocess.Popen, process_group: bool) -> None:
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
        # The shell may have exited while a compiler or optimizer descendant
        # remained. Clear the isolated group before returning failure.
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
    graph["sizes"] = {
        str(mapping[int(label)]): size for label, size in graph["sizes"].items()
    }
    Path(destination).write_text(
        json.dumps(graph, sort_keys=True, separators=(",", ":"))
    )


def fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)


def make_row(argv: list[str]) -> None:
    if len(argv) != 9:
        die(
            "usage: summarize_attempt.py row "
            "<instance> <relabel> <seed> <arm> <fraction> <sweep_cap> <log> <tree>"
        )
    (
        instance,
        relabel_text,
        seed_text,
        arm,
        fraction_text,
        sweep_cap,
        log_path,
        tree_path,
    ) = argv[1:]
    points = []
    final: dict[str, str] = {}
    config: dict[str, str] = {}
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("ATT_CONFIG "):
            config = fields(line)
        elif line.startswith("ATT_POINT "):
            points.append(fields(line))
        elif " tc_final=" in line:
            final = fields(line)
    with open(tree_path) as stream:
        tree = json.load(stream)
    if (
        not isinstance(tree, dict)
        or not isinstance(tree.get("inputs"), list)
        or not isinstance(tree.get("output"), list)
        or not isinstance(tree.get("tree"), dict)
        or "isleaf" not in tree["tree"]
    ):
        die(f"invalid NestedEinsum JSON in {tree_path}")

    required_config = {
        "mode",
        "n",
        "epoch_sweeps",
        "planned_cold_sweeps",
        "switch_cold_sweeps",
    }
    if not required_config.issubset(config) or not final or not points:
        die(f"missing attempt-066 diagnostics in {log_path}")
    if config["mode"] != "composite":
        die(f"expected composite mode in {log_path}, got {config['mode']}")
    if abs(float(config["switch_fraction"]) - float(fraction_text)) > 1e-12:
        die(f"fixed-switch override was not honored in {log_path}")

    row = {
        "attempt": 66,
        "instance": instance,
        "relabel": int(relabel_text),
        "relabel_seed": int(seed_text),
        "arm": arm,
        "mode": config["mode"],
        "switch_fraction": float(fraction_text),
        "n_reduced": int(config["n"]),
        "epoch_sweeps": int(config["epoch_sweeps"]),
        "planned_cold_sweeps": int(config["planned_cold_sweeps"]),
        "switch_cold_sweeps": int(config["switch_cold_sweeps"]),
        "sweep_cap": int(sweep_cap) if sweep_cap else None,
        "sweeps": int(final["sweeps"]),
        "tc": float(final["tc_final"]),
        "trajectory": [
            {
                "t_ms": float(point["t_ms"]),
                "sweeps": int(point["sweeps"]),
                "cold_sweeps": int(point["cold_sweeps"]),
                "schedule": point["schedule"],
                "tc": float(point["tc"]),
            }
            for point in points
        ],
    }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))


def first_time_at_or_below(row: dict, target: float) -> float | None:
    return next(
        (point["t_ms"] for point in row["trajectory"] if point["tc"] <= target),
        None,
    )


def report(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print("final wall-matched results and tc(t) coverage")
    print("instance         relabel seed   arm   switch cold_cut sweeps points       tc")
    for row in rows:
        fraction = f'{100 * row["switch_fraction"]:.0f}%'
        print(
            f'{row["instance"]:<16} {row["relabel"]:>7} {row["relabel_seed"]:>5} '
            f'{row["arm"]:<5} {fraction:>6} {row["switch_cold_sweeps"]:>8} '
            f'{row["sweeps"]:>6} {len(row["trajectory"]):>6} {row["tc"]:>8.4f}'
        )
    mismatches = [
        row
        for row in rows
        if row["sweep_cap"] is not None and row["sweeps"] != row["sweep_cap"]
    ]
    if mismatches:
        die("one or more runs hit the wall deadline before the matched sweep cap")

    expected = {
        (instance, relabel, arm)
        for instance in ("reg3_250", "ksg")
        for relabel in (0, 1)
        for arm in ("q025", "q040")
    }
    actual = {(row["instance"], row["relabel"], row["arm"]) for row in rows}
    if actual != expected:
        die(
            f"incomplete evidence matrix: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )

    reg3 = [row for row in rows if row["instance"] == "reg3_250"]
    earlier_or_equal = 0
    comparisons = 0
    print(
        f"\nreg3_250 time-to-record-eps (tc <= {RECORD_EPS_TEXT}): "
        "q0.40 vs q0.25 cross-relabel"
    )
    print("q040_relabel q025_relabel q040_t_ms q025_t_ms earlier_or_equal")
    for later in sorted(
        (row for row in reg3 if row["arm"] == "q040"),
        key=lambda row: row["relabel"],
    ):
        later_time = first_time_at_or_below(later, RECORD_EPS)
        for earlier in sorted(
            (row for row in reg3 if row["arm"] == "q025"),
            key=lambda row: row["relabel"],
        ):
            earlier_time = first_time_at_or_below(earlier, RECORD_EPS)
            wins = later_time is not None and (
                earlier_time is None or later_time <= earlier_time
            )
            comparisons += 1
            earlier_or_equal += int(wins)
            later_text = "not-reached" if later_time is None else f"{later_time:.3f}"
            earlier_text = "not-reached" if earlier_time is None else f"{earlier_time:.3f}"
            print(
                f'{later["relabel"]:>12} {earlier["relabel"]:>12} '
                f"{later_text:>9} {earlier_text:>9} {str(wins).lower():>16}"
            )
    print(
        f"result: q0.40 earlier-or-equal in {earlier_or_equal}/{comparisons} "
        "comparisons (target >=3/4)"
    )


def main() -> None:
    if len(sys.argv) < 2:
        die("expected subcommand: exec, supervise, relabel, row, or report")
    command = sys.argv[1]
    argv = sys.argv[1:]
    if command == "exec":
        run_capped(argv)
    elif command == "supervise":
        supervise(argv)
    elif command == "relabel":
        relabel(argv)
    elif command == "row":
        make_row(argv)
    elif command == "report" and len(argv) == 2:
        report(argv[1])
    else:
        die(f"unknown or malformed subcommand: {command}")


if __name__ == "__main__":
    main()
