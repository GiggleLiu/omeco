#!/usr/bin/env python3
"""Run caps and stdlib-only matched-sweep summaries for attempt 062."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


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


def fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)


def make_row(argv: list[str]) -> None:
    if len(argv) != 7:
        die(
            "usage: summarize_attempt.py row "
            "<instance> <arm> <fraction> <sweep_cap> <log> <tree>"
        )
    instance, arm, fraction_text, sweep_cap, log_path, tree_path = argv[1:]
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
    if not required_config.issubset(config) or not final:
        die(f"missing attempt-062 diagnostics in {log_path}")

    row = {
        "attempt": 62,
        "instance": instance,
        "arm": arm,
        "mode": config["mode"],
        "switch_fraction": (
            float(fraction_text) if fraction_text not in {"parent", "front"} else None
        ),
        "n_reduced": int(config["n"]),
        "epoch_sweeps": int(config["epoch_sweeps"]),
        "planned_cold_sweeps": int(config["planned_cold_sweeps"]),
        "switch_cold_sweeps": int(config["switch_cold_sweeps"]),
        "sweep_cap": int(sweep_cap),
        "sweeps": int(final["sweeps"]),
        "tc": float(final["tc_final"]),
        "trajectory": [
            {
                "sweeps": int(point["sweeps"]),
                "cold_sweeps": int(point["cold_sweeps"]),
                "schedule": point["schedule"],
                "tc": float(point["tc"]),
            }
            for point in points
        ],
    }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))


def tc_at_cold(row: dict, cold_sweeps: int) -> float:
    matches = [
        point["tc"]
        for point in row["trajectory"]
        if point["cold_sweeps"] == cold_sweeps
    ]
    if not matches:
        die(
            f'missing cold-sweep checkpoint {cold_sweeps} for '
            f'{row["instance"]}/{row["arm"]}'
        )
    return matches[-1]


def report(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print("final matched-sweep results")
    print("instance         arm        switch  cold_cut  sweeps       tc")
    for row in rows:
        fraction = (
            "-" if row["switch_fraction"] is None else f'{100 * row["switch_fraction"]:.0f}%'
        )
        cold_cut = "-" if row["arm"] == "parent061" else row["switch_cold_sweeps"]
        print(
            f'{row["instance"]:<16} {row["arm"]:<10} {fraction:>6} '
            f'{str(cold_cut):>9} {row["sweeps"]:>7} {row["tc"]:>8.4f}'
        )
    mismatches = [row for row in rows if row["sweeps"] != row["sweep_cap"]]
    if mismatches:
        die("one or more runs hit the wall deadline before the matched sweep cap")

    print("\nevidence: composite early vs 061; composite final vs 059")
    print(
        "instance         switch  cold_cut  early_sweep  composite_early  "
        "parent061_early  delta_early  composite_final  front059_final  delta_final"
    )
    for instance in ("ksg", "surfacecode_d13"):
        group = [row for row in rows if row["instance"] == instance]
        parent = next((row for row in group if row["arm"] == "parent061"), None)
        front = next((row for row in group if row["arm"] == "front059"), None)
        composites = sorted(
            (row for row in group if row["arm"].startswith("switch")),
            key=lambda row: row["switch_fraction"],
        )
        if parent is None or front is None or len(composites) != 3:
            die(f"incomplete evidence arms for {instance}")
        for composite in composites:
            cold_cut = composite["switch_cold_sweeps"]
            composite_early = tc_at_cold(composite, cold_cut)
            parent_early = tc_at_cold(parent, cold_cut)
            early_sweep = next(
                point["sweeps"]
                for point in composite["trajectory"]
                if point["cold_sweeps"] == cold_cut
            )
            print(
                f'{instance:<16} {100 * composite["switch_fraction"]:>5.0f}% '
                f'{cold_cut:>9} {early_sweep:>12} {composite_early:>16.4f} '
                f'{parent_early:>15.4f} {composite_early - parent_early:>12.4f} '
                f'{composite["tc"]:>16.4f} {front["tc"]:>14.4f} '
                f'{composite["tc"] - front["tc"]:>11.4f}'
            )


def main() -> None:
    if len(sys.argv) < 2:
        die("expected subcommand: exec, supervise, row, or report")
    command = sys.argv[1]
    argv = sys.argv[1:]
    if command == "exec":
        run_capped(argv)
    elif command == "supervise":
        supervise(argv)
    elif command == "row":
        make_row(argv)
    elif command == "report" and len(argv) == 2:
        report(argv[1])
    else:
        die(f"unknown or malformed subcommand: {command}")


if __name__ == "__main__":
    main()
