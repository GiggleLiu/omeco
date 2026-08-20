#!/usr/bin/env python3
"""Run caps and stdlib-only summaries for attempt 061 diagnostics."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def die(message: str) -> "NoReturn":
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
        die("usage: summarize_attempt.py row <instance> <mode> <c> <sweep_cap> <log> <tree>")
    instance, mode, c_text, sweep_cap, log_path, tree_path = argv[1:]
    epochs = []
    final: dict[str, str] = {}
    config: dict[str, str] = {}
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("ATT_CONFIG "):
            config = fields(line)
        elif line.startswith("ATT_EPOCH "):
            epochs.append(fields(line))
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

    def total(key: str, integer: bool = False):
        values = (int(epoch[key]) if integer else float(epoch[key]) for epoch in epochs)
        return sum(values)

    in_proposals = total("in_proposals", True)
    out_proposals = total("out_proposals", True)
    in_accepts = total("in_accepts", True)
    out_accepts = total("out_accepts", True)
    row = {
        "attempt": 61,
        "instance": instance,
        "mode": mode,
        "band_bits": None if c_text == "parent" else float(c_text),
        "n_reduced": int(config["n"]),
        "sweep_cap": int(sweep_cap),
        "sweeps": int(final["sweeps"]),
        "tc": float(final["tc_final"]),
        "epoch_count": len(epochs),
        "in_band": {
            "proposals": in_proposals,
            "accepts": in_accepts,
            "acceptance_rate": in_accepts / in_proposals if in_proposals else None,
            "net_gain_bits": total("in_net_gain"),
            "downhill_gain_bits": total("in_downhill_gain"),
        },
        "outside_band": {
            "proposals": out_proposals,
            "accepts": out_accepts,
            "acceptance_rate": out_accepts / out_proposals if out_proposals else None,
            "net_gain_bits": total("out_net_gain"),
            "downhill_gain_bits": total("out_downhill_gain"),
        },
        "waist_trajectory": [
            {
                "sweep": int(epoch["sweep"]),
                "before": float(epoch["waist_before"]),
                "after": float(epoch["waist_after"]),
            }
            for epoch in epochs
        ],
        "band_fraction_mean": (
            sum(int(epoch["band_nodes"]) / int(epoch["internal_nodes"]) for epoch in epochs)
            / len(epochs)
            if epochs
            else None
        ),
    }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))


def report(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print("instance         mode   c   sweeps       tc  in_accept out_accept in_net_gain out_net_gain")
    for row in rows:
        inside, outside = row["in_band"], row["outside_band"]
        c = "-" if row["band_bits"] is None else f'{row["band_bits"]:g}'
        print(
            f'{row["instance"]:<16} {row["mode"]:<6} {c:>2} {row["sweeps"]:>8} '
            f'{row["tc"]:>8.4f} {inside["acceptance_rate"] or 0:>9.4f} '
            f'{outside["acceptance_rate"] or 0:>10.4f} {inside["net_gain_bits"]:>11.3f} '
            f'{outside["net_gain_bits"]:>12.3f}'
        )
    mismatches = [row for row in rows if row["sweeps"] != row["sweep_cap"]]
    if mismatches:
        die("one or more runs hit the wall deadline before the matched sweep cap")


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
