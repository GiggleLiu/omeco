#!/usr/bin/env python3
"""Run caps and stdlib-only evidence summaries for attempt 064."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


FAMILIES = {
    "ksg": "ksg",
    "reg3_250": "expander",
    "surfacecode_d13": "separable",
}


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
    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    for handled in handled_signals:
        previous_handlers[handled] = signal.signal(handled, interrupted)
    try:
        returncode = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        kill(process, process_group=True)
        die(f"benchmark exceeded hard {seconds:g}s total wall cap")
    except BaseException:
        kill(process, process_group=True)
        raise
    finally:
        for handled, previous in previous_handlers.items():
            signal.signal(handled, previous)
    if returncode:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise SystemExit(returncode)


def fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)


def validate_tree(path: str) -> None:
    with open(path) as stream:
        tree = json.load(stream)
    if (
        not isinstance(tree, dict)
        or not isinstance(tree.get("inputs"), list)
        or not isinstance(tree.get("output"), list)
        or not isinstance(tree.get("tree"), dict)
        or "isleaf" not in tree["tree"]
    ):
        die(f"invalid NestedEinsum JSON in {path}")


def make_row(argv: list[str]) -> None:
    if len(argv) != 7:
        die(
            "usage: summarize_attempt.py row "
            "<instance> <arm> <parameter> <sweep_cap> <log> <tree>"
        )
    instance, arm, parameter, sweep_cap, log_path, tree_path = argv[1:]
    config: dict[str, str] = {}
    switch: dict[str, str] | None = None
    final: dict[str, str] = {}
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("ATT_CONFIG "):
            config = fields(line)
        elif line.startswith("ATT_SWITCH "):
            switch = fields(line)
        elif " tc_final=" in line:
            final = fields(line)
    validate_tree(tree_path)

    required_config = {
        "mode",
        "n",
        "epoch_sweeps",
        "planned_cold_sweeps",
        "stall_window_sweeps",
    }
    if not required_config.issubset(config) or not final:
        die(f"missing attempt-064 diagnostics in {log_path}")

    row = {
        "attempt": 64,
        "instance": instance,
        "family": FAMILIES[instance],
        "arm": arm,
        "parameter": float(parameter),
        "mode": config["mode"],
        "n_reduced": int(config["n"]),
        "epoch_sweeps": int(config["epoch_sweeps"]),
        "stall_window_sweeps": int(config["stall_window_sweeps"]),
        "planned_cold_sweeps": int(config["planned_cold_sweeps"]),
        "sweep_cap": int(sweep_cap),
        "sweeps": int(final["sweeps"]),
        "tc": float(final["tc_final"]),
        "trigger": None,
    }
    if switch is not None:
        row["trigger"] = {
            "reason": switch["reason"],
            "time_ms": float(switch["t_ms"]),
            "cold_sweeps": int(switch["cold_sweeps"]),
            "waist_start": float(switch["waist_start"]) if "waist_start" in switch else None,
            "waist_end": float(switch["waist_end"]) if "waist_end" in switch else None,
            "improvement": float(switch["improvement"]) if "improvement" in switch else None,
        }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))


def report(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    expected = {
        (instance, arm)
        for instance in FAMILIES
        for arm in ("event-w", "event-2w", "fixed15", "fixed25", "fixed40")
    }
    observed = [(row.get("instance"), row.get("arm")) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        duplicates = sorted(pair for pair in set(observed) if observed.count(pair) > 1)
        die(
            "invalid evidence matrix: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    incomplete = [row for row in rows if row["sweeps"] != row["sweep_cap"]]
    if incomplete:
        names = ", ".join(f'{row["instance"]}/{row["arm"]}' for row in incomplete)
        die(f"runs hit the wall deadline before matched sweep cap: {names}")

    print("trigger-time table by family")
    print(
        "family       instance          window  trigger_ms  cold_sweeps  "
        "waist_gain       tc"
    )
    for instance in FAMILIES:
        group = [row for row in rows if row["instance"] == instance]
        for arm in ("event-w", "event-2w"):
            row = next((item for item in group if item["arm"] == arm), None)
            if row is None:
                die(f"missing event arm: {instance}/{arm}")
            trigger = row["trigger"]
            if trigger is None:
                print(
                    f'{row["family"]:<12} {instance:<17} {row["parameter"]:>5.0f}W '
                    f'{"never":>11} {"-":>12} {"-":>11} {row["tc"]:>8.4f}'
                )
            else:
                print(
                    f'{row["family"]:<12} {instance:<17} {row["parameter"]:>5.0f}W '
                    f'{trigger["time_ms"]:>11.3f} {trigger["cold_sweeps"]:>12} '
                    f'{trigger["improvement"]:>11.4f} {row["tc"]:>8.4f}'
                )

    print("\nfinal tc: event W versus best fixed-fraction arm")
    print("instance          event_W  best_fixed  fraction   delta")
    for instance in FAMILIES:
        group = [row for row in rows if row["instance"] == instance]
        event = next(row for row in group if row["arm"] == "event-w")
        fixed = min((row for row in group if row["mode"] == "fixed"), key=lambda row: row["tc"])
        print(
            f'{instance:<17} {event["tc"]:>8.4f} {fixed["tc"]:>11.4f} '
            f'{fixed["parameter"]:>8.0%} {event["tc"] - fixed["tc"]:>8.4f}'
        )

    print("\nstall-window sensitivity")
    print("instance             W_ms      2W_ms    delta_ms      W_tc     2W_tc  delta_tc")
    for instance in FAMILIES:
        group = [row for row in rows if row["instance"] == instance]
        event_w = next(row for row in group if row["arm"] == "event-w")
        event_2w = next(row for row in group if row["arm"] == "event-2w")
        w_time = event_w["trigger"]["time_ms"] if event_w["trigger"] else None
        two_w_time = event_2w["trigger"]["time_ms"] if event_2w["trigger"] else None
        w_text = "never" if w_time is None else f"{w_time:.3f}"
        two_w_text = "never" if two_w_time is None else f"{two_w_time:.3f}"
        delta_text = (
            "-" if w_time is None or two_w_time is None else f"{two_w_time - w_time:.3f}"
        )
        print(
            f'{instance:<17} {w_text:>9} {two_w_text:>10} {delta_text:>11} '
            f'{event_w["tc"]:>9.4f} {event_2w["tc"]:>9.4f} '
            f'{event_2w["tc"] - event_w["tc"]:>9.4f}'
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
