#!/usr/bin/env python3
"""Hard caps, snapshot curves, and stdlib-only evidence for attempt 063."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path


CHECKPOINTS_S = (1.0, 3.0, 10.0, 30.0)
POLL_INTERVAL_S = 0.01


def die(message: str) -> "NoReturn":
    raise SystemExit(message)


def kill(process: subprocess.Popen, process_group: bool) -> None:
    """Immediately kill a command and, when requested, its process group."""
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
        die("usage: summarize_attempt.py relabel <source.json> <out.json> <seed>")
    source, destination, seed_text = argv[1:]
    graph = json.loads(Path(source).read_text())
    labels = sorted(
        {int(label) for indices in graph["ixs"] for label in indices}
        | {int(label) for label in graph["iy"]}
        | {int(label) for label in graph["sizes"]}
    )
    shuffled = labels.copy()
    random.Random(int(seed_text)).shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == labels:
        shuffled = shuffled[1:] + shuffled[:1]
    mapping = dict(zip(labels, shuffled))
    graph["ixs"] = [[mapping[int(label)] for label in indices] for indices in graph["ixs"]]
    graph["iy"] = [mapping[int(label)] for label in graph["iy"]]
    graph["sizes"] = {str(mapping[int(label)]): size for label, size in graph["sizes"].items()}
    Path(destination).write_text(json.dumps(graph, sort_keys=True, separators=(",", ":")))


def log2sumexp2(values: list[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    return maximum + math.log2(sum(math.exp2(value - maximum) for value in values))


def snapshot_tc(snapshot: dict, sizes: dict[str, int]) -> float:
    log_sizes = {int(label): math.log2(size) for label, size in sizes.items()}

    def walk(tree: dict) -> list[float]:
        if tree["isleaf"]:
            return []
        costs = []
        for child in tree["args"]:
            costs.extend(walk(child))
        labels = {int(label) for indices in tree["eins"]["ixs"] for label in indices}
        labels.update(int(label) for label in tree["eins"]["iy"])
        costs.append(sum(log_sizes.get(label, 0.0) for label in labels))
        return costs

    return log2sumexp2(walk(snapshot["tree"]))


def observe_run(argv: list[str]) -> None:
    if len(argv) < 11 or argv[9] != "--":
        die(
            "usage: summarize_attempt.py run <cap_s> <instance> <relabeling> <arm> "
            "<graph> <stdout> <stderr> <tree> -- <command...>"
        )
    cap_s = float(argv[1])
    instance, relabeling_id, arm = argv[2:5]
    graph_path, stdout_path, stderr_path, tree_path = argv[5:9]
    command = argv[10:]
    graph = json.loads(Path(graph_path).read_text())
    output = Path(tree_path)
    output.unlink(missing_ok=True)
    observations: list[dict] = []
    last_content: bytes | None = None
    nested = os.environ.get("ATT_BENCH_GUARDED") == "1"

    def sample(elapsed: float) -> None:
        nonlocal last_content
        try:
            content = output.read_bytes()
            if content != last_content:
                snapshot = json.loads(content)
                observations.append(
                    {
                        "time_s": elapsed,
                        "tc": snapshot_tc(snapshot, graph["sizes"]),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
                last_content = content
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        start = time.monotonic()
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, start_new_session=not nested
        )
        while True:
            elapsed = time.monotonic() - start
            if elapsed > cap_s:
                kill(process, process_group=not nested)
                die(f"command exceeded {cap_s:g}s cap: {' '.join(command)}")
            sample(elapsed)
            returncode = process.poll()
            if returncode is not None:
                finished_s = time.monotonic() - start
                # The attempt force-flushes immediately before exit; drain once
                # after poll() so a rename between the pre-poll read and exit
                # cannot be omitted from tc@30.
                sample(finished_s)
                break
            time.sleep(POLL_INTERVAL_S)

    if returncode:
        die(f"command failed ({returncode}); stderr: {stderr_path}")
    if not observations:
        die(f"no valid snapshot observed: {tree_path}")

    curve = {}
    for checkpoint in CHECKPOINTS_S:
        eligible = [point for point in observations if point["time_s"] <= checkpoint]
        curve[f"{checkpoint:g}"] = eligible[-1]["tc"] if eligible else None
    row = {
        "attempt": 63,
        "instance": instance,
        "relabeling": relabeling_id,
        "arm": arm,
        "first_snapshot_s": observations[0]["time_s"],
        "finished_s": finished_s,
        "tc_at_s": curve,
        "snapshots": observations,
    }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))


def format_tc(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:7.3f}"


def report(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print("instance       relabel arm       first_s    tc@1    tc@3   tc@10   tc@30 snapshots")
    for row in rows:
        curve = row["tc_at_s"]
        print(
            f'{row["instance"]:<14} {row["relabeling"]:<7} {row["arm"]:<8} '
            f'{row["first_snapshot_s"]:>7.3f} {format_tc(curve["1"])} '
            f'{format_tc(curve["3"])} {format_tc(curve["10"])} '
            f'{format_tc(curve["30"])} {len(row["snapshots"]):>9}'
        )

    indexed = {(row["instance"], row["relabeling"], row["arm"]): row for row in rows}
    expected = {
        (instance, relabeling_id, arm)
        for instance in ("sycamore_m20", "reg3_250")
        for relabeling_id in ("r0", "r1")
        for arm in ("parent", "robust")
    }
    missing = sorted(expected - indexed.keys())
    if missing:
        die(f"missing evidence rows: {missing}")

    print("\ncomparison     relabel first<=0.5  tc@1 tc@3 tc@10 tc@30 overall")
    all_pass = True
    for instance in ("sycamore_m20", "reg3_250"):
        for relabeling_id in ("r0", "r1"):
            parent = indexed[(instance, relabeling_id, "parent")]
            robust = indexed[(instance, relabeling_id, "robust")]
            first_ok = robust["first_snapshot_s"] <= 0.5
            curve_ok = []
            for checkpoint in CHECKPOINTS_S:
                key = f"{checkpoint:g}"
                candidate = robust["tc_at_s"][key]
                control = parent["tc_at_s"][key]
                curve_ok.append(
                    candidate is not None and control is not None and candidate <= control + 1e-9
                )
            row_ok = first_ok and all(curve_ok)
            all_pass &= row_ok
            marks = ["PASS" if value else "FAIL" for value in curve_ok]
            print(
                f"{instance:<14} {relabeling_id:<7} {'PASS' if first_ok else 'FAIL':<11} "
                f"{marks[0]:<4} {marks[1]:<4} {marks[2]:<5} {marks[3]:<5} "
                f"{'PASS' if row_ok else 'FAIL'}"
            )
    print(f"expected evidence: {'PASS' if all_pass else 'FAIL'}")


def main() -> None:
    if len(sys.argv) < 2:
        die("expected subcommand: exec, supervise, relabel, run, or report")
    command = sys.argv[1]
    argv = sys.argv[1:]
    if command == "exec":
        run_capped(argv)
    elif command == "supervise":
        supervise(argv)
    elif command == "relabel":
        relabel(argv)
    elif command == "run":
        observe_run(argv)
    elif command == "report" and len(argv) == 2:
        report(argv[1])
    else:
        die(f"unknown or malformed subcommand: {command}")


if __name__ == "__main__":
    main()
