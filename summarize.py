#!/usr/bin/env python3
"""Extract ATT_DIAG records and print a compact matched-budget report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def extract(args: argparse.Namespace) -> None:
    tree_bytes = Path(args.tree).read_bytes()
    tree = json.loads(tree_bytes)
    if not isinstance(tree, dict) or not ({"inputs", "output", "tree"} <= tree.keys()):
        raise SystemExit(f"invalid nested einsum tree: {args.tree}")
    leaves: list[int] = []

    def walk(node: object) -> None:
        if not isinstance(node, dict) or not isinstance(node.get("isleaf"), bool):
            raise SystemExit(f"malformed tree node: {args.tree}")
        if node["isleaf"]:
            index = node.get("tensorindex")
            if not isinstance(index, int):
                raise SystemExit(f"malformed leaf index: {args.tree}")
            leaves.append(index)
            return
        children = node.get("args")
        if not isinstance(children, list) or len(children) != 2:
            raise SystemExit(f"non-binary internal node: {args.tree}")
        for child in children:
            walk(child)

    walk(tree["tree"])
    if sorted(leaves) != list(range(len(tree["inputs"]))):
        raise SystemExit(f"leaves are not a permutation of inputs: {args.tree}")

    diag = None
    for line in Path(args.stderr).read_text().splitlines():
        if line.startswith("ATT_DIAG "):
            diag = json.loads(line.removeprefix("ATT_DIAG "))
    if diag is None:
        raise SystemExit(f"missing ATT_DIAG record: {args.stderr}")
    if diag.get("mode") != args.mode:
        raise SystemExit(f"mode mismatch: expected {args.mode}, got {diag.get('mode')}")

    record = {
        "instance": args.instance,
        "budget_ms": args.budget_ms,
        "tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
        **diag,
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))


def report(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print(
        "instance             budget mode    tc       attempts/s accepts/s active-frac "
        "cold(full/active)"
    )
    for row in rows:
        print(
            f"{row['instance']:<20} {row['budget_ms'] / 1000:>5.0f}s "
            f"{row['mode']:<7} {row['tc_final']:>8.4f} "
            f"{row['attempts_per_sec']:>10.1f} {row['accepts_per_sec']:>9.1f} "
            f"{row['active_fraction_mean']:>11.3f} "
            f"{row['cold_full_sweeps']}/{row['cold_active_sweeps']}"
        )
    parents = {
        (row["instance"], row["budget_ms"]): row for row in rows if row["mode"] == "parent"
    }
    print("\nmatched active/parent deltas (negative tc delta is better):")
    print("instance             budget tc-delta sweep-x attempt-x accept-x active-frac")
    for active in (row for row in rows if row["mode"] == "active"):
        parent = parents.get((active["instance"], active["budget_ms"]))
        if parent is None:
            continue
        print(
            f"{active['instance']:<20} {active['budget_ms'] / 1000:>5.0f}s "
            f"{active['tc_final'] - parent['tc_final']:>8.4f} "
            f"{active['sweeps_per_sec'] / parent['sweeps_per_sec']:>7.3f} "
            f"{active['attempts_per_sec'] / parent['attempts_per_sec']:>9.3f} "
            f"{active['accepts_per_sec'] / parent['accepts_per_sec']:>8.3f} "
            f"{active['active_fraction_mean']:>11.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report")
    parser.add_argument("--instance")
    parser.add_argument("--mode", choices=("active", "parent"))
    parser.add_argument("--budget-ms", type=int)
    parser.add_argument("--stderr")
    parser.add_argument("--tree")
    args = parser.parse_args()
    if not args.report and any(
        value is None
        for value in (args.instance, args.mode, args.budget_ms, args.stderr, args.tree)
    ):
        parser.error("extraction requires --instance --mode --budget-ms --stderr --tree")
    return args


def main() -> None:
    args = parse_args()
    if args.report:
        report(args.report)
    else:
        extract(args)


if __name__ == "__main__":
    main()
