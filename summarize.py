#!/usr/bin/env python3
"""Record and summarize attempt-059 matched-sweep diagnostic runs."""

import argparse
import json
import pathlib
import sys


def record(args: argparse.Namespace) -> None:
    diagnostic = None
    for line in pathlib.Path(args.stderr_log).read_text().splitlines():
        if line.startswith("ATT_DIAG "):
            diagnostic = json.loads(line.removeprefix("ATT_DIAG "))
    if diagnostic is None:
        raise SystemExit(f"no ATT_DIAG record in {args.stderr_log}")

    tree = json.loads(pathlib.Path(args.result).read_text())
    required = {"inputs", "label-type", "output", "tree"}
    if not isinstance(tree, dict) or not required.issubset(tree):
        raise SystemExit(f"result does not match writejson nested-einsum schema: {args.result}")
    if diagnostic["mode"] != args.arm:
        raise SystemExit(
            f"arm mismatch: requested {args.arm}, binary reported {diagnostic['mode']}"
        )

    diagnostic.update(
        {
            "instance": args.instance,
            "sweep_budget": args.sweep_budget,
            "completed_budget": diagnostic["sweeps"] == args.sweep_budget,
        }
    )
    with pathlib.Path(args.out_jsonl).open("a", encoding="utf-8") as stream:
        json.dump(diagnostic, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def summarize(path: pathlib.Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {(row["instance"], row["sweep_budget"], row["mode"]): row for row in rows}
    print(
        "instance\tsweeps\tcontinuous_tc\tparent_tc\tdelta\t"
        "continuous_yield_rate\tparent_yield_rate\tcomplete"
    )
    missing = False
    pairs = sorted({(row["instance"], row["sweep_budget"]) for row in rows})
    for instance, sweeps in pairs:
        continuous = keyed.get((instance, sweeps, "continuous"))
        parent = keyed.get((instance, sweeps, "parent"))
        if continuous is None or parent is None:
            print(f"{instance}\t{sweeps}\tMISSING MATCHED ARM", file=sys.stderr)
            missing = True
            continue
        continuous_yield = sum(band["improving_accepts"] for band in continuous["bands"])
        parent_yield = sum(band["improving_accepts"] for band in parent["bands"])
        continuous_attempts = sum(band["attempts"] for band in continuous["bands"])
        parent_attempts = sum(band["attempts"] for band in parent["bands"])
        continuous_rate = continuous_yield / max(continuous_attempts, 1)
        parent_rate = parent_yield / max(parent_attempts, 1)
        delta = continuous["best_tc"] - parent["best_tc"]
        complete = continuous["completed_budget"] and parent["completed_budget"]
        print(
            f"{instance}\t{sweeps}\t{continuous['best_tc']:.6f}\t"
            f"{parent['best_tc']:.6f}\t{delta:+.6f}\t{continuous_rate:.6g}\t"
            f"{parent_rate:.6g}\t{str(complete).lower()}"
        )
    if missing:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_jsonl", nargs="?", type=pathlib.Path)
    parser.add_argument(
        "--record",
        nargs=6,
        metavar=("INSTANCE", "ARM", "SWEEPS", "STDERR", "RESULT", "OUT_JSONL"),
    )
    parsed = parser.parse_args()
    if parsed.record:
        (
            parsed.instance,
            parsed.arm,
            sweeps,
            parsed.stderr_log,
            parsed.result,
            parsed.out_jsonl,
        ) = parsed.record
        parsed.sweep_budget = int(sweeps)
    elif parsed.summary_jsonl is None:
        parser.error("provide <out.jsonl> or --record arguments")
    return parsed


def main() -> None:
    args = parse_args()
    if args.record:
        record(args)
    else:
        summarize(args.summary_jsonl)


if __name__ == "__main__":
    main()
