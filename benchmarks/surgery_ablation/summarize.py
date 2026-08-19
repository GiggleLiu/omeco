#!/usr/bin/env python3
"""Summarize surgery_ablation JSONL artifacts as Markdown."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("key")
                if not isinstance(key, str):
                    raise ValueError(f"{path}:{line_number}: missing string key")
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
    return rows


def markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def quality_table(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["instance"], row["arm"])].append(float(row["tc"]))
    body = []
    for (instance, arm), values in sorted(grouped.items()):
        body.append(
            [
                instance,
                arm,
                str(len(values)),
                f"{min(values):.6f}",
                f"{statistics.median(values):.6f}",
            ]
        )
    return markdown_table(["instance", "arm", "n", "min tc", "median tc"], body)


def surgery_wtl(rows: list[dict[str, Any]]) -> list[str]:
    by_key = {(row["instance"], row["label"], row["arm"]): row for row in rows}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        arm = row["arm"]
        if not arm.startswith("surg_"):
            continue
        suffix = arm.rsplit("_r", 1)
        if len(suffix) != 2:
            continue
        cold = by_key.get((row["instance"], row["label"], f"cold_only_r{suffix[1]}"))
        if cold is None:
            continue
        delta = float(row["tc"]) - float(cold["tc"])
        if delta < -1e-9:
            counts[arm][0] += 1
        elif delta > 1e-9:
            counts[arm][2] += 1
        else:
            counts[arm][1] += 1
    body = [[arm, *(str(value) for value in values)] for arm, values in sorted(counts.items())]
    return markdown_table(["surgery arm vs cold-only", "W", "T", "L"], body)


def work_table(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"].startswith("treesa_x"):
            grouped[row["arm"]].append(row)
    body = []
    for arm, values in sorted(grouped.items()):
        body.append(
            [
                arm,
                str(len(values)),
                f"{statistics.median(float(row['tc']) for row in values):.6f}",
                f"{statistics.median(int(row['total_node_visits']) for row in values):.0f}",
                f"{statistics.median(float(row['wall_seconds']) for row in values):.3f}",
            ]
        )
    return markdown_table(
        ["work-matched arm", "n", "median tc", "median node visits", "median wall s"],
        body,
    )


def accepted_table(rows: list[dict[str, Any]]) -> list[str]:
    totals: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[row["arm"]] += int(row.get("accepted_rebuilds", 0))
        calls[row["arm"]] += 1
    body = [[arm, str(calls[arm]), str(total)] for arm, total in sorted(totals.items())]
    return markdown_table(["arm", "jobs", "accepted rebuilds"], body)


def render(rows: list[dict[str, Any]]) -> str:
    sections = [
        "# Surgery ablation summary",
        "",
        f"Rows: {len(rows)}",
        "",
        "## Per-instance quality",
        "",
        *quality_table(rows),
        "",
        "## Surgery versus matched cold-only",
        "",
        *surgery_wtl(rows),
        "",
        "## Work-matched TreeSA",
        "",
        *work_table(rows),
        "",
        "## Accepted rebuilds",
        "",
        *accepted_table(rows),
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render(load_rows(args.jsonl))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
