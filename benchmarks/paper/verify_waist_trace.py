#!/usr/bin/env python3
"""Verify the dense fixed-work waist-cut trace used by Figure 2(a)."""

import json
import math
from pathlib import Path
import sys


ROUND6_QUANTUM = 1e-6


def fail(message: str) -> None:
    raise SystemExit(f"waist-trace verification FAILED: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: verify_waist_trace.py <waist_trace.json>")
    path = Path(sys.argv[1])
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")

    if data.get("format") != 2 or data.get("set") != "waist_trace":
        fail("wrong artifact format or set")
    rows = data.get("results")
    if not isinstance(rows, list) or len(rows) != 10:
        fail("expected ten relabeled trajectories")

    families = {"surfacecode_d21": 0, "ksg": 0}
    comparisons = []
    for row in rows:
        name = row.get("instance", "")
        family = next((key for key in families if name.startswith(f"{key}_r")), None)
        if family is None:
            fail(f"unexpected instance {name!r}")
        families[family] += 1
        if row.get("arm") != "treesa_rounds":
            fail(f"{name}: wrong arm")
        curve = row.get("curve")
        if not isinstance(curve, list) or len(curve) != 128:
            fail(f"{name}: expected 128 rounds")
        for expected_round, point in enumerate(curve):
            if point.get("round") != expected_round:
                fail(f"{name}: broken round sequence")
            waist = point.get("waist")
            if not isinstance(waist, dict):
                fail(f"{name} round {expected_round}: missing waist comparison")
            incumbent = waist.get("incumbent_cut_cost")
            candidate = waist.get("best_alt_cut_cost")
            waist_node = waist.get("waist_node_cost")
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in (incumbent, candidate, waist_node)):
                fail(f"{name} round {expected_round}: non-finite cut cost")
            if candidate > incumbent + 1e-9:
                fail(f"{name} round {expected_round}: FM lost the incumbent seed")
            attempted = waist.get("rebuild_attempted")
            if not isinstance(attempted, bool):
                fail(f"{name} round {expected_round}: missing rebuild gate flag")
            if attempted and candidate > waist_node + 1e-9:
                fail(f"{name} round {expected_round}: rebuild gate mismatch")
            if not attempted and candidate < waist_node - ROUND6_QUANTUM - 1e-9:
                fail(f"{name} round {expected_round}: rebuild gate mismatch")
            if waist.get("rebuild_accepted") is not point.get("surgery_accepted"):
                fail(f"{name} round {expected_round}: acceptance mismatch")
            if point["tc_retained"] > point["tc_before"] + 1e-9:
                fail(f"{name} round {expected_round}: incumbent ratchet rose")
            comparisons.append((family, incumbent, candidate))

    if families != {"surfacecode_d21": 5, "ksg": 5}:
        fail(f"wrong relabeling counts: {families}")
    if len(comparisons) != 1280:
        fail(f"expected 1280 comparisons, found {len(comparisons)}")
    improved = sum(candidate < incumbent - 1e-9 for _, incumbent, candidate in comparisons)
    print(f"verified waist trace: 1280 current-master comparisons, {improved} cheaper FM cuts")


if __name__ == "__main__":
    main()
