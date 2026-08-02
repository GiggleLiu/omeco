#!/usr/bin/env python3
"""Verify the semantic contract behind the Figure 2(b) reproduction set."""

import json
import sys
from pathlib import Path


PRE_SURGERY_RECORD = 47.824


def fail(message: str) -> None:
    raise SystemExit(f"verify_figure2b.py: FAIL: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: verify_figure2b.py ARTIFACT.json")
    artifact = json.loads(Path(sys.argv[1]).read_text())
    if artifact.get("format") != 2 or artifact.get("set") != "figure2b":
        fail("expected format 2, set figure2b")

    rows = {row.get("arm"): row for row in artifact.get("results", [])}
    if set(rows) != {"treesa", "treesa_rounds"}:
        fail(f"expected treesa and treesa_rounds rows, got {sorted(rows)}")
    baseline = rows["treesa"]
    surgery = rows["treesa_rounds"]
    if surgery["tc"] >= PRE_SURGERY_RECORD:
        fail(
            f"surgery tc={surgery['tc']} does not cross the Figure 2(b) "
            f"record {PRE_SURGERY_RECORD}"
        )
    if surgery["tc"] >= baseline["tc"]:
        fail("surgery does not improve the plain TreeSA baseline")

    curve = surgery.get("curve", [])
    if len(curve) != 32:
        fail(f"expected 32 fixed-work rounds, got {len(curve)}")
    accepted_drops = 0
    worse_endpoints = 0
    previous = None
    for point in curve:
        if point["tc_retained"] > point["tc_before"]:
            fail(f"retained incumbent rises in round {point['round']}")
        if previous is not None and point["tc_before"] != previous:
            fail(f"round {point['round']} does not continue from the retained incumbent")
        if point["surgery_accepted"] and point["tc_after_surgery"] < point["tc_before"]:
            accepted_drops += 1
        if point["tc_after_anneal"] > point["tc_before"]:
            worse_endpoints += 1
        previous = point["tc_retained"]
    if accepted_drops == 0:
        fail("no accepted waist rebuild initiates a tc drop")
    if worse_endpoints == 0:
        fail("fine-tuning endpoint never exposes an uphill excursion")
    if previous != surgery["tc"]:
        fail(f"final retained tc={previous} differs from row tc={surgery['tc']}")

    print(
        f"OK: tc={surgery['tc']} < {PRE_SURGERY_RECORD}; "
        f"accepted rebuild drops={accepted_drops}; uphill endpoints={worse_endpoints}; "
        f"rounds={len(curve)}"
    )


if __name__ == "__main__":
    main()
