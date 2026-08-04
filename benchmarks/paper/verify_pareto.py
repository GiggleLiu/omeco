#!/usr/bin/env python3
"""Verify the fixed-work, measured-time inputs for the paper Pareto plot."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROUNDS = (0, 1, 2, 4, 8, 24)
FAMILIES = ("surfacecode_d21", "sycamore_53_20_0")


def fail(message: str) -> None:
    raise SystemExit(f"pareto verification FAILED: {message}")


def default_score(row: dict) -> float:
    """TreeSA default score reconstructed from emitted log2 complexities."""
    return math.exp2(row["tc"]) + max(0.0, math.exp2(row["sc"]) - math.exp2(20.0))


def load_provenance(artifact_path: Path, set_name: str, artifact_bytes: bytes) -> dict:
    """Read the run_master.py sidecar next to an artifact and bind it to the bytes."""
    sidecar = Path(f"{artifact_path}.provenance.json")
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, ValueError) as exc:
        fail(f"cannot read provenance sidecar {sidecar}: {exc}")
    if record.get("set") != set_name:
        fail(f"{sidecar}: provenance records set {record.get('set')!r}, not {set_name!r}")
    output = record.get("output")
    recorded_hash = output.get("sha256") if isinstance(output, dict) else None
    if recorded_hash != hashlib.sha256(artifact_bytes).hexdigest():
        fail(f"{sidecar}: recorded output hash does not match {artifact_path}")
    manifest = record.get("manifest")
    fields = {
        "revision": record.get("revision"),
        "manifest_sha256": manifest.get("sha256") if isinstance(manifest, dict) else None,
        "binary_sha256": record.get("binary_sha256"),
    }
    for key, value in fields.items():
        if not isinstance(value, str) or not value:
            fail(f"{sidecar}: missing or invalid {key}")
    return fields


def main() -> None:
    if len(sys.argv) != len(ROUNDS) + 1:
        fail("usage: verify_pareto.py pareto_r0.json pareto_r1.json ... pareto_r24.json")

    artifacts = {}
    provenances = {}
    for argument in sys.argv[1:]:
        path = Path(argument)
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            fail(f"cannot read {path}: {exc}")
        set_name = data.get("set")
        if data.get("format") != 2 or set_name not in {f"pareto_r{r}" for r in ROUNDS}:
            fail(f"{path}: wrong artifact format or set")
        if set_name in artifacts:
            fail(f"duplicate artifact for {set_name}")
        artifacts[set_name] = data
        provenances[set_name] = load_provenance(path, set_name, raw)

    expected_sets = {f"pareto_r{r}" for r in ROUNDS}
    if set(artifacts) != expected_sets:
        fail(f"expected sets {sorted(expected_sets)}, found {sorted(artifacts)}")

    for key, label in (
        ("revision", "revision"),
        ("manifest_sha256", "manifest hash"),
        ("binary_sha256", "runner binary hash"),
    ):
        values = {provenance[key] for provenance in provenances.values()}
        if len(values) != 1:
            fail(f"mixed {label} across the sweep: {sorted(values)}")
    revision = provenances[f"pareto_r{ROUNDS[0]}"]["revision"]

    trajectories = {(family, repeat): [] for family in FAMILIES for repeat in range(3)}
    for rounds in ROUNDS:
        data = artifacts[f"pareto_r{rounds}"]
        rows = data.get("results")
        if not isinstance(rows, list) or len(rows) != 6:
            fail(f"pareto_r{rounds}: expected six results")
        expected_arm = "treesa" if rounds == 0 else "treesa_rounds"
        seen = set()
        for row in rows:
            name = row.get("instance", "")
            match = next(
                (
                    (family, repeat)
                    for family in FAMILIES
                    for repeat in range(3)
                    if name == f"{family}_r{repeat}"
                ),
                None,
            )
            if match is None or match in seen:
                fail(f"pareto_r{rounds}: unexpected or duplicate instance {name!r}")
            seen.add(match)
            if row.get("arm") != expected_arm:
                fail(f"pareto_r{rounds} {name}: expected arm {expected_arm}")
            values = (row.get("tc"), row.get("sc"), row.get("rwc"), row.get("elapsed_s"))
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                fail(f"pareto_r{rounds} {name}: missing or non-finite metric")
            if row["elapsed_s"] <= 0:
                fail(f"pareto_r{rounds} {name}: non-positive elapsed time")
            curve = row.get("curve")
            if rounds == 0:
                if curve is not None:
                    fail(f"pareto_r0 {name}: baseline unexpectedly has a round trace")
            elif not isinstance(curve, list) or len(curve) != rounds:
                fail(f"pareto_r{rounds} {name}: expected {rounds} curve points")
            trajectories[match].append((rounds, default_score(row)))

    for (family, repeat), trajectory in trajectories.items():
        trajectory.sort()
        for (before_rounds, before), (after_rounds, after) in zip(trajectory, trajectory[1:]):
            # tc/sc are rounded to 1e-6 before score reconstruction, so allow
            # the corresponding relative quantization band.
            if after > before * (1.0 + 2e-6):
                fail(
                    f"{family}_r{repeat}: configured score rose from r{before_rounds} "
                    f"to r{after_rounds}"
                )

    summaries = []
    for rounds in ROUNDS:
        rows = artifacts[f"pareto_r{rounds}"]["results"]
        median_s = statistics.median(row["elapsed_s"] for row in rows)
        summaries.append(f"r{rounds}={median_s:.1f}s")
    print(
        f"verified fixed-work Pareto sweep: 36 runs at revision {revision[:12]}; "
        "median times " + ", ".join(summaries)
    )


if __name__ == "__main__":
    main()
