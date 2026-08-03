# Paper master gate design

## Goal

Paper results must come from the benchmark-relevant content on the current
`origin/master`. Historical pins, research branches, attempt worktrees, patched
trees, and stale local checkouts cannot supply manuscript data. Each run must
record enough deterministic provenance to establish exactly which executable
and inputs produced its artifact.

## Master-run wrapper

Add `benchmarks/paper/run_master.py` as the canonical paper-data entry point.
It mirrors the runner's manifest, set, output, and repository-root arguments.
Before building, it fetches `origin/master`, requires `HEAD` to equal the
fetched remote branch, and rejects any tracked or untracked working-tree
change. Ignored build outputs such as `target/` do not fail the check.

The wrapper parses the selected manifest set, resolves its declared instance
paths, and computes SHA-256 identities for the manifest and selected input
bundle. It builds `paper_bench` from the verified checkout, hashes the binary,
runs it, hashes the completed JSON artifact, and atomically writes a sibling
`<output>.provenance.json` file. Provenance contains a schema version, producing
revision, set name, binary hash, manifest hash, selected-input hash, and output
hash. It contains no hostname, timestamp, or absolute path.

The `paper-bench` Make target uses this wrapper. PR CI retains the existing raw
small-set reproduction because a PR commit cannot equal `origin/master`; CI
instead unit-tests the gate and provenance logic. A failed revision, cleanliness,
build, benchmark, or hashing check exits nonzero and leaves no provenance file.

## Figure 2(b)

Remove the archived `47.377` campaign median from `plot_figure2b.py`, including
its range contribution, horizontal line, and legend item. Keep the current-main
pre-surgery record, retained-incumbent curve, raw fine-tuning endpoints, and
accepted-rebuild markers. Regenerate the committed SVG and run the semantic and
exact-artifact checks.

## Existing Huawei measurements

After merge, rebuild on Huawei and compare the new binary, manifest, and selected
input hashes with those recorded for commit `41833440`. Existing numerical
artifacts remain valid only when benchmark-relevant identities are unchanged;
otherwise the affected sets are rerun from the new `origin/master`.
