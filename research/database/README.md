# Domain database — beat-existing-optimizers

Structured data the autoresearch run loop and validator draw on. All records
are machine-generated; provenance noted per file.

## graphs.json

One record per benchmark instance in `benchmarks/graphs/` (checked into this
repo; same graphs used by the Julia/Python/Rust cross-implementation
benchmarks).

| field | meaning |
|---|---|
| `name` | graph id, matches the `name` field inside the source JSON |
| `family` | `chain` \| `grid` \| `reg3` (random 3-regular) \| `named` (Petersen) |
| `schema` | `einsum` (`ixs`/`iy`/`sizes` einsum spec) or `edge_list` (raw graph; `reg3_220.json` only — not consumed by the benchmark example) |
| `tensors` | number of input tensors (vertices) |
| `indices` | number of distinct indices (edges); all bond dimensions are 2 |
| `file` | repo-relative path to the source JSON |

Provenance: generated from `benchmarks/graphs/*.json` at the commit recorded
in `baselines.json`.

## baselines.json

Single-threaded baseline measurements the primary metric Δtc@budget compares
against (see `research/topics.md`).

- `protocol` — measurement conditions: `RAYON_NUM_THREADS=1`, release build,
  machine, date, omeco commit, TreeSA params (benchmark example defaults,
  ntrials=1).
- `results.<graph>.{greedy,treesa}` — `tc`, `sc`, `rwc` (log2 scale) and
  `avg_ms` (mean optimizer wall-clock per run). `avg_ms` for `treesa` is the
  budget `T(g)` referenced by the metric definition.

Provenance: produced by `RAYON_NUM_THREADS=1 cargo run --release --example
benchmark -p omeco` on 2026-07-22 (commit in file); raw output also written
by that command to `benchmarks/results/rust_results.json`.

## Regenerating

Re-run the benchmark example (command above), then rebuild both JSONs; keep
the `protocol` block accurate (machine, date, commit). Baselines must be
re-measured on the machine the validator scores on — budgets are wall-clock.
