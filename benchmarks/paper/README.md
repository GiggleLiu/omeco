# Paper benchmark artifact

Every number in the paper's released-artifact table is produced here, by the
current `master` library, on the reader's own machine. There is no reference
host and no recorded oracle: the artifact is a *program plus a manifest*, and
reproducing that table means running it. The `figure2b` set separately
reproduces the surface-code mechanism panel with a deterministic 32-round work
budget: it must cross the panel's pre-surgery record and show an accepted
rebuild causing a drop. It must also expose at least one uphill fine-tuning
endpoint separately from the monotone retained incumbent. `waist_trace` is the
dense mechanism set: five deterministic relabelings each
of surfacecode_d21 and ksg, with 128 fixed-work rounds per relabeling and exact
like-for-like waist/FM cut weights for every completed surgery call. The outputs under
`expected/` are committed only so that they can be *re-derived* — CI
regenerates the `ci` one on every push and fails on a single changed field, so
a committed number can never quietly drift away from what the code does.

## Master revision hard gate

**The paper always tracks the current `master` revision.** A paper run must be
made from a clean checkout of the commit currently at `origin/master`, and its
results remain valid only while the benchmark-relevant content of
`origin/master` — benchmark code, manifests, and instance data — still matches
the run's recorded binary and input manifest. Results from a historical pin,
research branch, attempt worktree, patched source tree, or stale local `master`
are not paper data.

`run_master.py` enforces this gate. Its checks are equivalent to:

```bash
git fetch origin master
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)" \
    || { echo "HEAD is not at origin/master" >&2; exit 1; }
test -z "$(git status --porcelain)" \
    || { echo "working tree is not clean" >&2; exit 1; }
```

The wrapper builds the runner and records the revision plus binary, manifest,
selected-input, and output hashes in `<output>.provenance.json`. If benchmark
code, manifests, or instance data on `origin/master` change, the paper artifacts
are stale and must be recomputed before manuscript numbers or figures are
updated. A later documentation-only or artifact-only commit does not invalidate
an unchanged recorded binary and input manifest. Revision mismatch at run time
is a hard failure; there is no fallback to archived campaign data.

This directory is self-contained — manifest, instances, checker:

| Path | Role |
|---|---|
| `manifest.json` | Which instances, which optimizer arms, which overrides. The only place a benchmark parameter may be set. |
| `instances/` | The thirteen paper instances, as JSON tensor networks. |
| `expected/ci.json` | The `ci` set's output, re-derived and compared by `make paper-bench-check` (and by CI). `expected/full.json` is what `make paper-bench` writes. |
| `expected/figure2b.json` | The 32-round surface-code reproduction, checked numerically and by the Figure 2(b) semantic contract. |
| `expected/figure2b.svg` | Static fixed-work rendering of that JSON: retained incumbent, raw fine-tuning endpoints, and accepted rebuilds. |
| `run_master.py` | Enforces current clean `origin/master`, builds the runner, and emits deterministic provenance. |
| `test_run_master.py` | Unit tests for revision, cleanliness, tracking, hashing, and atomic-write behavior. |
| `check.py` | Compares two artifacts and exits nonzero on any disagreement. Standard library only, Python 3.8+. |
| `verify_figure2b.py` | Verifies the record crossing, incumbent ratchet, and accepted-rebuild mechanism. |
| `plot_figure2b.py` | Renders the checked JSON as a dependency-free vector figure. |
| `README.md` | This file. |

The runner itself is `omeco/examples/paper_bench.rs`, an example binary in the
`omeco` crate — no extra dependencies, nothing added to the library's public
surface.

### Instance provenance

`instances/*.json` were copied verbatim from `research/benchmark/targets/` when
the artifact was assembled. The copies committed on `master` are now canonical:
a paper run never reads an instance from a research branch. They live here so a
checkout of `master` alone reproduces the paper — a benchmark that also needs
some other branch is not reproducible.

The one exception is `petersen`, which is read from the library's own
`benchmarks/graphs/petersen.json`: it is a shared fixture the rest of the test
suite already uses, and duplicating it here would create two copies to keep in
step.

All instance files share one schema, the same as `benchmarks/graphs/*.json`:

```json
{"name": …, "description": …, "ixs": [[label, …], …], "iy": [label, …],
 "sizes": {"label": dimension, …}}
```

with integer labels. `description` records where each network came from.

## Running

From the repository root:

```bash
# Small master set.
python3 benchmarks/paper/run_master.py \
    --set ci \
    --out ci.json

# The paper's full set; the large instances dominate.
python3 benchmarks/paper/run_master.py \
    --set full \
    --out full.json

# Figure 2(b): deterministic surfacecode_d21 trajectory (32 rounds).
python3 benchmarks/paper/run_master.py \
    --set figure2b \
    --out figure2b.json
python3 benchmarks/paper/verify_figure2b.py figure2b.json
python3 benchmarks/paper/plot_figure2b.py figure2b.json figure2b.svg

# Dense current-master mechanism trace: 10 trajectories × 128 calls.
python3 benchmarks/paper/run_master.py \
    --set waist_trace \
    --out waist_trace.json
python3 benchmarks/paper/verify_waist_trace.py waist_trace.json
```

The wrapper defaults to `benchmarks/paper/manifest.json` and the repository
root. Its flags are:

- `--manifest <file>` — tracked manifest to read.
- `--set <name>` — set within the manifest: `ci`, `figure2b`, `waist_trace`, or `full` (required).
- `--out <file>` — where to write the JSON artifact (required).
- `--repo-root <dir>` — root that instance paths in the manifest resolve
  against; every selected instance must remain inside and tracked by this
  checkout.
- `--cargo <path>` — Cargo executable; defaults to `$CARGO`, then `cargo`.

The wrapper writes the artifact to `--out` and the provenance sidecar to
`<out>.provenance.json`. It publishes neither file from a failed run. The
underlying Rust runner remains directly available for CI regression checks and
development, but its ungated output is not paper data.

Six Make targets wrap the benchmark workflow:

```bash
make paper-bench-check  # ci set -> a temporary file -> check.py against expected/ci.json
make paper-master-gate-test  # unit-test the revision/provenance gate
make paper-figure2b-check  # 32 rounds -> semantic verifier + exact artifact check
make paper-figure2b-plot   # committed JSON -> publication-ready static SVG
make paper-waist-trace     # gated 1,280-call mechanism artifact + semantic check
make paper-bench        # gated full set -> expected/full.json + provenance
```

To compare two artifacts:

```bash
python3 benchmarks/paper/check.py fresh.json expected.json
# OK: 17 results compared
```

`check.py` exits 0 on agreement and 1 on the first disagreement it finds, naming
the offending `(instance, arm)` and field. It fails on a `format`/`set`
mismatch, on differing `(instance, arm)` key sets, on differing key sets *within*
a result, and on any float leaf differing by more than a **relative** `1e-9`
(with an absolute `1e-9` floor). Integers must match exactly.

## Arms

| Arm | What it runs |
|---|---|
| `greedy` | `GreedyMethod::default()` — the cheap baseline. |
| `treesa` | `TreeSA::default()` with the manifest's overrides and `surgery_iters: 0` — annealing alone. |
| `treesa_rounds` | The reduced-network `treesa` result of the *same* configuration, then `rounds` interleaved surgery and cold span-gated fine-tuning rounds (`anneal_surgery_rounds`, the paper's Algorithm 1), then splice-back. |
| `treesa_surgery_rule` | One `TreeSA::default()` run with `surgery_probability` set from the required manifest `probability`; surgery replaces local sweeps at the current temperature. |
| `treewidth` | `Treewidth::default()`, on instances flagged `"treewidth": true` only. |

`treesa` and `treesa_rounds` are deliberately built from the same reduced-network
seed computation, so a row pair at equal `ntrials`/`niters` isolates exactly what
the rounds loop adds. Both splice back only after their reduced-network search.
The runner composes `simplify`, `anneal_surgery_rounds`, and `splice` directly;
that is bit-identical to setting `TreeSA::surgery_iters` (a library test pins
this) and additionally exposes the per-round trajectory.

`treewidth` is opt-in per instance because the elimination-order heuristic is
only meaningful on the structured, low-treewidth instances — running it on a
random circuit costs time and reports nothing anyone wants to read.

## Output schema

```json
{
  "format": 2,
  "set": "ci",
  "results": [
    {
      "instance": "dbn_13",
      "arm": "treesa_rounds",
      "tc": 32.924983,
      "sc": 29.0,
      "rwc": 30.808836,
      "curve": [
        {
          "round": 0,
          "tc_before": 38.329681,
          "tc_after_surgery": 34.087463,
          "tc_after_anneal": 35.103284,
          "tc_retained": 34.087463,
          "surgery_accepted": true
        }
      ]
    }
  ]
}
```

- `tc`, `sc`, `rwc` are `contraction_complexity` in log2 scale.
- `curve` appears on `treesa_rounds` rows only. It is the log2 time-complexity
  trace from `RoundsReport::round_trace`: the retained incumbent before the
  round, the surgery candidate, the fine-tuning endpoint, and the retained
  incumbent afterward. `tc_after_anneal` may be worse after fine tuning, but
  `tc_retained` cannot increase and is the tree used by the next round.
  `surgery_accepted` marks the rebuild events plotted as crosses in the paper's
  Figure 2(b).
- When the manifest sets `trace_cuts: true`, each curve point also contains a
  `waist` object with the exactly rescored incumbent partition cut, best
  comparable-balance FM cut, waist-node cost, and rebuild attempt/accept flags.
  Ordinary benchmark sets omit this object and remain byte-compatible with
  their existing artifacts.
- `results` is sorted by `(instance, arm)`, independent of manifest order.
- `format` is the schema version; bump it when a field's meaning changes, so
  that `check.py` refuses to compare artifacts across the change.

## Determinism contract

**On one machine, the artifact is byte-identical across runs.** Not "equal
within tolerance" — the same bytes, and `diff` is the honest check there. CI
compares with `check.py` instead, because the committed `expected/ci.json` and
the CI runner need not share a platform. The guarantees behind byte-equality on
one machine:

- Every RNG is seeded from a fixed constant plus a loop index. Trials are run in
  parallel with rayon but collected in index order, and the best is chosen by a
  stable `min_by`, so thread scheduling cannot change the result.
- The rounds loop counts rounds; it never watches the clock. Nothing in the
  pipeline is budgeted by wall time, so a slow machine and a fast machine take
  the same path.
- The artifact carries no timestamp, hostname, thread count, or library
  version — nothing that would make two correct runs differ.
- Floats are rounded to six decimals before serialization, and every container
  in the output path is an ordered struct or an explicitly sorted `Vec`. No
  `HashMap` iteration order reaches the file.

**Across platforms, expect agreement to a relative `1e-9`, not to the byte.**
`libm`'s `exp`/`log` implementations differ between platforms and the annealer's
Metropolis test calls them millions of times; a last-bit difference there can
tip an accept/reject decision and change the whole trajectory. In practice the
complexity metrics land in the same place, which is what `check.py`'s tolerance
checks. So `check.py` is what CI runs, since the committed artifact and the
runner need not share a platform; `diff` is the stricter check available when
they do — regenerating locally and diffing against the committed file tells you
whether *anything at all* moved.

The tolerance is relative — `abs(a - b) <= max(1e-9, 1e-9 * max(abs(a), abs(b)))`
— because the fields being compared span twelve orders of magnitude. `tc`, `sc`
and `rwc`, including every `curve` complexity, are log2 quantities of order 10,
where an absolute `1e-9` is a reasonable band.

## Manifest policy

The manifest is the *entire* configuration. Nothing about a benchmark run may
live in a shell script, an environment variable, or a default buried in the
runner — if it changes a number, it is in `manifest.json` and it shows up in
`git diff`.

Allowed keys:

- Set: `arms`, `instances`.
- Arm `greedy`: none; `{}` is the only legal value.
- Arm `treesa`: `ntrials`, `niters`.
- Arm `treesa_rounds`: `ntrials`, `niters`, `rounds` (required).
- Arm `treesa_surgery_rule`: `ntrials`, `niters`, `probability` (required).
- Instance: `name`, `path` (repo-root-relative), `treewidth` (required).

Any other key is a hard error naming the key. Omitted `ntrials`/`niters` mean
"use `TreeSA::default()`", which is what the `full` set does.

**Reductions must be visible.** The `ci` set runs with `ntrials: 4` and
`rounds: 2` against the defaults of 10 and the `full` set's 8. That is a
deliberate, reviewable weakening, written down where a reviewer reading the diff
will see it. If the `ci` set ever needs to get cheaper again, the way to do it is
to lower these numbers in this file — never to shorten a schedule inside the
runner, skip an instance from a wrapper script, or special-case CI in code. A
benchmark that is quietly weaker than it looks is worse than no benchmark.

Adding an instance means adding a file to `instances/` and a row to the
manifest — the runner has no instance list of its own and no discovery pass over
the directory, so an instance nobody declared is never silently benchmarked.

## Scope of the artifact

The `full`, `figure2b`, and `ci` sets are the canonical sources for the paper's
omeco results. Archived development campaigns may be useful for debugging or
historical comparison, but they cannot supply manuscript numbers. When a paper
table and a fresh current-`master` run disagree, the fresh run wins and the
paper must be updated.
