# Paper benchmark artifact

Every number in the paper's released-artifact table is produced here, by the
released library, on the reader's own machine. There is no reference host and no
recorded oracle: the artifact is a *program plus a manifest*, and reproducing
that table means running it. The paper's headline campaign numbers are a
different measurement — wall-clock-budgeted on one fixed host — and this
directory does not claim to reproduce them; see
[Calibration vs the frozen campaign](#calibration-vs-the-frozen-campaign) for
the size of the gap and why it is there. The outputs under `expected/` are committed only so that
they can be *re-derived* — CI regenerates the `ci` one on every push and fails on
a single changed field, so a committed number can never quietly drift away from
what the code does.

This directory is self-contained — manifest, instances, checker:

| Path | Role |
|---|---|
| `manifest.json` | Which instances, which optimizer arms, which overrides. The only place a benchmark parameter may be set. |
| `instances/` | The thirteen paper instances, as JSON tensor networks. |
| `expected/ci.json` | The `ci` set's output, re-derived and compared by `make paper-bench-check` (and by CI). `expected/full.json` is what `make paper-bench` writes. |
| `check.py` | Compares two artifacts and exits nonzero on any disagreement. Standard library only, Python 3.8+. |
| `README.md` | This file. |

The runner itself is `omeco/examples/paper_bench.rs`, an example binary in the
`omeco` crate — no extra dependencies, nothing added to the library's public
surface.

### Instance provenance

`instances/*.json` are copied verbatim from `research/benchmark/targets/` on the
research branch (`jg/autoresearch`), which is where the paper campaign draws its
inputs from; the campaign and this benchmark therefore run on the same bytes.
They live here rather than being referenced across branches so that a checkout
of this branch alone can reproduce the paper — a benchmark that only runs if you
also happen to have some other branch checked out is not reproducible.

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
# Small set: the one CI runs on every push (under a minute on a modern laptop).
cargo run --release --example paper_bench -p omeco -- \
    --manifest benchmarks/paper/manifest.json \
    --set ci \
    --out ci.json

# The paper's set. ~21 min on an Apple-silicon laptop; the large instances dominate.
cargo run --release --example paper_bench -p omeco -- \
    --manifest benchmarks/paper/manifest.json \
    --set full \
    --out full.json
```

Flags:

- `--manifest <file>` — manifest to read (required).
- `--set <name>` — set within the manifest, `ci` or `full` (required).
- `--out <file>` — where to write the JSON artifact (required).
- `--repo-root <dir>` — root that instance paths in the manifest resolve
  against. Defaults to `.`, so the commands above must be run from the
  repository root; pass it explicitly to run from elsewhere.

Progress goes to stderr, the artifact to `--out`. Anything the user can get
wrong — an unknown flag, a missing instance file, malformed JSON, a mistyped
manifest key — prints a message naming the problem and exits with status **2**.
The runner never reports a typo by silently falling back to a default.

Two Make targets wrap the same commands:

```bash
make paper-bench-check  # ci set -> a temporary file -> check.py against expected/ci.json
make paper-bench        # full set -> expected/full.json (~21 min, see below)
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
| `treesa_rounds` | The `treesa` result of the *same* configuration, then `rounds` interleaved anneal–surgery rounds (`anneal_surgery_rounds`, the paper's Algorithm 1). |
| `treewidth` | `Treewidth::default()`, on instances flagged `"treewidth": true` only. |

`treesa` and `treesa_rounds` are deliberately built from the same seed
computation, so a row pair at equal `ntrials`/`niters` isolates exactly what the
rounds loop adds. The runner calls `anneal_surgery_rounds` directly rather than
setting `TreeSA::surgery_iters`; the two are bit-identical (a library test pins
this), and the direct call additionally exposes the per-round trajectory.

`treewidth` is opt-in per instance because the elimination-order heuristic is
only meaningful on the structured, low-treewidth instances — running it on a
random circuit costs time and reports nothing anyone wants to read.

## Output schema

```json
{
  "format": 1,
  "set": "ci",
  "results": [
    {
      "instance": "dbn_13",
      "arm": "treesa_rounds",
      "tc": 32.924983,
      "sc": 29.0,
      "rwc": 30.808836,
      "curve": [
        {"round": 0, "score": 3620306748202.0137},
        {"round": 1, "score": 5525217500107.966}
      ]
    }
  ]
}
```

- `tc`, `sc`, `rwc` are `contraction_complexity` in log2 scale.
- `curve` appears on `treesa_rounds` rows only. It is
  `RoundsReport::round_scores` — the *trajectory*, not a running minimum, so
  entries may increase: round `r + 1` continues from round `r`'s tree even when
  that tree was worse than the incumbent best. That is the escape mechanism the
  paper is about, and flattening it to a running minimum would hide it. `round`
  is zero-based, matching `RoundsReport::best_round`. A `curve` entry records
  that round's *post-anneal* candidate only, while the returned tree may instead
  be a round's pre-anneal surgery tree — so a row's `tc`/`sc` can be better than
  every entry in its own `curve`.
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
and `rwc` are log2 quantities of order 10, where an absolute `1e-9` is a
reasonable band. A `curve` score is a raw annealing score, i.e. a weighted sum of
`2^tc`-scale terms, so on realistic instances it is around `1e12`; there the same
absolute band would be finer than the spacing between adjacent doubles, and would
turn "compare across platforms" into "demand bit-exactness". Six-decimal rounding
is likewise a no-op at that magnitude. Scaling the band with the value keeps one
rule honest for both.

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

## Calibration vs the frozen campaign

The full set takes about 21 minutes end to end on an Apple-silicon laptop
(1266 s for 40 results; the slowest single instance × arm is `nqueens_28`'s
`treesa_rounds` at 6 minutes).

The paper's headline numbers come from a wall-clock-budgeted campaign on a
fixed host, not from this artifact. It is worth knowing how far apart the two
are, so here are three instances side by side. Campaign values are the *best*
`tc` recorded anywhere in the paper repository's `data/huawei_campaign.json`
for that instance; `treesa_rounds` values are the committed `expected/full.json`
rows produced by the `full` set (`TreeSA::default()`, `rounds: 8`).

| Instance | Campaign best tc (huawei, wall-clock budget) | `treesa_rounds` tc (rounds = 8, deterministic) | Gap |
|---|---|---|---|
| `surfacecode_d13` | 30.485 — `p4_family["13"]`, the surface-code family sweep: 5 reps at the 90 s budget, both methods tie | 30.898 | +0.413 |
| `surfacecode_d21` | 47.364 — `p2_budget_scaling`, our TreeSA baseline (`ref`) at the 900 s budget | 49.180 | +1.816 |
| `ksg` | 36.356 — `p3_distributions`, best of 15 reps at the 90 s budget | 38.291 | +1.935 |

(The campaign file records `tc` only, so there is no `sc` column to compare;
this artifact's `sc` for the three rows is 24, 40 and 27 respectively.)

The gaps are expected and they are not a defect. The two sides are budgeted by
different quantities: `rounds` is a *schedule* budget — eight rounds is eight
rounds on any machine, forever — while the campaign gave each attempt a fixed
number of seconds on one 2-core x86 VM and took the best of many repetitions.
A wall-clock budget buys more search on a faster host and less on a slower one,
which is exactly the property this artifact is built to *not* have. What
`expected/full.json` claims is bit-reproducibility: run the same commit on the
same machine and you get the same file. Across machines the claim is weaker and
partly untested — agreement to `1e-9` is *confirmed* for the `ci` set, which the
ubuntu CI job re-derives and compares on every push, and *expected* for
`full.json` by the same mechanism, but nobody has run the full set on a second
platform. It does not claim to reproduce a record. Raising
`rounds` in the manifest closes the gap — the rounds loop is the mechanism the
campaign was exploiting too — at a cost that is proportional and, unlike a
timer, still deterministic.

## Frozen campaign data

This directory reproduces the numbers. It does not archive them. The frozen
per-run campaign data behind the published figures — every trial's score, timing
and provenance, far more than the summary tables show — lives with the
manuscript, in the paper repository
(<https://github.com/GiggleLiu/contraction-order-frontiers>, private). When a
paper table and a fresh `paper_bench` run disagree, the paper repository says
which library revision produced the table; this directory says what the current
one does.
