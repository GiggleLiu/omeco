# Draft PR: surgery v2, matched controls, and work-matched ablation

## Summary

- expose `anneal_refine_rounds` + `RoundsOptions`, including the exactly matched
  cold-only arm;
- add opt-in warm-restricted side initialization and local subtree surgery;
- add the opt-in `RoundsSchedule` composite fine-tuning schedule (below);
- add deterministic seeded TreeSA repetitions, fine-tune sweep accounting, and
  a resumable JSONL ablation driver with process-level sharding;
- guard every rounds arm after splice-back against the spliced baseline and
  record `post_splice_guard_triggered` in each JSONL row;
- commit a four-instance, two-label R=8 smoke and a Markdown summarizer.

All defaults remain unchanged. `anneal_surgery_rounds` delegates to
`anneal_refine_rounds(..., &RoundsOptions::default())`; `refine` and
`refine_capped` retain their historical signatures and default implementation
path. Added tests compare serialized trees and complete reports for the default
wrapper and reconstruct the cold-only fine-tuning path independently.

## Composite schedule

`RoundsOptions::schedule` selects which fine tuner each round runs. It replaces
only the rounds fine-tuning pass; surgery, the splice-back guard, and the global
never-worse acceptance gate are untouched.

- `RoundsSchedule::Cold` (default) is the historical coarse-to-fine, span-gated
  cold pass. It is byte-identical to the previous behaviour: same proposal
  order, beta selection, RNG consumption, and serialized tree.
- `RoundsSchedule::BandReheatThenFront { switch_fraction }` ports autoresearch
  attempts 061/059/062. It runs in two phases:
  1. **Band reheating (061).** Every epoch of
     `ceil(sqrt(n)).clamp(8, 32)` sweeps snapshots the nodes within a
     size-scaled cost band of the current waist (1/2/4 bits for
     `n < 256` / `n < 2048` / larger) plus their root paths, and anneals that
     band on a warmer beta ramp while the rest of the tree stays cold.
  2. **Continuous freeze-out front (059).** Each span level then descends a
     one-octave log-span front from its own span to the next finer one, with
     the finest level descending to a span of 1. Nodes ahead of the front
     accept only non-regressing moves, nodes inside it ramp from warm to cold,
     and nodes behind it stay cold.

The switch between the phases (062) is clamped: `switch_fraction` is first
clamped to `[0, 0.40]`, then the resulting sweep index is constrained to occur
no earlier than two band epochs and no later than 40% of the planned sweeps.
NaN selects the earliest permitted switch; infinities saturate to the nearest
bound. The clamp is computed against every planned sweep, and the phase counter
advances on every sweep, so the realized front fraction matches the request on
every span ladder — including the single-level ladder that `n <= 60` produces.

Both variants are documented in rustdoc on `RoundsSchedule` and in the mdBook
default-pipeline chapter. Tests cover default identity against the historical
path, band membership, front beta monotonicity, switch clamping (including the
NaN/infinity and zero-sweep edges), the front phase actually running on a
single-level span ladder, and an end-to-end 8x8 grid that stays valid and never
worse than its seed.

## Why

The previous rounds result conflated a global surgery call with repeated cold,
span-gated fine tuning. The new cold-only control separates those mechanisms.
Warm restriction avoids discarding optimized side topology, while local scope
avoids promoting a deep waist cut to the root over the full network.

## Smoke evidence

The committed smoke uses 200,000 target baseline visits (a deliberate runtime
override from the 140-million-visit full protocol), R=8, two labels, and the
four requested instances. It completed 40 unique rows and resumed without
changing the JSONL bytes.

| instance | cold-only median tc | greedy/root | warm/root | greedy/local | warm/local |
| --- | ---: | ---: | ---: | ---: | ---: |
| dbn_13 | 45.215401 | 38.298190 | 36.944167 | 39.316334 | 33.659685 |
| petersen | 8.491853 | 8.491853 | 8.491853 | 8.491853 | 8.491853 |
| qft_27 | 38.003223 | 32.121959 | 31.226241 | 30.319744 | 31.915470 |
| surfacecode_d9 | 21.965192 | 21.808134 | 21.886859 | 21.965192 | 21.965192 |

Paired W/T/L against cold-only: greedy/root 6/2/0, warm/root 5/3/0,
greedy/local 4/4/0, warm/local 4/4/0. These are smoke checks, not paper
estimates.

The Local root-output-order defect was representational only: time complexity
is invariant under permutation of the final output axes, so it does not change
the numeric results of the campaign already running from the pre-fix binary.
Rows produced by the updated binary additionally contain the post-splice guard
flag described above.

## Full campaign commands

```bash
INSTANCE_DIR=/Users/liujinguo/rcode/contraction-order-frontiers/benchmarks/omeco/instances

RAYON_NUM_THREADS=1 cargo run --release -p omeco --example surgery_ablation -- \
  --instances "$INSTANCE_DIR" \
  --out benchmarks/surgery_ablation/results/full-preprocessed.jsonl

RAYON_NUM_THREADS=1 cargo run --release -p omeco --example surgery_ablation -- \
  --instances "$INSTANCE_DIR" \
  --out benchmarks/surgery_ablation/results/full-raw.jsonl \
  --raw

python3 benchmarks/surgery_ablation/summarize.py \
  benchmarks/surgery_ablation/results/full-preprocessed.jsonl \
  --out benchmarks/surgery_ablation/results/full-preprocessed.md

python3 benchmarks/surgery_ablation/summarize.py \
  benchmarks/surgery_ablation/results/full-raw.jsonl \
  --out benchmarks/surgery_ablation/results/full-raw.md
```

The default is serial because set-A wall timing is part of the result. `--jobs
N` is available as explicit process-level parallelism when throughput matters
more than uncontended timing.

## Validation

- `make check-all`
- 547 library tests passed; one performance test ignored
- 4 example tests passed
- 43 doctests passed; one JSON-writing doctest ignored
- smoke JSONL contains 40/40 unique resumable keys
- Local Greedy and WarmRestricted accepted-rebuild regressions preserve the
  exact, deliberately unsorted root output order

## Known limitations

- The machine-independent work coordinate counts TreeSA node visits and cold
  fine-tune sweeps times `(n-1)`; FM and greedy/warm rebuild work appears only
  in wall time.
- Wall time is machine/load dependent, especially with opt-in `--jobs N`.
- The smoke visit override is intentionally much smaller than the full protocol
  and cannot support quality conclusions.
- Tree conversion remains recursive and inherits the crate's documented deep
  tree stack limit; the driver runs each group on the same large-stack policy.
- The rebuild trigger still compares node cost with cut cost (issue #23), which
  is explicitly out of scope for this change.
