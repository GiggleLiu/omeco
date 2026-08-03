# Current-master waist-call trace design

## Goal

Recompute the paper's dense waist-cut measurement from the current `master`
algorithm without changing optimization behavior.  The artifact must expose the
incumbent waist cut and the best comparable-balance FM cut for every completed
surgery call, while retaining the existing clean-`origin/master` provenance
gate.

## Chosen design

Add an immutable `WaistCallTrace` record to `WaistReport`.  A record is appended
after FM has produced an exactly rescored candidate and is updated only with the
two downstream facts needed for diagnosis: whether a rebuild was attempted and
whether it was accepted.  The record contains the incumbent partition cut,
incumbent waist-node cost, and best FM cut.  It does not add callbacks, logging,
random draws, clocks, or acceptance branches, so tracing cannot perturb the
proposal stream or optimizer result.

Propagate the single call record through `RoundTrace` and serialize it from the
existing `paper_bench` runner.  A new `waist_trace` manifest set runs 128
interleaved surgery/cold-fine-tuning rounds for each of five deterministic tensor
relabelings of `surfacecode_d21` and `ksg`.  This yields 1,280 fixed-work calls,
close to the historical 1,226-call sample while removing its wall-clock stopping
condition.  Relabeling seeds are explicit manifest data; all ten trajectories
remain byte-reproducible on one machine.

## Validation

- Unit-test that every completed FM comparison emits one exact trace record.
- Differential-test traced and untraced optimizer outputs by pinning the existing
  deterministic tree and report fields.
- Extend paper-runner schema tests for optional relabeling and serialized cut
  fields.
- Run the gated `waist_trace` set from a clean commit at `origin/master` on the
  Huawei host, recording revision, binary, manifest, input, and output hashes.
- Regenerate Figure 2(a) from the new artifact before restoring the manuscript's
  mechanism claim.  Figure 2(b) and the Shi-Xin Zhang TensorCircuit table remain
  separate current-master artifacts.

## Rejected alternatives

A source-patched Huawei binary violates the paper gate.  Aggregate
`cheaper_cuts` counters cannot reproduce the scatter or audit individual calls.
Repeating a wall-clock budget preserves the old protocol but makes call count and
sampling machine-dependent; fixed work is both gentler to compare and consistent
with the revised algorithm.
