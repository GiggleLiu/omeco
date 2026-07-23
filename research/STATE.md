# Autoresearch State

- stage: run               # topics | db | validator | run | done
- topic: beat-existing-optimizers  # slug of the chosen topic once stage >= db
- batch_size: 10           # attempts per cycle
- time_limit_seconds: 300  # hard wall-clock limit per scored run
- authorized_rounds: 1     # cycles the loop may run without user review
- next_attempt: 32         # next .worktrees/attempt-NNN number
- next_cycle: 7            # next reflection cycle number
- gates:
  - survey_gate: passed 2026-07-22  # pending | passed YYYY-MM-DD
  - validator_gate: passed 2026-07-22 (v2)  # pending | passed YYYY-MM-DD
- validator_env: fallback (no Docker on this macOS host; sandbox-exec deny-network + holdout-read-denial, harness-enforced wall/CPU limits)
- overrides:
  - 2026-07-22: v1 pass/fail bar replaced by optimization/leaderboard mode
    (user direction: "do not use pass or fail to measure"); targets reduced
    to reg3_250 + sycamore_m20 at 90s/instance.
  - 2026-07-22: holdout/twin anti-memorization guards dropped by user
    decision ("no twins"); risk accepted, mitigations: per-run relabeling,
    confirmation runs, code review of record-setters.
  - 2026-07-23: objective simplified to PURE tc (user direction: "ignore the
    sc_target, only optimize tc, unbounded"); sc-cap guard removed, cap-era
    leaderboard archived to leaderboard_v2cap_archive.json, references
    re-measured under pure tc.
  - 2026-07-23: cycle 7 authorized by user directive ("push towards a better
    algorithm for contraction order, that worth publication"). Primary axis
    pivoted to ANYTIME performance (time-to-frontier), per the topic's own
    definition ("or fast to compute to achieve the same contraction
    complexity") — final-tc at 90 s is saturated across 13 mechanisms and 2
    toolchains (cycles 1-5 evidence). Validator extended to v2.2: harness-
    clocked polling of the anytime out.json writes, TTF leaderboard with
    confirm-twice/worse-of-two. tc records remain live (window-exact splice
    lottery). Memorization risk on TTF inherits the accepted "no twins"
    override; mitigations unchanged (per-run relabeling, code review of
    record-setters).
