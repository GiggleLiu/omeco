# Autoresearch State

- stage: run               # topics | db | validator | run | done
- topic: beat-existing-optimizers  # slug of the chosen topic once stage >= db
- batch_size: 10           # attempts per cycle
- time_limit_seconds: 300  # hard wall-clock limit per scored run
- authorized_rounds: 0     # cycles the loop may run without user review
- next_attempt: 11         # next .worktrees/attempt-NNN number
- next_cycle: 2            # next reflection cycle number
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
