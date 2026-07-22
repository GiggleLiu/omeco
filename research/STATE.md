# Autoresearch State

- stage: run               # topics | db | validator | run | done
- topic: beat-existing-optimizers  # slug of the chosen topic once stage >= db
- batch_size: 10           # attempts per cycle
- time_limit_seconds: 300  # hard wall-clock limit per scored run
- authorized_rounds: 0     # cycles the loop may run without user review
- next_attempt: 1          # next .worktrees/attempt-NNN number
- next_cycle: 1            # next reflection cycle number
- gates:
  - survey_gate: passed 2026-07-22  # pending | passed YYYY-MM-DD
  - validator_gate: passed 2026-07-22  # pending | passed YYYY-MM-DD
- validator_env: fallback (no Docker on this macOS host; sandbox-exec deny-network + holdout-read-denial, harness-enforced wall/CPU limits)
- overrides: (none)        # every user-approved protocol deviation, dated
