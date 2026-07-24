# Autoresearch State

- stage: run               # topics | db | validator | run | done
- topic: beat-existing-optimizers  # slug of the chosen topic once stage >= db
- batch_size: 10           # attempts per cycle
- time_limit_seconds: 300  # hard wall-clock limit per scored run
- authorized_rounds: 1     # cycle 10 (user 2026-07-24: waist surgery batch +
                           # Tamaki transfer — 054/055/056); cycle 9 reflected
                           # 2026-07-24 (HTML renders pending: ~/Documents
                           # TCC-blocked; md/json canonical)
- next_attempt: 57         # cycle-10 batch = 054-056
- next_cycle: 10           # next reflection cycle number
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
  - 2026-07-23: cycle 8-9 authorized (user selected scale axis + "propose new
    algorithms"; 2 rounds). Validator v2.3: scale instances (reg3_1000,
    rqc_97_m24) scored as MEDIAN-OF-3 independent runs (cycle-5 lesson:
    single-run variance exceeds method differences at n>=1000); scale scored
    runs get a raised wall limit (1200 s) to fit 3x90 s per instance —
    user-approved deviation from the 300 s limit for scale mode only.
    Planned batch: 034 multilevel V-cycle, 035 racing portfolio, 036 deep
    seed + cold refine + window-exact repair.
  - 2026-07-24: user directed import of harder instances from
    OMEinsumContractionOrdersBenchmark: sycamore_53_20_0 (real Sycamore,
    3369 tensors), surfacecode_d21 (2203), ksg (5197), nqueens_28 (4252)
    join the median-of-3 scale set; dbn_13, qft_27 added as small targets.
    --scale default group = reg3_1000 + sycamore_53_20_0 + surfacecode_d21
    + ksg; scale wall limit raised 1200 -> 2400 s to fit 4x(3x90 s) plus
    confirmation runs.
