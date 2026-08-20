# Autoresearch State

- stage: run               # topics | db | validator | run | done
- topic: beat-existing-optimizers  # slug of the chosen topic once stage >= db
- batch_size: 10           # attempts per cycle
- time_limit_seconds: 300  # hard wall-clock limit per scored run
- authorized_attempts: 0   # cycle 13 (066) executed 2026-08-20: falsified per
                           # pre-registered gate; campaign winds down to write-up.
                           # Was: 1, authorized by user 2026-08-20 ("1"):
                           # single attempt 066 = composite reg3 TTF re-claim
                           # (labelled 067 in the soft-gate discussion).
                           # Wind-down track runs in parallel (PR #40, library
                           # port, write-up). [cycle 12 (062-065) done; soft
                           # gate passed]
                           # 064 set confirmed sycamore_m20 TTF record 5.274s.
                           # Was: 4, authorized by user 2026-08-20 ("yes, go"):
                           # 062 phase-switched composite (improve/061),
                           # 063 confirmation-robust early descent (improve/061),
                           # 064 event-triggered switch (draft),
                           # 065 early-descent mechanism trace (measurement).
                           # Execution pattern as cycle 11 (codex, huawei dev
                           # benches <=600s, validator local).
                           # [cycle 11 (059-061) done: 061 set confirmed
                           # reg3_250 TTF record 7.069s]
                           # ("continue the autoresearch with these 3 plans"):
                           # 059 beta(span,t) freeze-out, 060 activity-based
                           # worklist freezing, 061 targeted waist-band reheat.
- next_attempt: 67         # 057/058 consumed by the attempt-057/058 campaign;
                           # counter was stale at 57, corrected 2026-08-20
- next_cycle: 14
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
  - 2026-07-25: measurement moved to remote host `huawei` (2-core Ubuntu ECS,
    dedicated/quiet) after repeated local interruptions — full 305-job paper
    campaign rerun there same-machine (research/paper_data/runs_huawei/),
    matched-budget Julia ladder v2 running through the benchmark repo harness.
  - 2026-07-25: user directed next batch: show the new methods work better on
    inference tasks from ~/.julia/dev/TensorInference — "handle the hard
    instances properly". 65 uai2014 MAR instances exported (validator schema,
    faithful to TensorNetworkModel full-card construction); 5 s hardness scan
    ranked DBN/linkage hardest; 200-job batch (10 hard instances x
    {ref,a038,a050,a054} x 5 reps x 90 s) + Julia baselines (incl.
    TensorInference default TreeSA) chained on huawei after the ladder.
  - 2026-08-20: cycle 11 (attempts 059-061) authorized by user with three
    deviations: (a) implementation delegated to codex sessions per attempt;
    (b) dev/mechanism benchmarks run on huawei, hard cap 600 s per benchmark
    run (user: "each benchmark at most cost 10min"); (c) validator scoring
    stays on the local canonical host (sandbox-exec env; leaderboard records
    were measured here — scoring elsewhere would break record comparability).
    Attempt worktrees base on jg/surgery-v2 (eeaccf5) to reuse the
    RoundsOptions/warm-rebuild machinery from PR #40.
