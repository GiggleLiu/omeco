# Attempt 066 — Composite reg3 TTF re-claim (later switch)

- **attempt**: 066 (labelled 067 in the cycle-12 soft-gate discussion)
- **date**: 2026-08-20
- **kind**: improve
- **parent**: 062 (phase-switched composite, this worktree)

## Hypothesis (pre-registered)

062 missed the reg3_250 TTF record bar by 0.09 s (runs 3.9 s / 5.745 s
vs bar 5.655 s = 0.8 x 7.069 s) with switch fraction q = 0.25. Evidence
for a later switch on expanders: (a) 065 — band-phase descent is sc-led
on expanders, so more band time drops the bottleneck rank further before
the front polishes; (b) 062's own dev bench — the best FIXED fraction
for reg3 final tc at matched sweeps is 40% (41.68 vs 42.26). ATOMIC
CHANGE: default switch fraction q: 0.25 -> 0.40. Nothing else.

Claim: reg3_250 TTF record under worse-of-two (both runs <= 5.655 s),
with the reg3 tc tie (39.882) retained.

## Expected evidence

Validator primaries (90 s): confirmed reg3_250 TTF record; tc delta on
reg3 >= 0 (tie retained). sycamore_m20 makes no claim (its record 5.274 s
belongs to 064's immediate-switch regime; q=0.40 may worsen its curve —
report, don't claim). Dev bench (huawei, <= 600 s): reg3_250 + ksg tc(t)
at q in {0.25, 0.40} x 2 relabelings, showing the q=0.40 curve reaches
record-eps earlier or equal on reg3 in >= 3 of 4 comparisons.

## Falsification

If worse-of-two TTF again lands > 5.655 s while the q=0.40 dev curves
are not faster than q=0.25, the record miss is relabeling variance, not
switch placement — record both curves, close the direction, and the
write-up proceeds with the 061/064 records as they stand.

## Constraints (validator contract — non-negotiable)

As parent 062 (this worktree): example `attempt`, contract
`attempt <graph.json> <budget_ms> <out.json>`, eager atomic writes,
single thread, relabeling-invariant, pure tc, knobs functions of n,
fixed seed, LINEAR ramps. ATT_PARENT=1 stays pure 061;
ATT_FIXED_SWITCH=q still overrides (the change is only the DEFAULT q).
dev_bench.sh hard cap 600 s total, 2-core-Linux-safe, budget plan first.

## Outcome (recorded 2026-08-20)

**Validator (canonical host):** score -0.2201, NO RECORD, claim
falsified: reg3_250 tc=40.321 (delta -0.438), ttf=inf on this
relabeling pair. sycamore_m20 tc=61.524 (-0.003, near-tie), ttf 15.4 s
(record 5.274 s untouched).

**Dev bench (huawei):** the pre-registered evidence gate FAILED 0/4 —
q=0.40 reaches record-eps at 9.57 s vs q=0.25's 8.05-8.22 s on every
cross-relabel comparison, even though q=0.40's matched-work final tc is
slightly better (39.8817; ksg 38.5965 vs 38.6075).

**Verdict (honest, falsification clause applied):** a later switch is
better for final tc at matched work but WORSE for time-to-frontier —
the band phase's extra rank-work delays the front's polish past the eps
threshold. 062's 0.09 s record miss was relabeling variance, not switch
placement. The direction closes: the write-up proceeds with the 061
(reg3 7.069 s) and 064 (sycamore 5.274 s) records and 062's composite
(q=0.25) as the recommended general-purpose schedule.
