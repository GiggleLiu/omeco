# Attempt 027 — Profile-aware SA (sharpened γ-energy + near-max-node targeting)

- **Attempt:** 027
- **Date:** 2026-07-23
- **Kind:** draft
- **Parent:** none (pre-registered test flowing from attempt-022's width-optimality finding)

## Hypothesis

Search guided by the cost PROFILE escapes the pure-tc plateau where twelve
profile-blind mechanisms stall at the records (reg3_250 tc=39.95,
sycamore_m20 tc=61.544). attempt-022 proved the sycamore frontier is
width-optimal (max contraction = 53 = minimum possible); the residual tc is the
PROFILE — how many contractions sit near the maximum (frontier overhead
8.5 ≈ log2(561) ⇒ MANY near-max contractions). Two coupled mechanisms:

1. **Sharpened energy.** Anneal on `E_γ = log2 Σ_v 2^{γ·cost_v}` with γ ramping
   1.0→1.5 across each cycle. γ=1 is exactly tc, whose gradient for shaving one
   near-max node is ~1/count (invisible at 300+ nodes). γ>1 up-weights the
   near-max nodes (softmax weight 2^{γ·cost_v}/Σ), concentrating annealing
   pressure exactly where tc lives.
2. **Targeted move selection.** Pick the SA move's node from the top-cost decile
   with probability p≈0.7, uniformly otherwise — multiplying the rate of
   profile-relevant proposals (implemented per-node: top-decile nodes proposed
   w.p. 1, others w.p. 1/21, giving 0.7 proposal mass on the top decile).

Combined, profile-aware SA finds width-optimal trees with FLATTER profiles
(fewer near-max nodes) and strictly lower tc.

## Expected evidence

- **Primary:** confirmed record on either instance — tc < 39.90 (reg3_250) or
  tc < 61.49 (sycamore_m20).
- **Secondary (even without a record):** measurably flatter profile — fewer
  nodes within 1.0 of max at equal tc — reported as per-node cost histograms,
  mine vs treesa-inf's.
- **Null:** identical profiles across restarts supports "the profile is already
  optimal at the frontier" (feeds the certification story with attempt-026's
  bound).

## Implementation

Single file `omeco/examples/attempt.rs` (reuses omeco public `expr_tree`
primitives: `rule_diff`, `apply_rule_mut`, `tree_complexity`, `tcscrw`, the four
rotation rules; SA machinery adapted from siblings 015/017). Pure tc — sc handling
OFF everywhere (`sc_target = ∞`, no sc penalty). Single thread. No library edits.

Final pipeline (the profile-aware mechanisms are the novelty; the warm start is a
robustness scaffold):

1. **Greedy seed** emitted immediately (anytime fallback).
2. **omeco-TreeSA warm start** (~35 % of budget, anytime niters-doubling,
   `sc_target=∞`), best kept by true tc and emitted — a strong, ROBUST floor.
3. **Population (P=4)** seeded from the TreeSA floor: replica 0 = floor, replica 1
   = perturbed clone, replicas 2–3 = stochastic-greedy diversity.
4. **Cyclic-cool profile-aware SA** for the remaining budget: NCYCLES=4 β
   cool-downs (BETA_LO=0.02 → BETA_HI=30 within each cycle = independent descent
   attempts), round-robin over replicas with clone-the-best exploitation.
5. **Per-move energy** `ΔE = (tc1−tc0) + κ·Δpeak`, where the plain local dtc
   `tc1−tc0` (from `rule_diff`) is the workhorse gradient and `peak(c) =
   2^{α·(c−c_max)}` (α=2) is the profile-aware peak-pressure term — ≈0 for bulk
   nodes, steep near the max. κ ramps 0→0.3 only in the **cold tail** of each
   cycle (`pc ≥ 0.45`), after the plain gradient has done the bulk descent.
   `c_max` and the top-decile threshold θ are recomputed once per sweep
   (`collect_costs` / `tcscrw`); the four individual affected-node costs for
   `Δpeak` are recomputed in-file, mirroring `rule_diff`'s internal arithmetic.
6. **Targeting**: in the cold tail, a move is proposed at a node with prob 1 if
   its cost ≥ θ (top decile) and 1/21 otherwise ⇒ ≈0.7 proposal mass on the top
   decile.
7. **Emission/best** ALWAYS by TRUE tc (`tree_complexity`) ⇒ worst case = the
   TreeSA floor (the SA can only replace best on a strict true-tc improvement).

**Attribution knobs** (env): `ATTEMPT_MODE = control` (κ=0, uniform), `target`
(targeting only, κ=0), `profile` (default: targeting + κ). `ATTEMPT_{KAPPA,ALPHA,
BETAHI,TREESA,WARMUP}` tune constants. RNG seeded from `n` ⊕ wall-clock so repeat
local runs are independent samples.

## Decisions (and course corrections)

- **Global E_γ = log2 Σ 2^{γ·cost_v} was implemented first and abandoned.** At
  γ=1 (and near it) the global energy's gradient on a sub-max node is
  `2^{c}/Σ2^{c}` — vanishing for bulk nodes — so pure-global annealing never
  reaches the width-optimal frontier: reg3_250 stalled at **tc≈44.8** (vs 40),
  and the profile phase made no progress. **This is the hypothesis's own premise
  turned against it**: the "no gradient" problem is a property of the GLOBAL-tc
  energy, whereas real TreeSA uses LOCAL dtc, which DOES give every node a
  gradient. I pivoted to the correct operationalization: local-dtc workhorse +
  a `κ·Δpeak` peak-pressure term (the local-move analogue of γ-sharpening —
  same "up-weight near-max" effect without killing the bulk gradient).
- **Large κ hurts.** κ=1.0 accepts tc-INCREASING profile-flattening moves and
  steers out of the good basin (reg3_250 40.79 vs control 40.23, and a FATTER
  top). Reduced to a gentle bias κ_max=0.3, engaged only in the cold tail.
- **The in-binary "first 30 % plain, rest profile" split is confounded** — with a
  monotonic cool the first 30 % is the HOT phase (no descent happens there), so
  "control-arm best" was just the greedy seed. Replaced by (a) a cyclic schedule
  with profile mechanisms in each cycle's cold tail, and (b) a clean full-budget
  control via `ATTEMPT_MODE=control` for the LOG comparison.
- **Custom SA alone is not robust on the 560-node sycamore** — failed-descent
  runs reached tc 65–69 (both control and profile). Added the omeco-TreeSA warm
  start as a floor; this removed all catastrophic runs and made the binary
  competitive (hits ~record on both instances).

## Precheck result

**PASS** — `validate --precheck` on `precheck_chain_10` (2000 ms): `status:
scored`, `result: pass` ("structure ok"), `errors: []`, environment fallback.
Report `/tmp/pc_027.json`. Build clean; `cargo clippy` (deny-warnings) and
`cargo fmt --check` both clean.

## Results (own local runs at 90 s single-thread — NOT scored validation)

Records: reg3_250 **39.95**, sycamore_m20 **61.544** (beat threshold <39.90 /
<61.49; confirmed record = worse-of-2 beats by >0.05).

### tc across replicates (true tc, validator scorer)

| instance | ATTEMPT_MODE=control | ATTEMPT_MODE=profile | treesa-inf (local) |
|---|---|---|---|
| reg3_250 | 39.950, 39.882, 40.024, 40.070, 40.24 | **39.923**, **39.883**, 40.024, 40.197, 40.24, 40.55 | 40.024 (record seeded 39.95) |
| sycamore_m20 | 61.609, 61.680, 61.703 | **61.508**, 61.578, 61.634, 61.722, 61.733, 62.06 | 63.576 (record seeded 61.544) |

- Best single runs **touch/dip below both records**: reg3_250 39.882 (< 39.90);
  sycamore 61.508 (< 61.544 but not < 61.49). But variance is large and no
  worse-of-2 pair reliably clears the confirmed-record threshold ⇒ **no confirmed
  record expected**.
- **Profile ≈ control at equal budget** — the two columns overlap; profile shows
  no reliable advantage. The tc gains over treesa-inf come from the warm-started
  SA descent (present in BOTH modes), not from the κ/targeting mechanisms.

### Profile histograms (per-node cost, top of the distribution)

reg3_250 (nodes=249):

| tree | tc | max cost | within 1.0 of max | within 2.0 |
|---|---|---|---|---|
| treesa-inf | 40.024 | 37 | 7 | 17 |
| control (mine) | 39.882 | **36** | 19 | 26 |
| profile (mine) | 39.883 | **36** | 18 | 25 |

sycamore_m20 (nodes=560):

| tree | tc | max cost | within 1.0 of max | within 2.0 |
|---|---|---|---|---|
| treesa-inf (local) | 63.576 | 61 | 5 | 6 |
| control (mine) | 61.703 | 58 | 4 | 20 |
| profile (mine) | 62.059 | 58 | 3 | 19 |
| profile best (SYC-PROF1) | 61.508 | **57** | 17 | 65 |

### Interpretation — the pre-registered gain did NOT materialise (informative null)

1. **Profile-aware SA does not beat plain-tc control.** κ-peak-pressure and
   top-decile targeting produce trees statistically indistinguishable from the
   pure local-dtc control at equal budget. Repeated across restarts, the
   mechanisms neither lower tc nor reliably flatten the profile.
2. **Lower tc comes with a FATTER near-max shelf, not a flatter one — the
   opposite of the hypothesis.** The record-relevant move is dropping the MAX
   (width) by one unit (reg3_250 37→36; sycamore 61→57-58), which necessarily
   piles more contractions just below the new max (reg3 within-1 goes 7→18-19).
   Flops are conserved: you cannot lower the peak height without increasing the
   near-peak COUNT. tc improves only when the whole-unit width drop outweighs the
   fatter shelf.
3. **This supports the "profile is already ~optimal at the frontier" reading**
   (feeding the certification story with attempt-026's bound): given a width, the
   near-max count is essentially forced by conservation, so there is no slack for
   a profile-aware search to exploit. The remaining headroom is entirely in
   lowering the width itself — where plain local-dtc SA is already as effective as
   the profile-aware variant (both reach width 36 on reg3_250, 55–58 on sycamore).

### Deviations from the guidance

- Energy is **local-dtc + κ·Δpeak**, not the literal global `E_γ = log2 Σ 2^{γc}`
  annealed directly (which was implemented, tested, and shown to stall at tc≈44.8
  because its bulk gradient vanishes — see Decisions). `κ·Δpeak` is the
  faithful local-move realisation of "concentrate pressure on near-max nodes".
- Added an **omeco-TreeSA warm-start floor** (not in the original plan) for
  robustness on the 560-node instance; the profile SA refines it, best-gated by
  true tc.
- The in-binary 30/70 control split was replaced (confounded); clean attribution
  uses `ATTEMPT_MODE=control` full runs.
- No confirmed record was pursued (guidance: do not run scored validation).

## Scored outcome (validator v2.1 pure-tc, 2026-07-23)
- status: scored, score (mean Δtc vs pre-run records): -0.1086
- record_updates: none
- reg3_250: pass — tc=40.114 sc=34.000 record=39.95029599283971 delta=-0.163
- sycamore_m20: pass — tc=61.598 sc=53.000 record=61.54428977050287 delta=-0.054
