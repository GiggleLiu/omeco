# attempt 058

- date: 2026-07-26
- kind: improvement
- parent: 053 (falsified: "bounded cheap-first VE peel")

## Hypothesis (pre-registered)

attempt-053's bounded cheap-first peel was FALSIFIED on `nqueens_28` (4086
labels): peel+residual-TreeSA lost to full-graph TreeSA at every cap (>=188 vs
134). Measured cause: good large-treewidth orders interleave core and periphery,
and a fixed peel boundary removes that freedom.

**Why 058 is NOT a repeat of that falsification** — two independent reasons:

1. **New regime.** 053 was measured where full-graph TreeSA is COMPETITIVE
   (nqueens, ~4k tensors: 134). On the UAI relational instances it is not: at
   30k-70k tensors the annealer is IMMOBILE — `uai_relational_4` (30,400
   tensors, 10,200 binary labels, treewidth 100) sits at tc~202/sc=200 at every
   budget 90-900s, double the optimum, because at that size the annealer barely
   completes a sweep. Here the alternative to peeling is not "better
   interleaving" but "no optimization at all", so the 053 interleaving objection
   does not apply.

2. **Adaptive ladder + racing fallback**, not one fixed cap. The peel cap is an
   escalating LADDER whose TOP rung is cap = infinity — full min-cost VE run to
   completion. And the full-graph anneal still RACES as a fallback (gated to
   sizes where it can actually run), so the falsified regime is protected: on
   instances where peeling/VE is useless the base annealer's result is kept.

**Claim:** on the annealer-immobile relational instances an adaptive
width-capped peel beats the plateau decisively (relational_4 tc <= 130,
relational_5 tc <= 25), while regressions and the expander control stay within a
bit of the base annealer.

## What changed vs 053

- **The win is the cap=infinity rung = full min-cost VE run to completion.**
  053 boxed VE to 6% of budget and discarded it unless it beat greedy. 058
  promotes VE to Rung 0 of the ladder and gives it a real box (<=25% of budget,
  capped by the 40% Phase-A deadline). On the relational instances VE fully
  eliminates in well under a second and reaches near-optimal tc; the bounded
  finite rungs (which 053 relied on) do NOT help there — see measurements.

- **Fixed the real blocker: the library greedy hangs at scale.** `optimize_code`
  with `GreedyMethod` is ~O(n·deg^2) and does not finish at 30k+ tensors (it ran
  >60s on relational_4 and never returned). TreeSA's default `Greedy` initializer
  calls the same routine, so full-graph TreeSA also hangs at scale. 053 ran
  greedy first, unconditionally — which is why the code never even reached its VE
  seed on the relational instances. 058:
    - runs the library greedy only for `n <= 6000`;
    - at scale uses the VE rung as the first always-valid tree (a proper
      elimination tree is cheap to build AND score; a chain over an un-eliminated
      core is O(n·m) and was itself too slow — an intermediate peel-construct
      floor is what made an early build of this attempt hang);
    - switches TreeSA to the `Random` initializer for any graph > 4000 tensors.

- **Adaptive cap ladder.** Probe a geometric cap ladder {6, 10, 16, 26, 42} to
  find w0 = the smallest cap already peeling >=50% of tensors, then evaluate
  {0.8·w0, w0, 1.25·w0}. Each peel is a few hundred ms even at 70k (no quadratic
  spots; the 053 heap+lazy peel already scales — measured 26-232 ms at 70k).

- **Mobility gate (the operative form of the hypothesis).** A rung's residual is
  annealed only when it is small enough that the annealer can actually move it
  (<=3000 tensors). On the relational_4 core the min-cost peel cannot shrink
  below ~10,400 tensors (no cheap eliminations remain past the periphery), and a
  10,400-tensor residual anneals to ~203 with random init even at 30s — as stuck
  as the full graph — so those rungs are recorded but skipped, and VE owns the
  instance. Where the peel DOES reach a small core (relational_5 -> 500), the
  rung is annealed.

- **Budget split.** Phase A (ladder) <= 40% of budget. Phase B gives the rest to
  the winning arm (residual-continue if a peel rung produced the best tree, else
  the full-graph fallback), with the full-graph TreeSA doubling run racing —
  GATED to `n <= 8000`, because above that a single uninterruptible niters round
  overruns the budget by minutes and the annealer is immobile anyway. On the
  large relational instances the process therefore terminates as soon as VE has
  won (5-36s of the 90-120s budget) rather than burning budget it cannot use.

- Non-binary cardinalities: the cost cap uses summed log2 dims (`set_cost` over
  `log2[label]`), unchanged from 053; verified on linkage (cards 1-5).

## Measurements (this Mac, with concurrent campaign load)

Peel scaling / stats (stderr `t_peel=..ms cap=.. peeled=../N residual=..`):

- relational_4 (30,400): every cap 6..42 peels 20,000/30,400 -> residual 10,400
  in ~208 ms (immobile core; all finite rungs skipped).
- relational_5 (70,000): cap 6,10 peel nothing; cap>=16 peels 69,500/70,000 ->
  residual 500 in ~200 ms (mobile; annealed).

Final tc (scorer.py tc/sc), all `MODE=auto`:

| instance         | tensors | budget | tc      | sc  | source        | target                    |
|------------------|---------|--------|---------|-----|---------------|---------------------------|
| relational_4     | 30,400  | 120 s  | 108.97  | 100 | ve(cap=inf)   | <= ~130 (plateau 202) OK  |
| relational_5     | 70,000  |  90 s  |  24.03  |  10 | ve(cap=inf)   | <= 25 (900 s SA: 24.3) OK |
| DBN_13           |    572  |  60 s  |  28.79  |  22 | ve(cap=inf)   | ~29 (053 winner) OK       |
| linkage_15       |  2,304  |  60 s  |  30.48  |  24 | residual      | ~31 OK                    |
| reg3_250 (expdr) |    250  |  60 s  |  40.02  |  34 | residual      | 40-47, not > 50 OK        |

- relational_4: VE reaches **108.97** at t=3.7s (optimum ~100), obliterating the
  202/900s plateau. Full-graph fallback correctly skipped (immobile) -> ends at
  5.4s.
- relational_5: VE **24.03** < the 24.3 that plain SA needs 900s for, and <<
  construction-only ~29. Mobile-residual rungs also reach 24.03.
- DBN / linkage within ~1 bit of 053 (linkage via the peel/residual arm, DBN via
  VE). reg3 expander: VE blows up (81, discarded), greedy/residual/full-graph
  win at 40.02 — matches plain annealing, guarding the falsified regime.

## Verdict

Confirmed on the new (annealer-immobile) regime. Honest attribution: on these
instances the decisive win is the **unbounded (cap=infinity) VE rung**, not the
bounded-peel residual anneal that 053 tested — the finite rungs are skipped on
relational_4 (immobile core) and merely match VE on relational_5. The bounded
finite rungs still carry linkage (2,304) and the reg3 expander, and the ladder is
the mechanism that selects the right rung per instance. The two real enablers
were (a) recognising that the peel-to-completion order IS full VE and letting it
run, and (b) removing the library-greedy hang that had prevented the code from
ever reaching VE at scale.

## Deviations from spec

- The task framed the win as "peel to the width-~100 core and anneal it". Measured
  reality: the min-cost peel cannot shrink relational_4 below ~10,400 tensors, and
  annealing that residual stays at ~203 (as immobile as the full graph). The
  target is instead hit by the ladder's top rung, full VE, which reaches 108.97 in
  <1s. Implemented as specified (adaptive ladder + racing fallback + mobility
  gate); reporting that the effective rung is cap=infinity.
- Full-graph fallback is GATED to n<=8000 rather than "always racing": at 30k-70k
  a single uninterruptible TreeSA round overruns the budget by minutes and cannot
  beat VE, so racing it there is pure downside. Protection is preserved exactly
  where 053 was falsified (n~4k) and on the expander control.
