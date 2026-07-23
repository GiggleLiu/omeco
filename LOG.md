# LOG — autoresearch attempt-021

- **attempt:** 021
- **date:** 2026-07-23
- **kind:** draft (MEASUREMENT — certified lower bound, not an optimizer)
- **parent:** none
- **instance:** research/benchmark/targets/reg3_250.json (random 3-regular graph, 250 tensors, 372 indices, all bond dim 2, closed network iy=[])
- **hypothesis:** certified lower bounds via treewidth of the line graph place the 39.95 frontier within a small gap of optimal
- **expected evidence:** an LB number with a verifiable certificate; gap = 39.95 − LB(tc)

## Notes / progress

### Instance
- `reg3_250.json`: G = 250 vertices, 372 edges, all bond dim 2, closed (iy=[]). 247 degree-3 + 3 degree-1 tensors. Every index on exactly 2 tensors.
- Line graph L(G): |V|=372, |E|=741, connected, ~4-regular (369 deg-4, 3 deg-2). Exported to `certs/L.gr`.

### Theory (exact inequality, derived in LOWER_BOUND.md)
- Markov–Shi: contraction complexity cc = min-over-trees max-intermediate-rank = tw(L(G)) exactly (tw = maxbag−1).
- For every binary contraction tree: `tc >= max_node w_node >= sc(tree) >= tw(L(G))`, where w_node = log2 FLOPs of one contraction = |ind(A)∪ind(B)|.
- **Certified constant: tc >= tw(L(G))** (additive 0). Refinement `tc >= tw+1` very likely (holds unless a top-rank outer product occurs; none in frontier trees) but not proven for all trees.

### Certified result
- Ran `twalgor/tw` UpLow (commit 17fcb3cd) on L(G). LB minors climb 14→15→16→17→18, then plateau (~2 min). Best certificate: a 32-vertex minor H (width 18) in `certs/L.mnr`.
- Verified H is a genuine minor of L(G) (disjoint, each-connected contraction sets) via `certs/reproduce.py`. H: |V|=32, |E|=154.
- tw(H) = 18 EXACT, confirmed by TWO independent PACE-2017 exact solvers (`PACE2017-TrackA` tw.exact, commit 72783906; and `tw` ExactTW). td-validate confirms the width-18 decomposition `certs/H_exact.td` is valid.
- Minor-monotonicity ⇒ **tw(L(G)) ≥ 18 ⇒ tc ≥ 18**.

### Numbers
- Certified: `tc ≥ 18`. Best known tc = 39.95. **Gap = 39.95 − 18 = 21.95.**
- Bracket on true tw: `18 ≤ tw(L(G)) ≤ 34` (UB from real sc-34 trees; heuristic solvers under-converge to ~50 in-budget).
- Cheap LBs (cross-check, all weak/useless): degeneracy 3, MMW 10.

### Verdict
- Hypothesis NOT supported by the certified number (gap 21.95), for a METHODOLOGICAL reason: certifying high treewidth on this expander line graph is co-NP-hard and solvers plateau at 18 (exact won't finish for n=372, tw~34).
- Even a perfect tw certificate (tw≤34) floors this method at tc≥34, gap 5.95: the residual ~5–6 bits is log-sum overhead (249 contractions, many near-max on a 3-regular expander), needing a separate counting argument (method d, not attempted rigorously).
- Structurally 39.95 ≈ 34 (treewidth, UB-certified) + ~5–6 (log-sum): frontier very likely near-optimal, but not certified from below.

### Deliverables
- `LOWER_BOUND.md` — full report, derivation, certificate, reproduce commands, commits.
- `certs/` — L.gr, L.mnr, H.gr, H_exact.td, reproduce.py (self-contained verifier), bounds.py, verify_minor.py.
- NOTE: no contraction tree produced, no records set, validator intentionally NOT run (scored runs meaningless for a measurement attempt).
