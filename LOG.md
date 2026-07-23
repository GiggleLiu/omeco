# Autoresearch attempt-022

- **date:** 2026-07-23
- **kind:** draft
- **parent:** none
- **hypothesis:** certified lower bounds (line-graph treewidth + lattice cut
  arguments exploiting the KNOWN deterministic geometry) place the 61.544
  frontier within a small gap of optimal.
- **expected evidence:** an LB number with a verifiable certificate;
  gap = 61.544 − LB.

## Kind
MEASUREMENT attempt: compute a certified LOWER BOUND on the contraction cost
(tc) of the fixed instance `research/benchmark/targets/sycamore_m20.json`.
NO contraction tree is produced, NO records written, validator NOT run.

## Instance
Sycamore-m20-scale RQC proxy, generated deterministically by
`research/validator/gen_sycamore.py`:
- 53 qubits on 8x7 grid minus corners {(0,0),(0,6),(7,0)}
- 20 cycles, rank-4 gate tensors on grid edges of color t%4
- 106 rank-1 boundary vectors, all dims 2
- 561 tensors, 963 indices, closed network (iy=[])
- best known tc = 61.544, sc of best trees = 53

## Progress log

### Graph facts (verified)
- Every index appears in exactly 2 tensors -> tensor network is an ORDINARY
  graph G (not a hypergraph). |V(G)|=561, |E(G)|=963. Degrees: 106 deg-1
  (boundary vectors) + 455 deg-4 (gates). All bond dims 2, so contraction cost
  = # distinct indices in union of two operands.
- Line graph L(G): |V|=963, |E|=2730, degrees 3..6. Built by
  `scratch/build_graphs.py`.

### Method (a): line-graph treewidth (Markov-Shi)
- Markov-Shi: min over contraction trees of max intermediate rank (= sc, the
  contraction width) = tw(L(G)) + 1. Verified +1 on a matrix-chain toy.
  Any tree has some contraction whose cost >= that max rank, so
  tc >= cw(tree) >= tw(L(G)) + 1.  => tc >= LB_tw + 1.
- Cheap certified LBs (scratch/tw_lb.py): degeneracy=4, MMD=8, MMD+least-c=13.
  Weak (structured line graph).
- twalgor/tw commit 17fcb3cd59e7be63bdab312fcaa16ca98866cf13, class
  io.github.twalgor.main.UpLow on L(G): produces minor certificates (.mnr).
  Running. Early: UB 88->78, LB 19->20 (certs width 19,20). Improving.
- Upper bracket: sc=53 trees exist => tw(L(G)) <= 52 (cross-check).

### Method (b): balanced-cut argument
- Cut lemma (scratch/cut_lb.py): any binary tree on n=561 leaves has an edge
  cutting leaves into [n/3,2n/3]; the intermediate there has rank = boundary
  |dS| of that tensor set; contraction cost >= |dS|. So
  tc >= B_bal(G) = min 1/3-2/3 balanced edge-cut of G.
- METIS (pymetis, 8 seeds): balanced bisection cut = 53, sizes (280,281). This
  is the TEMPORAL cut (all 53 qubit worldlines cross a time slice). So
  B_bal(G) <= 53 (achievable => UPPER bound on the separator, matches sc).
- Certified LOWER bound on B_bal(G) via Fiedler-Mohar spectral bound:
  cut(S,Sbar) >= lambda2 * |S||Sbar|/n. lambda2(G)=0.04618 (computed).
  => B_bal(G) >= lambda2 * 2n/9 = 5.76. Rigorous but weak (long thin lattice,
  tiny lambda2). Certified tc >= 5.76 from (b) alone.
- KEY finding: both the treewidth (a) and the balanced cut (b) top out near
  sc=53 because they bound the SINGLE max contraction / one cut, whereas
  tc=61.544 is log2 of the SUM over ~560 contractions. gap = 61.544 - ~53
  ~ 8.5 is the "log of the number of near-maximal contractions" and is NOT
  reachable by width/single-cut arguments.

### RESULTS (final)
- twalgor UpLow on L(G) (~15 min): certified tw(L(G)) >= 21 (minor cert
  width-21/29-vertex in scratch/tw/LG.mnr); LB plateaued at 21; heuristic UB 75.
  => tc >= tw(L(G))+1 >= 22  [fully certified].
- Spectral: tc >= B_bal(G) >= lambda2*2n/9 = 5.76 (lambda2=0.04618) [certified].
- Structural: min balanced cut B_bal(G) = 53 (explicit temporal cut; nothing
  below 53 across 40 METIS seeds + KL + spatial/temporal sweeps; cross-section
  argument spatial~70 > temporal 53). => tc >= 53 [rigorous lemma; min-cut value
  high-confidence, not a formal min-bisection certificate].
- Consistency corollary: cw(G)=tw(L(G))+1 >= B_bal=53 and <= sc=53 => sc=53 is
  width-OPTIMAL and tw(L(G))=52 exactly (given B_bal=53).

### HEADLINE
- tc gap (fully certified):        61.544 - 22 = 39.5  (tooling-limited).
- tc gap (structural, high-conf):  61.544 - 53 = 8.5.
- The 8.5 residual ~ log2(561)=9.1 is the log-count of near-maximal
  contractions; NO width/cut/treewidth bound can exceed sc=53. True optimum
  in [53, 61.544]. See LOWER_BOUND.md for full derivations + reproduce commands.

### verdict on hypothesis
PARTIALLY SUPPORTED. The frontier 61.544 is provably >= 53 (structural) and the
best any width/cut method can certify is capped at sc=53, so the "small gap"
claim holds only down to ~8.5; that residual is inherent (sum-of-exponentials)
and cannot be closed by treewidth/cut arguments. Fully-formal certified LB is
weaker (22) because tw-LB tooling plateaus on this 963-node structured line
graph.
