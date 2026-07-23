# Certified lower bound on the contraction cost of `reg3_250`

**Attempt 021 — MEASUREMENT (no contraction tree produced, no records set, validator not run).**
**Date:** 2026-07-23

## Headline

- **Certified lower bound:** `tw(L(G)) >= 18`, hence **`tc >= 18`** for every contraction order.
- **Best known tc:** `39.95` log2-flops.
- **tc gap: 39.95 − LB(tc) = 39.95 − 18 = 21.95.**
- Certificate: a 32-vertex **minor** `H` of the line graph `L(G)` with `tw(H) = 18` (exact, two independent solvers). Treewidth is minor-monotone, so `tw(L(G)) >= 18`.

The bound is *fully certified and independently checkable*. It is also *weak*: the true `tw(L(G))` is `<= 34` (a constructive upper bound — real contraction trees achieve max-rank / space-complexity `sc = 34`). Certifying high treewidth on this expander-like line graph is the bottleneck, not the theory (see "Honest assessment").

---

## The instance

`research/benchmark/targets/reg3_250.json`: a graph `G` with **250 vertices** (tensors) and **372 edges** (indices), all bond dimension 2, closed network (`iy = []`).
- Every index appears on exactly **2** tensors (`G` is a genuine graph, not a hypergraph).
- Arity distribution: 247 tensors of degree 3, 3 tensors of degree 1 (`247*3 + 3*1 = 744 = 2*372`).

**Line graph `L(G)`** (Markov–Shi construction): vertices = the 372 indices; two indices are adjacent iff they co-occur on some tensor (each degree-3 tensor contributes a triangle).
- `|V(L)| = 372`, `|E(L)| = 741`, connected.
- Almost 4-regular: 369 vertices of degree 4, 3 of degree 2 (the edges incident to the degree-1 vertices of `G`).

---

## Theory: from treewidth to a `tc` lower bound (exact inequality)

All bond dimensions are 2, so every log2-size equals an index count.

**Setup.** A contraction tree is a binary tree whose 250 leaves are the input tensors; each internal node contracts children `A, B` into parent `C`. For any node/tensor `T`, let `ind(T)` be its index set (= the boundary between `T`'s leaves and the rest of the closed network), and `rank(T) = |ind(T)|`. The cost of one contraction `(A,B) -> C` is
`w = |ind(A) ∪ ind(B)| = log2(FLOPs of that pairwise contraction)`.
Define `sc(tree) = max over nodes of rank(C)` (max intermediate rank) and
`tc(tree) = log2( Σ over the 249 internal nodes of 2^{w_node} )`.

**Step 1 — union dominates each rank.** Every index of `C` appears on `A` or `B`, so `ind(C) ⊆ ind(A) ∪ ind(B)`, giving `w_node >= rank(C)` (and trivially `>= rank(A), rank(B)`).

**Step 2 — biggest contraction dominates space.** `max_node w_node >= max_node rank(C) = sc(tree)` (every intermediate tensor is some node's output; leaves have rank <= 3).

**Step 3 — Markov–Shi.** The contraction complexity `cc` (min over orders of the max intermediate rank) equals `tw(L(G))` exactly, with the standard convention `tw = (max bag size) − 1`. Hence **for every tree, `sc(tree) >= tw(L(G))`.**

**Step 4 — log-sum dominates its max term.** `tc(tree) = log2(Σ 2^{w_node}) >= log2(max_node 2^{w_node}) = max_node w_node`.

**Chain.** For **every** contraction tree:
```
tc  >=  max_node w_node  >=  sc(tree)  >=  tw(L(G)).
```
Therefore **`tc >= tw(L(G))`** with exact additive constant **0** (this is the rigorous, fully-certified inequality used for the headline).

**Refinement `tc >= tw + 1` (very likely, not proven for all trees).** At the node that first produces the max-rank tensor `C*` (`rank = sc`), `w = rank(C*) + (#indices summed at that step)`. Unless `C*` is a pure outer product (`ind(A) ∩ ind(B) = ∅` with nothing eliminated), at least one index is summed and `w >= sc + 1 >= tw + 1`. On this closed, connected, ~4-regular line graph outer products never help and do not appear in frontier trees, so the operative refinement is `tc >= tw + 1`. We report the safe `tc >= tw` for certification and note `+1` as the expected sharpening.

---

## The certificate and how to verify it

`twalgor/tw`'s `UpLow` (improved lower-bound machinery) emits, in `certs/L.mnr`, a sequence of **minors** of `L(G)` of increasing treewidth. The strongest is:

```
certificate width 18 n 32 time 105795
```
i.e. a graph `H` obtained from `L(G)` by contracting 32 pairwise-disjoint, each-connected vertex sets (a genuine minor). Because treewidth is **minor-monotone**, `tw(L(G)) >= tw(H)`.

Verification is a three-link chain, each link independently checkable:

1. **`H` is a valid minor of `L(G)`.** `certs/reproduce.py` rebuilds `L(G)` from the JSON, reads `certs/L.mnr`, and checks that the 32 contraction sets are pairwise disjoint and each induces a connected subgraph of `L(G)`. Output: `minor validity ... : True`. It then builds `H` (`|V|=32, |E|=154`) and re-emits `certs/H.gr`.
2. **`tw(H) = 18` (upper side).** `certs/H_exact.td` is a width-18 tree decomposition of `H`; `td-validate` reports `valid`. So `tw(H) <= 18`.
3. **`tw(H) = 18` (lower side / exactness).** Two independent PACE-2017 exact solvers — `TCS-Meiji/PACE2017-TrackA` (`tw.exact`, positive-instance-driven DP) and `twalgor/tw` (`ExactTW`) — both return `tw(H) = 18`, i.e. no width-17 decomposition exists.

Chain: `tw(L(G)) >= tw(H) = 18`  ⟹  `tc >= tw(L(G)) >= 18`.

---

## Reproduce

```bash
# 0. From the worktree root. Requires python3+networkx and a JDK (openjdk 21+).
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"   # or your JDK

# 1. Build L(G) and the certified-LB solver, run it (LB minors -> L.mnr, UB td -> L.td)
python3 - <<'PY'   # builds scratch/L.gr ; see certs/reproduce.py for the exact construction
import json,networkx as nx
d=json.load(open('research/benchmark/targets/reg3_250.json'));L=nx.Graph()
for x in d['ixs']:
    xs=list(set(x))
    for i in range(len(xs)):
        for j in range(i+1,len(xs)): L.add_edge(xs[i],xs[j])
nodes=sorted(L.nodes()); m={n:i+1 for i,n in enumerate(nodes)}
open('scratch/L.gr','w').write(f'p tw {L.number_of_nodes()} {L.number_of_edges()}\n'+
    ''.join(f'{m[u]} {m[v]}\n' for u,v in L.edges()))
PY
git clone https://github.com/twalgor/tw.git scratch/tw           # commit 17fcb3cd
javac -d scratch/tw/out $(find scratch/tw/src -name '*.java')
java -Xmx8g -cp scratch/tw/out io.github.twalgor.main.UpLow \
     scratch/L.gr scratch/L.td scratch/L.mnr      # LB climbs 14->18 (plateaus ~2 min); Ctrl-C

# 2. Verify the minor certificate end-to-end (rebuilds L(G), checks minor validity, builds H)
python3 certs/reproduce.py

# 3. Confirm tw(H)=18 exactly with two independent solvers + td-validate
git clone https://github.com/TCS-Meiji/PACE2017-TrackA.git scratch/PACE  # commit 72783906
javac scratch/PACE/tw/exact/*.java
java -cp scratch/PACE tw.exact.MainDecomposer < certs/H.gr            # -> width 18
java -cp scratch/tw/out io.github.twalgor.main.ExactTW certs/H.gr /tmp/H.td   # -> width 18
(cd scratch/PACE/td-validate-master && make) && \
     scratch/PACE/td-validate-master/td-validate certs/H.gr certs/H_exact.td   # -> valid
```

**Repo commits used:**
- `twalgor/tw` @ `17fcb3cd59e7be63bdab312fcaa16ca98866cf13` (UpLow improved bounds; ExactTW exact solver)
- `TCS-Meiji/PACE2017-TrackA` @ `7278390fe81191f238206b822fa2941d068a1214` (PID exact solver `tw.exact`; `td-validate`)

**Certificate files (checked in under `certs/`):** `L.gr` (the line graph), `L.mnr` (minor certificates), `H.gr` (the width-18 minor), `H_exact.td` (its optimal decomposition), `reproduce.py`, `bounds.py`, `verify_minor.py`.

---

## Cross-checks (upper bounds and cheap lower bounds)

| Quantity | Value | Method | Certified? |
|---|---|---|---|
| `tw(L(G))` lower bound | **18** | minor `H` + exact `tw(H)` | **yes (this attempt)** |
| `tw(L(G))` upper bound | `<= 34` | real contraction trees reach `sc = 34` (`cc = tw`) | yes, constructive (external) |
| degeneracy / core LB | 3 | `bounds.py` | yes but useless |
| MMW (minor-min-width, least-c) LB | 10 | `bounds.py` | yes but weak |
| min-fill / min-degree heuristic UB | 58 / 72 | `bounds.py` | UB only |
| PACE-2017 heuristic UB | ~50 after ~2 min (still descending) | `tw.heuristic` | UB only, under-converged |

So `18 <= tw(L(G)) <= 34`. The interval is wide because both *certifying* a high LB and *heuristically reaching* the ~34 UB are hard on this instance within a short budget.

---

## Honest assessment (hypothesis verdict)

**Hypothesis** ("certified treewidth LBs place the 39.95 frontier within a small gap of optimal"): **not supported by the certified number**, but for a methodological rather than a theoretical reason.

1. **Certification is the bottleneck.** The true `tw(L(G))` is `<= 34` and almost certainly close to it, but certifying treewidth from below is co-NP-hard and, on this ~4-regular expander line graph, the state-of-the-art improved-bound solver plateaus at **18** (climbs `14->15->16->17->18` in ~2 min, then stalls; exact solvers do not finish for `n=372, tw~34`). So the certified gap is `39.95 − 18 = 21.95`.

2. **Even a perfect treewidth certificate leaves ~5–6 bits.** This method bounds `tc` by the *single largest* contraction, `tc >= tw` (`>= tw+1` in practice). With the UB `tw <= 34`, the best this family of bounds could ever certify is `tc >= 34` (or `35`), i.e. a floor gap of `39.95 − 34 = 5.95` (`4.95` with the `+1`). The remaining ~5 bits are the **log-sum overhead**: `tc` aggregates 249 contractions, and on a 3-regular expander a constant fraction of contractions sit within 1–2 bits of the max, adding ~`log2(#near-max terms)` ≈ 5 bits. Closing that final gap needs a genuinely different argument (method (d): a counting/expander lower bound on how many contractions must be near-maximal), which we did **not** attempt rigorously here — it is left as the clear next step.

**Bottom line.** Rigorously, for `reg3_250`: `tc >= 18` (certificate attached, gap 21.95). Structurally, the frontier `tc = 39.95` decomposes as `~34` (space/treewidth, UB-certified) `+ ~5–6` (log-sum overhead), so a *believed* — but not from-below-certified — gap to optimal `tc` is only ~5–6 bits. The 39.95 frontier is very likely near-optimal; proving it requires a treewidth LB much stronger than any solver delivered in-budget, plus a log-sum lifting argument.
