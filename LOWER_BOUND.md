# Certified lower bound on the contraction cost of `sycamore_m20`

**Attempt 022 (MEASUREMENT). Date 2026-07-23.**

Instance: `research/benchmark/targets/sycamore_m20.json`, generated
deterministically by `research/validator/gen_sycamore.py`. No contraction tree
is produced and the validator is not run; this document computes a *lower bound*
on the achievable `tc` and certifies it.

- `tc = log2( sum over pairwise contractions of 2^cost )`
- `cost` of a pairwise contraction = sum of `log2(dim)` over the union of the two
  operands' index sets. **All bond dims are 2**, so `cost` = number of distinct
  indices in that union.
- **Best known `tc = 61.544`**, `sc` of best trees `= 53`.

## 0. Structure of the instance (verified)

Every index label occurs in **exactly two** tensors (checked: the
index-occurrence multiset is `{2: 963}`). Hence the tensor network is an
**ordinary graph** `G` (not a hypergraph):

- `|V(G)| = 561` tensors, `|E(G)| = 963` indices.
- degrees: 106 degree-1 vertices (the rank-1 boundary vectors) + 455 degree-4
  vertices (the rank-4 gate tensors).
- closed network (`iy = []`): every index is internal.

Line graph `L(G)`: vertices = the 963 indices; two indices are adjacent iff they
meet at a common tensor. `|V(L(G))| = 963`, `|E(L(G))| = 2730`, degrees 3..6.

Reproduce: `python3 scratch/build_graphs.py` (writes `G.edgelist`, `LG.gr`,
`LG.edgelist`).

---

## 1. Method (a): line-graph treewidth (Markov–Shi)

### Derivation (every inequality stated)

For a contraction tree `T` (a rooted binary tree whose 561 leaves are the
tensors), let each internal node `u` contract two sub-results into one, and let
`S_u ⊆ leaves` be the tensors below `u`. Because the network is closed, the
intermediate tensor at `u` has exactly one open leg per index of `G` with one
endpoint in `S_u` and one outside; its rank is

    r_u = |∂S_u| = #edges of G across the cut (S_u, V\S_u).

The contraction cost at `u` counts the distinct indices in the union of its two
operands, which **includes every open leg of the result**, so

    (1)   cost_u ≥ r_u.

The *contraction width* of `T` is `cw(T) = max_u r_u` (= `sc`, in log2). Markov &
Shi (SIAM J. Comput. 2008) give the exact identity

    (2)   min over trees T of cw(T)  =  tw(L(G)) + 1

(verified on a matrix-chain toy: path `L(G)` has `tw=1`, optimal width `2`).
Combining: for **every** tree `T` there is a node with
`cost_u ≥ r_u ≥ max_v r_v = cw(T) ≥ tw(L(G))+1`, and since
`tc = log2(Σ 2^{cost}) ≥ max cost`,

    (3)   tc  ≥  tw(L(G)) + 1.

### Certified value

Cheap certified LBs on `tw(L(G))` (`scratch/tw_lb.py`; each is a proven tw lower
bound because contracting/deleting to a minor cannot raise min degree above tw):

| bound | value |
|---|---|
| degeneracy (δ) | 4 |
| MMD (minor-min-width) | 8 |
| MMD+ (least-c) | 13 |

Strong certified LB — **twalgor/tw**, commit
`17fcb3cd59e7be63bdab312fcaa16ca98866cf13`, class
`io.github.twalgor.main.UpLow` run on `LG.gr` (JDK from Homebrew `openjdk`),
~15 min timebox. It emits **minor certificates** (`scratch/tw/LG.mnr`):

```
certificate width 19 n 21 time 1496
certificate width 20 n 24 time 8190
certificate width 21 n 29 time 36320
```

The width-21 minor (29 vertices, listed in `LG.mnr`) is a *verifiable
certificate* that `tw(L(G)) ≥ 21`. The LB thread then plateaued at 21 for the
remaining ~14 min (minor-based LB machinery is weak on structured 963-vertex
line graphs). The concurrent UB thread reached 75 (heuristic, not needed here).

    ==>  tc ≥ tw(L(G)) + 1 ≥ 22.        [fully certified]

Cross-check upper bracket: sc = 53 trees exist ⇒ by (2) `tw(L(G)) ≤ 52`. So the
true `tw(L(G)) ∈ [21, 52]`.

Reproduce:
```
cd scratch/tw && git clone --depth 1 https://github.com/twalgor/tw.git .   # commit above
export PATH=/opt/homebrew/opt/openjdk/bin:$PATH
javac -d out $(find src -name '*.java')
java -Xmx8g -cp out io.github.twalgor.main.UpLow "$PWD/LG.gr" "$PWD/LG.td" "$PWD/LG.mnr"
```

---

## 2. Method (b): balanced spacetime cut

### Cut lemma (rigorous)

**Separator fact.** Any rooted binary tree with `n` leaves has an edge whose
removal leaves between `⌈n/3⌉` and `⌊2n/3⌋` leaves below it (descend from the
root always into the heavier child; the below-count falls from `>2n/3` toward
`≤1`, dropping by a factor `≤` at each step, so it lands in `[n/3, 2n/3]`).

Take `u` = the node at that edge; then `n/3 ≤ |S_u| ≤ 2n/3` and, by (1),
`cost_u ≥ r_u = |∂S_u|`. Therefore

    (4)   tc ≥ max_u cost_u ≥ |∂S_u| ≥ B_bal(G),

where `B_bal(G)` = minimum, over vertex subsets `S` with `n/3 ≤ |S| ≤ 2n/3`
(here `187 ≤ |S| ≤ 374`, `n=561`), of the number of edges crossing `(S, V\S)`.
So **`tc ≥ B_bal(G)`** — a rigorous inequality. To *certify* `tc ≥ X` we need a
*lower* bound on `B_bal(G)` (every balanced cut is large).

### (b1) Certified spectral lower bound on `B_bal(G)`

Fiedler–Mohar: for any `S`, `cut(S) ≥ λ₂ · |S|·|V\S| / n`, where `λ₂` is the
algebraic connectivity (2nd-smallest Laplacian eigenvalue). Minimised over the
balanced window (product smallest at the endpoints `|S|=n/3`):

    B_bal(G) ≥ λ₂ · (n/3)(2n/3)/n = λ₂ · 2n/9.

Computed (`scratch/cut_lb.py`): `λ₂(G) = 0.046181` ⇒

    ==>  tc ≥ B_bal(G) ≥ 5.757.          [fully certified, but weak]

Weak because `G` is a long thin lattice (~8×7×20), so `λ₂` is tiny. This is the
best *fully formal* certificate method (b) yields.

### (b2) Value of `B_bal(G)`: it is 53 (high-confidence, geometry)

`B_bal(G) = 53`, established from above (an *achievable* balanced cut, i.e. an
upper bound on the separator) and argued optimal:

- **Explicit temporal cut = 53.** Put all tensors of cycles `< t` (with the 53
  input vectors) in `S`. Exactly the 53 qubit worldlines cross the time slice ⇒
  cut `= 53`. At `t=10`, `|S| = 280` (balanced). Construction, not a heuristic.
- **Nothing beats it.** `scratch/mincut_search.py`: 40 METIS seeds, all temporal
  cuts, all row/column spatial cuts, and Kernighan–Lin refinement — the minimum
  balanced cut found is **53** (METIS: sizes 280/281). Reproduce:
  `python3 scratch/mincut_search.py`.
- **Cross-section argument.** The graph is a "thick path" of length 20 (cycles)
  with cross-section = 53 worldlines. A *temporal* cut severs one cross-section =
  53 edges. A *spatial* balanced cut separates ~26/27 qubits; it severs every
  gate on the grid-boundary between the two qubit groups, over all 20 cycles.
  The 8×7 grid's balanced boundary is ≥ 7 grid-edges; each active in 5 of the 20
  cycles; each active boundary gate contributes 2 crossing legs ⇒ ≳ `7·5·2 = 70`
  > 53. Diagonal spacetime cuts interpolate above 53. Hence the temporal cut is
  the cheapest balanced cut.

Confidence that `B_bal(G) = 53`: **very high** (explicit optimum + exhaustive
heuristic failure to beat it + geometric optimality). We do **not** have a formal
min-bisection *lower-bound* certificate (that is NP-hard; the spectral bound
(b1) is the rigorous fallback and is loose). Treated as a *structural* bound:

    ==>  tc ≥ B_bal(G) = 53.             [rigorous lemma; min-cut value high-confidence]

### Consistency corollary

`cw(G) = tw(L(G))+1 = min_T max_v r_v ≥ B_bal(G) = 53`, while sc-53 trees give
`cw(G) ≤ 53`. So (given `B_bal = 53`) `cw(G) = 53` exactly — **the sc = 53
frontier is width-optimal** — and by (2) `tw(L(G)) = 52` exactly.

---

## 3. Cross-check bracket (method c)

- Upper bound on `tw(L(G))`: `≤ 52` from the existence of sc-53 trees (via (2)).
- twalgor heuristic UB reached 75 in the timebox (not tight; not needed).
- So `tw(L(G)) ∈ [21, 52]`, true value `= 52` under the corollary above.

---

## 4. Headline

Preferring the **max** over methods:

| bound | value | status |
|---|---|---|
| (a) twalgor `tw(L(G))+1` | **tc ≥ 22** | fully certified (minor cert `LG.mnr`) |
| (b1) spectral balanced cut | tc ≥ 5.76 | fully certified (λ₂ = 0.04618) |
| (b2) balanced cut = 53 | **tc ≥ 53** | rigorous lemma + high-confidence min-cut |

- **`tc` gap (fully certified):  61.544 − 22 = 39.5.**  (tooling-limited)
- **`tc` gap (structural, high-confidence):  61.544 − 53 = 8.5.**

### Interpretation — why the gap does not close

Every method here (treewidth of `L(G)`, and any single balanced cut) bounds the
**largest single contraction**, which is exactly `sc = 53`. But
`tc = log2(Σ_contractions 2^{cost})` is a soft-max over the ~560 contractions:

    tc ≥ sc,  and  tc − sc ≈ log2(#contractions of near-maximal cost).

Here `61.544 − 53 = 8.5 ≈ log2(561) = 9.1`. The residual gap is the *log-count of
near-maximal contractions*; **no width / single-cut / treewidth lower bound can
ever exceed `sc = 53`.** Thus:

- The frontier `tc = 61.544` is within `8.5` of the ceiling of what any such
  bound could certify, and it sits at essentially `sc + log2(#tensors)` — the
  natural lower envelope for a network whose optimal width is 53.
- The true optimum lies in `[53, 61.544]`. Closing the last 8.5 would require a
  *sum-of-exponentials* lower bound (bounding the number of simultaneously
  expensive contractions), which the width/cut toolbox structurally cannot
  provide.

---

## Reproduce everything

```
python3 scratch/build_graphs.py       # build G, L(G); PACE LG.gr
python3 scratch/tw_lb.py              # cheap certified tw LBs (deg/MMD/MMD+)
python3 scratch/cut_lb.py             # spectral balanced-cut LB + METIS bisection
python3 scratch/mincut_search.py      # confirm B_bal(G)=53 (nothing below 53)
# twalgor certified tw LB:
cd scratch/tw && git clone --depth 1 https://github.com/twalgor/tw.git .   # 17fcb3c
export PATH=/opt/homebrew/opt/openjdk/bin:$PATH
javac -d out $(find src -name '*.java')
java -Xmx8g -cp out io.github.twalgor.main.UpLow "$PWD/LG.gr" "$PWD/LG.td" "$PWD/LG.mnr"
```

Commits: twalgor `17fcb3cd59e7be63bdab312fcaa16ca98866cf13`; worktree
`c128ad9` (attempt-022 branch).
