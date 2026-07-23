"""Method (b): certified lower bound on tc via balanced cuts of G.

CUT LEMMA (exact statement).  Let T be any contraction tree: a rooted binary
tree whose n=561 leaves are the tensors. Every internal node u contracts two
sub-results into one; let S_u be the set of leaf-tensors below u. The
intermediate tensor produced at u has one open leg per index of the network
with exactly one endpoint inside S_u (the network is CLOSED, iy=[], so an index
is open on the S_u-result iff it crosses the boundary of S_u). Its rank is thus
  r_u = |partial S_u| = # edges of G with one endpoint in S_u, one outside.
The contraction cost at u is the number of distinct indices in the union of the
two operands' legs, which INCLUDES every open leg of the result, so
  cost_u >= r_u = |partial S_u|            (all dims 2 => cost counts indices).
Hence  tc = log2( sum_u 2^{cost_u} ) >= max_u cost_u >= max_u |partial S_u|.

SEPARATOR FACT.  Any rooted binary tree with n leaves has an edge whose removal
puts between ceil(n/3) and floor(2n/3) leaves on the smaller side (walk down
from the root always into the heavier child; the leaf-count drops from >2n/3 to
<=1 by <=1/2 each step, so it lands in [n/3,2n/3]). Take u = child at that edge:
  n/3 <= |S_u| <= 2n/3.
Therefore  tc >= |partial S_u| >= min over BALANCED S (n/3<=|S|<=2n/3) of |partial S|
             =: B_bal(G).
So B_bal(G), the minimum 1/3-2/3 balanced edge-cut of G, is a certified LB on tc.

To CERTIFY tc >= X we need a LOWER bound on B_bal(G) (every balanced cut is big).
We use the spectral (Fiedler-Mohar) bound, which is fully rigorous:
  for any S, cut(S,Sbar) >= lambda2 * |S|*|Sbar| / n,
where lambda2 is the algebraic connectivity (2nd-smallest Laplacian eigenvalue).
Minimised over n/3<=|S|<=2n/3 (product smallest at the endpoints):
  B_bal(G) >= lambda2 * (n/3)*(2n/3)/n = lambda2 * 2n/9.
This is a certificate: lambda2 is a computed eigenvalue (verifiable), the
inequality is a theorem. (It is typically far from tight on long thin lattices.)

We ALSO report a heuristic balanced cut (METIS) = an ACHIEVABLE cut, i.e. an
UPPER bound on B_bal(G); combined with the temporal-cut construction it shows
B_bal(G) is ~53, but that is an upper bound on the separator, not a tc
certificate.
"""
import pathlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path(__file__).resolve().parent
edges = [tuple(map(int, l.split())) for l in open(HERE / "G.edgelist")]
n = 561
rows = [u for u, v in edges] + [v for u, v in edges]
cols = [v for u, v in edges] + [u for u, v in edges]
A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
deg = np.asarray(A.sum(1)).ravel()
L = sp.diags(deg) - A

# algebraic connectivity lambda2 (smallest nonzero Laplacian eigenvalue)
vals = spla.eigsh(L.astype(float), k=2, sigma=1e-8, which='LM',
                  return_eigenvectors=False)
vals = np.sort(vals)
lam2 = float(vals[1])
print(f"n={n}  lambda2(algebraic connectivity) = {lam2:.6f}")
bal_lo = lam2 * (2 * n / 9)      # min over 1/3-2/3 balanced cuts
half_lo = lam2 * n / 4           # bisection (|S|=n/2)
print(f"spectral LB on min 1/3-2/3 balanced cut  B_bal(G) >= {bal_lo:.3f}")
print(f"  -> certified tc >= {bal_lo:.3f}")
print(f"spectral LB on min bisection (|S|=n/2)          >= {half_lo:.3f}")

# ---- METIS heuristic balanced bisection: achievable cut (UPPER bnd on B_bal) ---
try:
    import pymetis
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    best = None
    for seed in range(8):
        ncut, membership = pymetis.part_graph(
            2, adjacency=adj, options=pymetis.Options(seed=seed))
        m = np.array(membership)
        # count cut edges
        c = sum(1 for u, v in edges if m[u] != m[v])
        sizes = (int((m == 0).sum()), int((m == 1).sum()))
        if best is None or c < best[0]:
            best = (c, sizes)
    print(f"METIS balanced bisection: cut={best[0]} sizes={best[1]}"
          f"  (achievable => B_bal(G) <= {best[0]})")
except Exception as e:
    print("pymetis failed:", e)
