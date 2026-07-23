"""Isoperimetric-profile DP.

Given a per-size boundary function b[k] (k = 1..n), compute
    f(k) = 2^{b[k]} + min_{1<=j<=k//2} [ f(j) + f(k-j) ],   f(1) = 0
and return F = f(n).  LB(tc) = log2 F.

f(n) = min over all binary-tree shapes on n leaves of  sum_v 2^{b[size(v)]},
the minimum being over the internal-node size multiset any shape induces.
Using b[k] = (a lower bound on) the min boundary over all k-subsets makes
log2 f(n) a valid lower bound on tc whenever b[k] <= true min-boundary(k).
"""
import math
from fractions import Fraction


def dp_float(bvals):
    """bvals: list length n+1, bvals[k] = boundary value for size k (float ok).
    Returns (F, log2F, split_argmin) using float64 arithmetic."""
    n = len(bvals) - 1
    f = [0.0] * (n + 1)
    f[1] = 0.0
    for k in range(2, n + 1):
        best = math.inf
        for j in range(1, k // 2 + 1):
            c = f[j] + f[k - j]
            if c < best:
                best = c
        f[k] = 2.0 ** bvals[k] + best
    F = f[n]
    return F, math.log2(F), f


def dp_exact_int(bvals):
    """bvals integer per size. Exact big-integer DP. Returns (F_int, log2F)."""
    n = len(bvals) - 1
    f = [0] * (n + 1)
    for k in range(2, n + 1):
        best = None
        for j in range(1, k // 2 + 1):
            c = f[j] + f[k - j]
            if best is None or c < best:
                best = c
        f[k] = (1 << int(bvals[k])) + best
    F = f[n]
    # exact log2 of a big integer
    log2F = math.log2(F) if F.bit_length() < 1000 else _log2_bigint(F)
    return F, log2F, f


def _log2_bigint(x):
    # log2 of arbitrarily large int without overflow
    shift = x.bit_length() - 53
    if shift > 0:
        mant = x >> shift
        return math.log2(mant) + shift
    return math.log2(x)
