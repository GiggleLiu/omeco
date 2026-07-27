#!/usr/bin/env python3
"""Verify the width-29 lower-bound certificate for tw(L(sycamore_m20)).

Chain: (1) the 55 branch sets in sycamore_m20.mnr (certificate width 29)
are pairwise disjoint and each induces a connected subgraph of the line
graph, so contracting them yields a minor H; (2) H (55 vertices, 364
edges, sycamore_m20_minor29.gr) has exact treewidth 29 (minor29.td:
's td 20 30 55', largest bag 30, produced by twalgor ExactTW); therefore
(3) tw(L(G)) >= tw(H) = 29.

Usage: python3 verify_minor.py   (run from this directory; exits 0 iff OK)
"""
import re, sys

lines = open('sycamore_m20.mnr').read().splitlines()
idx = max(i for i, l in enumerate(lines) if l.startswith('certificate width 29'))
block = []
for l in lines[idx + 1:]:
    if l.startswith(('certificate', 'title', 'param', 'graph')):
        break
    block.append(l)
sets = []
for l in block:
    m = re.match(r'\s*(\d+)\s+(\d+)\s*\{([^}]*)\}', l)
    assert m, l
    vs = [int(x) for x in m.group(3).replace(',', ' ').split()]
    assert len(vs) == int(m.group(2)), l
    sets.append(vs)

adj = {}
for l in open('sycamore_m20_linegraph.gr'):
    if l.startswith('p'):
        continue
    if l.strip():
        a, b = map(int, l.split()); a -= 1; b -= 1
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

seen = set()
for s in sets:
    assert not (set(s) & seen), 'branch sets overlap'
    seen |= set(s)
    ss, stack, vis = set(s), [s[0]], {s[0]}
    while stack:
        v = stack.pop()
        for w in adj.get(v, ()):
            if w in ss and w not in vis:
                vis.add(w); stack.append(w)
    assert vis == ss, 'branch set not connected'

which = {v: i for i, s in enumerate(sets) for v in s}
medges = {(min(which[a], which[b]), max(which[a], which[b]))
          for a in adj for b in adj[a]
          if a < b and a in which and b in which and which[a] != which[b]}
ref = set()
for l in open('sycamore_m20_minor29.gr'):
    if l.startswith('p'):
        k, m = map(int, l.split()[2:])
    elif l.strip():
        a, b = map(int, l.split()); ref.add((min(a,b)-1, max(a,b)-1))
assert medges == ref and len(sets) == k, 'minor graph mismatch'

hdr = open('minor29.td').readline().split()
assert hdr[:2] == ['s', 'td'] and int(hdr[3]) == 30, 'expected width-29 decomposition'
print('OK: 55-set minor valid; exact tw(minor) = 29; hence tw(L(G)) >= 29')
