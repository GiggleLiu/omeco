import sys, pickle, networkx as nx
# Rebuild L(G) exactly as exported (idmap: original label -> 1..N gr id). mnr uses 0-based
# over the SAME vertex set as the .gr, i.e. gr-id-1 (0-based). We must map minor's original
# vertices (0..371, = gr_id-1) back to L nodes.
L,idmap = pickle.load(open('scratch/L.pkl','rb'))
inv = {v-1:k for k,v in idmap.items()}   # 0-based gr id -> original L node label
mnr = open(sys.argv[1]).read().splitlines()
certs=[]
i=0
while i < len(mnr):
    line=mnr[i]
    if line.startswith('certificate'):
        parts=line.split()
        w=int(parts[2]); n=int(parts[4])
        bags=[]
        for j in range(1,n+1):
            row=mnr[i+j]
            # format: <vnum> <count> {a b c ...}
            lb=row.index('{'); rb=row.index('}')
            verts=[int(x) for x in row[lb+1:rb].replace(',',' ').split()]
            bags.append(verts)
        certs.append((w,n,bags))
        i+=n+1
    else:
        i+=1
w,n,bags = certs[-1]  # latest = highest width
print(f'Latest certificate: claimed width {w}, {n} minor-vertices')
# Validate: disjoint, within range, each induces connected subgraph in L
seen=set(); ok=True
for b in bags:
    for v in b:
        if v in seen: print('OVERLAP',v); ok=False
        seen.add(v)
    lnodes=[inv[v] for v in b]
    sub=L.subgraph(lnodes)
    if not nx.is_connected(sub):
        print('NOT CONNECTED bag', b); ok=False
print('all minor-vertices disjoint & connected in L(G):', ok)
# Build minor graph H: nodes = bag index; edge iff some L-edge between the two bags
part={}
for bi,b in enumerate(bags):
    for v in b: part[inv[v]]=bi
H=nx.Graph(); H.add_nodes_from(range(n))
for u,vv in L.edges():
    if u in part and vv in part and part[u]!=part[vv]:
        H.add_edge(part[u],part[vv])
print(f'minor H: |V|={H.number_of_nodes()} |E|={H.number_of_edges()} mindeg={min(dict(H.degree()).values())}')
# export H as .gr for exact tw
nodes=sorted(H.nodes()); m={x:k+1 for k,x in enumerate(nodes)}
with open('scratch/H.gr','w') as f:
    f.write(f'p tw {H.number_of_nodes()} {H.number_of_edges()}\n')
    for a,b in H.edges(): f.write(f'{m[a]} {m[b]}\n')
print('wrote scratch/H.gr for exact treewidth verification')
