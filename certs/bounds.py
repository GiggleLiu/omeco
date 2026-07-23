import pickle, networkx as nx, random
L,idmap = pickle.load(open('scratch/L.pkl','rb'))
n=L.number_of_nodes()

def degeneracy(G):
    G=G.copy(); mx=0
    while G.number_of_nodes():
        v=min(G.nodes(), key=lambda x:G.degree(x))
        mx=max(mx,G.degree(v)); G.remove_node(v)
    return mx
print('degeneracy LB (=core number):', degeneracy(L))

# MMW / minor-min-width (Gogate-Dechter): certified LB via minor contractions.
def mmw(G):
    G=nx.Graph(G); lb=0
    while G.number_of_nodes()>1:
        v=min(G.nodes(), key=lambda x:G.degree(x))
        d=G.degree(v); lb=max(lb,d)
        nbrs=list(G.neighbors(v))
        if not nbrs: G.remove_node(v); continue
        # contract v into its minimum-degree neighbor u (least-c heuristic)
        u=min(nbrs, key=lambda x:G.degree(x))
        for w in list(G.neighbors(v)):
            if w!=u: G.add_edge(u,w)
        G.remove_node(v)
    return lb
print('MMW (minor-min-width, least-c) LB:', mmw(L))

# min-fill heuristic upper bound
def minfill_ub(G):
    G=nx.Graph(G); width=0
    while G.number_of_nodes():
        best=None;bf=None
        for v in G.nodes():
            nb=list(G.neighbors(v)); f=0
            for i in range(len(nb)):
                for j in range(i+1,len(nb)):
                    if not G.has_edge(nb[i],nb[j]): f+=1
            if bf is None or f<bf: bf=f; best=v
        width=max(width,G.degree(best))
        nb=list(G.neighbors(best))
        for i in range(len(nb)):
            for j in range(i+1,len(nb)):
                G.add_edge(nb[i],nb[j])
        G.remove_node(best)
    return width
def mindeg_ub(G):
    G=nx.Graph(G); width=0
    while G.number_of_nodes():
        v=min(G.nodes(), key=lambda x:G.degree(x))
        width=max(width,G.degree(v))
        nb=list(G.neighbors(v))
        for i in range(len(nb)):
            for j in range(i+1,len(nb)):
                G.add_edge(nb[i],nb[j])
        G.remove_node(v)
    return width
print('min-fill heuristic UB (tw <= this):', minfill_ub(L))
print('min-degree heuristic UB:', mindeg_ub(L))
