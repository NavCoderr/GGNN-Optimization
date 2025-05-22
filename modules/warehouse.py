import random
import networkx as nx

def create_warehouse_graph(num_nodes=500, num_charging=10, num_rest=20, num_obstacles=30, num_edges=2000, seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i,
                   congestion=random.uniform(0,1),
                   energy_cost=random.uniform(1,10),
                   task_priority=random.choice(["Low","Medium","High"]))
    _connect_nodes(G, num_nodes, num_edges)
    charging = random.sample(range(num_nodes), min(num_charging,num_nodes))
    for n in charging:
        G.nodes[n]['type'] = 'charging_station'
    rest = random.sample([n for n in range(num_nodes) if n not in charging], min(num_rest,num_nodes))
    for n in rest:
        G.nodes[n]['type'] = 'rest_station'
    _place_obstacles(G, num_obstacles)
    return G

def _connect_nodes(G, num_nodes, num_edges):
    for _ in range(num_edges):
        u,v = random.sample(range(num_nodes),2)
        if not G.has_edge(u,v):
            w = random.uniform(1,5)*(1+random.uniform(0,0.5))
            G.add_edge(u,v,weight=w)
    for n in range(num_nodes):
        if len(list(G.neighbors(n)))==0:
            t = random.choice([m for m in G.nodes if m!=n])
            G.add_edge(n,t,weight=random.uniform(1,5))
        if len(list(G.predecessors(n)))==0:
            s = random.choice([m for m in G.nodes if m!=n])
            G.add_edge(s,n,weight=random.uniform(1,5))
    comps = list(nx.strongly_connected_components(G))
    if comps:
        largest = max(comps,key=len)
        isolated = set(G.nodes)-largest
        for n in isolated:
            t = random.choice(list(largest))
            G.add_edge(n,t,weight=random.uniform(1,5))
            G.add_edge(t,n,weight=random.uniform(1,5))

def _place_obstacles(G, num_obstacles):
    valid = [n for n in G.nodes if 'type' not in G.nodes[n]]
    for _ in range(min(len(valid),num_obstacles)):
        n = random.choice(valid)
        G.nodes[n]['type'] = 'obstacle'
        if not nx.is_strongly_connected(G):
            del G.nodes[n]['type']
        else:
            valid.remove(n)
