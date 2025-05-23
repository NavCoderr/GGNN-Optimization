import random
import math
import networkx as nx

def create_warehouse_graph(
    num_nodes=500,
    num_charging=10,
    num_rest=20,
    num_obstacles=30,
    num_edges=2000,
    seed=None
):
    
    if seed is not None:
        random.seed(seed)

    G = nx.DiGraph()

    # 1) Create nodes with random positions
    for i in range(num_nodes):
        pos = (random.uniform(0, 100), random.uniform(0, 100))
        G.add_node(i,
                   pos=pos,
                   congestion=0.0,
                   task_priority=random.choice(["Low", "Medium", "High"]),
                   type="normal")

    # 2) Add random directed edges with physical length & weight=length
    _connect_nodes(G, num_nodes, num_edges)

    # 3) Assign charging stations
    charging = random.sample(range(num_nodes), min(num_charging, num_nodes))
    for n in charging:
        G.nodes[n]['type'] = 'charging_station'

    # 4) Assign rest stations
    normals = [n for n, d in G.nodes(data=True) if d['type'] == 'normal']
    rest = random.sample(normals, min(num_rest, len(normals)))
    for n in rest:
        G.nodes[n]['type'] = 'rest_station'

    # 5) Place obstacle nodes (ensuring connectivity)
    _place_obstacles(G, num_obstacles)

    return G


def _connect_nodes(G: nx.DiGraph, num_nodes: int, num_edges: int) -> None:
     
    def add_edge(u: int, v: int):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        length = math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])
        G.add_edge(u, v,
                   weight=length,    # use physical length as weight
                   length=length,    # physical distance in meters
                   congestion=0.0,   # initial edge congestion
                   urgency=0)        # initial edge urgency

    # 2a) Random edges
    edges_added = 0
    while edges_added < num_edges:
        u, v = random.sample(range(num_nodes), 2)
        if not G.has_edge(u, v):
            add_edge(u, v)
            edges_added += 1

    # 2b) Guarantee every node has out-degree ≥1 and in-degree ≥1
    for n in range(num_nodes):
        if G.out_degree(n) == 0:
            m = random.choice([m for m in G.nodes if m != n])
            add_edge(n, m)
        if G.in_degree(n) == 0:
            m = random.choice([m for m in G.nodes if m != n])
            add_edge(m, n)

    # 2c) Strong connectivity: connect isolated components
    comps = list(nx.strongly_connected_components(G))
    if len(comps) > 1:
        main = max(comps, key=len)
        for comp in comps:
            if comp is main:
                continue
            # connect one node from comp to one in main, and vice versa
            u = random.choice(list(comp))
            v = random.choice(list(main))
            add_edge(u, v)
            add_edge(v, u)


def _place_obstacles(G: nx.DiGraph, num_obstacles: int) -> None:
     
    candidates = [n for n, d in G.nodes(data=True) if d['type'] == 'normal']
    random.shuffle(candidates)
    placed = 0

    for n in candidates:
        if placed >= num_obstacles:
            break
        G.nodes[n]['type'] = 'obstacle'
        if nx.is_strongly_connected(G):
            placed += 1
        else:
            # revert if graph becomes disconnected
            G.nodes[n]['type'] = 'normal'
