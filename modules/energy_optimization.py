import random
import networkx as nx

def compute_energy_cost(G: nx.DiGraph, start: int, end: int, speed: float = 1.0, load_weight: float = 1.2, congestion_factor: float = 1.0) -> float:
    if start not in G or end not in G:
        return 1000.0
    if not G.has_edge(start, end):
        try:
            path = nx.shortest_path(G, source=start, target=end, weight='weight')
            if len(path) > 1:
                total_weight = sum(G[path[i]][path[i+1]].get('weight', 50.0) for i in range(len(path)-1))
                return max(50.0, total_weight * load_weight / (speed * congestion_factor))
            return 1000.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1000.0
    base = G[start][end].get('weight', random.uniform(50.0, 120.0))
    cost = base * load_weight * congestion_factor / speed
    return max(50.0, min(cost, 1000.0))

def compute_energy_per_task(G: nx.DiGraph, agv_paths: dict, congestion_index: float = 1.0) -> float:
    total_tasks = len(agv_paths)
    if total_tasks == 0:
        return float('nan')
    total_energy = 0.0
    for path in agv_paths.values():
        e = sum(compute_energy_cost(G, path[i], path[i+1], congestion_factor=congestion_index) for i in range(len(path)-1))
        total_energy += e
    return total_energy / total_tasks

def optimize_energy_usage(agvs: list, warehouse_graph: nx.DiGraph) -> None:
    charging = [n for n, d in warehouse_graph.nodes(data=True) if d.get('type') == 'charging_station']
    if not charging:
        return
    for agv in agvs:
        if agv.get('battery', 100) < 30:
            try:
                lengths = {s: nx.shortest_path_length(warehouse_graph, source=agv['position'], target=s, weight='weight') for s in charging}
                nearest = min(lengths, key=lengths.get)
                agv['position'] = nearest
                agv['charging'] = True
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

if __name__ == '__main__':
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    print(compute_energy_cost(G, 0, 1))
    paths = {0: [0, 1, 2], 1: [1, 3]}
    print(compute_energy_per_task(G, paths, congestion_index=0.5))
    agvs = [{'position': 0, 'battery': 20}]
    optimize_energy_usage(agvs, G)
    print(agvs)
