import networkx as nx
from loguru import logger
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def dijkstra_path(G: nx.DiGraph, start: int, goal: int) -> list:
    if start not in G or goal not in G:
        logger.error(f"Invalid start or goal: {start}, {goal}")
        return []
    H = G.copy()
    for u, v, data in H.edges(data=True):
        base = data.get('weight', 1.0)
        congestion = get_congestion_level(G, v)
        penalty = congestion * 5.0
        energy = compute_energy_cost(G, u, v)
        data['weight'] = base + penalty + energy
    try:
        return nx.dijkstra_path(H, start, goal, weight='weight')
    except nx.NetworkXNoPath:
        logger.error(f"No path from {start} to {goal}")
        return []

if __name__ == '__main__':
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    path = dijkstra_path(G, 0, max(G.nodes))
    logger.info(f"Dijkstra path: {path}")
