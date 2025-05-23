import networkx as nx
from loguru import logger
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def dijkstra_path(G: nx.Graph, start: int, goal: int) -> list:
    
    if start not in G.nodes or goal not in G.nodes:
        logger.error(f"Invalid start or goal for Dijkstra: {start}, {goal}")
        return []

    # Build auxiliary directed graph H with updated edge weights
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        base = data.get('weight', 1.0)
        # correct call: get_congestion_level(G, u, v)
        congestion_penalty = get_congestion_level(G, u, v) * 5.0
        energy = compute_energy_cost(G, u, v)
        H.add_edge(u, v, weight=base + congestion_penalty + energy)

    # Run Dijkstra on H
    try:
        return nx.dijkstra_path(H, start, goal, weight='weight')
    except nx.NetworkXNoPath:
        logger.error(f"No path found by Dijkstra from {start} to {goal}")
        return []

if __name__ == "__main__":
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    path = dijkstra_path(G, 0, max(G.nodes))
    logger.info(f"Dijkstra path: {path}")
