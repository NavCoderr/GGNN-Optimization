import networkx as nx
from heapq import heappop, heappush
from loguru import logger
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def bfs_path(G: nx.DiGraph, start: int, goal: int) -> list:
    if start not in G or goal not in G:
        logger.error(f"Invalid nodes for BFS: start={start}, goal={goal}")
        return []

    visited = set()
    heap = [(0.0, start, [start])]

    while heap:
        cost, node, path = heappop(heap)
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nbr in G.neighbors(node):
            if nbr in visited:
                continue
            base = G[node][nbr].get('weight', 1.0)
            cong = get_congestion_level(G, nbr) * 5.0
            energy = compute_energy_cost(G, node, nbr)
            total = cost + base + cong + energy
            heappush(heap, (total, nbr, path + [nbr]))

    logger.error(f"No BFS path from {start} to {goal}")
    return []

if __name__ == '__main__':
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    path = bfs_path(G, 0, max(G.nodes))
    logger.info(f"BFS path: {path}")
