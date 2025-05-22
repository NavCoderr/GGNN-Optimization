import heapq
import networkx as nx
from loguru import logger
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def astar_path(G: nx.DiGraph, start: int, goal: int) -> list:
    def heuristic(u: int, v: int) -> float:
        try:
            return nx.shortest_path_length(G, u, v, weight='weight') * 0.8
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    if start not in G or goal not in G:
        logger.error(f"Invalid start/goal for A*: {start}, {goal}")
        return []

    open_set = [(heuristic(start, goal), start)]
    came_from = {}
    g_score = {n: float('inf') for n in G.nodes}
    g_score[start] = 0.0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for nbr in G.neighbors(current):
            base = G[current][nbr].get('weight', 1.0)
            penalty = get_congestion_level(G, nbr) * 5.0
            energy = compute_energy_cost(G, current, nbr)
            cost = base + penalty + energy
            tentative = g_score[current] + cost

            if tentative < g_score[nbr]:
                came_from[nbr] = current
                g_score[nbr] = tentative
                f = tentative + heuristic(nbr, goal)
                heapq.heappush(open_set, (f, nbr))

    logger.error(f"A* failed to find a path from {start} to {goal}")
    return []

if __name__ == '__main__':
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    path = astar_path(G, 0, max(G.nodes))
    logger.info(f"A* path: {path}")
