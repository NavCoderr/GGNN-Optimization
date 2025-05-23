import heapq
import networkx as nx
from loguru import logger
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def astar_path(G: nx.Graph, start: int, goal: int) -> list:
    
    def heuristic(u: int, v: int) -> float:
        # admissible heuristic: 0.8 × shortest‐path distance by weight
        try:
            return nx.shortest_path_length(G, u, v, weight='weight') * 0.8
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    if start not in G.nodes or goal not in G.nodes:
        logger.error(f"Invalid start/goal for A*: {start}, {goal}")
        return []

    open_set = [(heuristic(start, goal), start)]
    came_from = {}
    g_score = {n: float('inf') for n in G.nodes}
    g_score[start] = 0.0
    visited = set()

    while open_set:
        f_current, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)

        for nbr in G.neighbors(current):
            # base travel cost
            w = G.edges[current, nbr].get('weight', 1.0)
            # congestion penalty on this edge
            penalty = get_congestion_level(G, current, nbr) * 5.0
            # energy cost to traverse
            energy = compute_energy_cost(G, current, nbr)
            step_cost = w + penalty + energy

            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score[nbr]:
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score = tentative_g + heuristic(nbr, goal)
                heapq.heappush(open_set, (f_score, nbr))

    logger.error(f"A* failed to find a path from {start} to {goal}")
    return []

if __name__ == '__main__':
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    path = astar_path(G, 0, max(G.nodes))
    logger.info(f"A* path: {path}")
