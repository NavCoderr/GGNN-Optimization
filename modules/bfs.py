import networkx as nx
import heapq
from loguru import logger
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def bfs_path(G: nx.Graph, start: int, goal: int) -> list:
    """
    Uniform-cost search variant (often called Dijkstra) using BFS‐style expansion
    but incorporating weight, congestion, and energy penalties.
    Returns the list of nodes from start to goal, or [] if none.
    """
    if start not in G.nodes or goal not in G.nodes:
        logger.error(f"Invalid nodes for BFS: start={start}, goal={goal}")
        return []

    # priority queue of (cumulative_cost, node, path_list)
    frontier = [(0.0, start, [start])]
    visited = set()

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)

        for nbr in G.neighbors(node):
            if nbr in visited:
                continue

            # base graph weight
            w = G.edges[node, nbr].get('weight', 1.0)
            # congestion penalty on this edge
            penalty = get_congestion_level(G, node, nbr) * 5.0
            # energy cost to traverse
            energy = compute_energy_cost(G, node, nbr)
            step_cost = w + penalty + energy
            new_cost = cost + step_cost

            heapq.heappush(frontier, (new_cost, nbr, path + [nbr]))

    logger.error(f"No BFS path from {start} to {goal}")
    return []

if __name__ == '__main__':
    from modules.warehouse import create_warehouse_graph
    G = create_warehouse_graph(seed=42)
    start, goal = 0, max(G.nodes)
    path = bfs_path(G, start, goal)
    logger.info(f"BFS path: {path}")
