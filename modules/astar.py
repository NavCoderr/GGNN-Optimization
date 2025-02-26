import networkx as nx
import heapq
import random  
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def astar_path(G, start, goal):
     

    def heuristic(node1, node2):
         
        try:
            return nx.shortest_path_length(G, node1, node2, weight="weight") * 0.8  # Estimate based on warehouse layout
        except nx.NetworkXNoPath:
            return float('inf')  # Avoid incorrect estimates

    if start not in G or goal not in G:
        print(f"Error: Start ({start}) or Goal ({goal}) node does not exist.")
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in G.nodes}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return the path from start to goal

        for neighbor in G.neighbors(current):
            congestion_penalty = get_congestion_level(G, neighbor) * random.uniform(3, 7)  # Adaptive penalty
            energy_cost = compute_energy_cost(G, current, neighbor)
            weight = G[current][neighbor].get('weight', 1.0) + congestion_penalty + energy_cost  # Ensure weight exists

            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print(f"No path found between {start} and {goal}.")
    return None  # No path found

# Run A* pathfinding if executed as a script
if __name__ == "__main__":
    G = create_warehouse_graph()
    start, goal = 0, 49  # Example start and goal

    path = astar_path(G, start, goal)
    print("A* Path:", path if path else "No valid path found.")
