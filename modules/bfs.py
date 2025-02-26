import networkx as nx
import random
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost
from heapq import heappop, heappush   

def bfs_path(G, start, goal):
     
    
    if start not in G or goal not in G:
        print(f"Error: Start ({start}) or Goal ({goal}) node does not exist.")
        return None

    try:
        queue = [(0, start, [start])]  # (Total Cost, Current Node, Path)
        visited = set()

        while queue:
            total_cost, current, path = heappop(queue)  # Always explore the lowest-cost path first

            if current == goal:
                return path  # Return the first valid path found

            if current in visited:
                continue
            visited.add(current)  # Mark as visited immediately

            for neighbor in G.neighbors(current):
                if neighbor in visited:
                    continue

                congestion_penalty = get_congestion_level(G, neighbor) * random.uniform(2, 6)
                energy_cost = compute_energy_cost(G, current, neighbor)
                weight = G[current][neighbor]["weight"] + congestion_penalty + energy_cost

                heappush(queue, (total_cost + weight, neighbor, path + [neighbor]))  # Push into priority queue

        print(f"No path found between {start} and {goal}.")
        return None

    except nx.NetworkXNoPath:
        print(f"No path found between {start} and {goal}.")
        return None


# Run BFS pathfinding if executed as a script
if __name__ == "__main__":
    G = create_warehouse_graph()
    start, goal = 0, 49

    path = bfs_path(G, start, goal)
    print("BFS Path:", path if path else "No valid path found.")

