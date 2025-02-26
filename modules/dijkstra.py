import networkx as nx
import random 
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

def dijkstra_path(G, start, goal):
    
    if start not in G or goal not in G:
        print(f" Error: Start ({start}) or Goal ({goal}) node does not exist.")
        return None

    try:
        temp_graph = G.copy()  # Use a temporary graph

        for u, v in temp_graph.edges():
            congestion_penalty = get_congestion_level(temp_graph, v) * random.uniform(3, 6)   
            energy_cost = compute_energy_cost(temp_graph, u, v)
            temp_graph[u][v]['weight'] += congestion_penalty + energy_cost   

        path = nx.dijkstra_path(temp_graph, start, goal, weight='weight')
        return path

    except nx.NetworkXNoPath:
        print(f" No path found between {start} and {goal}.")
        return None

if __name__ == "__main__":
    G = create_warehouse_graph()
    start, goal = 0, 499  
    path = dijkstra_path(G, start, goal)
    print("Dijkstra Path:", path if path else "No valid path found.")
