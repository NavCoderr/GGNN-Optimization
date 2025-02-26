import random
from loguru import logger
import networkx as nx

def compute_energy_cost(G, start, end, speed=1.0, load_weight=1.2, congestion_factor=1.0):
    if start not in G or end not in G:
        return 1000  

    if not G.has_edge(start, end):
        try:
            path = nx.shortest_path(G, source=start, target=end, weight="weight")
            if len(path) > 1:
                total_weight = sum(G[path[i]][path[i + 1]].get("weight", 50) for i in range(len(path) - 1))
                return max(50, total_weight * load_weight / (speed * congestion_factor))
            else:
                return 1000  
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1000  

    energy = G[start][end].get("weight", random.uniform(50, 120))  
    return max(50, min((energy * load_weight * congestion_factor) / speed, 1000))  

def compute_energy_per_task(G, agv_paths, congestion_index=1.0):
    total_energy = 0
    total_tasks = len(agv_paths)

    if total_tasks == 0:
        return float("nan")

    for agv, path in agv_paths.items():
        energy_per_task = sum(
            compute_energy_cost(G, path[i], path[i+1], congestion_factor=congestion_index)
            for i in range(len(path)-1)
        )
        total_energy += energy_per_task

    return total_energy / total_tasks  

def optimize_energy_usage(agvs, warehouse_graph):
    charging_stations = [node for node in warehouse_graph.nodes if warehouse_graph.nodes[node].get("type") == "charging_station"]
    
    if not charging_stations:
        return

    for agv in agvs:
        if agv["battery"] < 30:  
            try:
                shortest_paths = {
                    station: nx.shortest_path_length(warehouse_graph, source=agv["position"], target=station, weight="weight")
                    for station in charging_stations
                }
                nearest_station = min(shortest_paths, key=shortest_paths.get)
                agv["position"] = nearest_station
                agv["charging"] = True  
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass  

if __name__ == "__main__":
    logger.info("Energy Optimization Module Loaded.")
