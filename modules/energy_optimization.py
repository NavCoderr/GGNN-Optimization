import random
from loguru import logger
import networkx as nx

def compute_energy_cost(G, start, end, speed=1.0):
    if start not in G or end not in G:
        return 1000  

    if not G.has_edge(start, end):
        try:
            path = nx.shortest_path(G, source=start, target=end, weight="weight")
            if len(path) > 1:
                total_weight = sum(G[path[i]][path[i + 1]].get("weight", 10) for i in range(len(path) - 1))
                return max(5, total_weight / speed)
            else:
                return 1000  
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1000  

    energy = G[start][end].get("weight", random.uniform(3, 7))
    return max(0.01, min(energy, 1000))

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
