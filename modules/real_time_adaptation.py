import random
import networkx as nx
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost
from modules.warehouse import create_warehouse_graph
from loguru import logger

def adapt_to_real_time_conditions(G, num_agvs=10, seed=None):
    if seed is not None:
        random.seed(seed)
    adaptation_data = []
    nodes = list(G.nodes)
    for agv_id in range(num_agvs):
        current = random.choice(nodes)
        congest = get_congestion_level(G, current)
        is_obstacle = G.nodes[current].get("type") == "obstacle"
        new_path = None
        delay_time = round(congest * 3, 2)
        if congest > 0.7 or is_obstacle:
            neighbors = list(G.successors(current)) + list(G.predecessors(current))
            valid_neighbors = [n for n in neighbors if G.has_edge(current, n)]
            if valid_neighbors:
                dest = random.choice(valid_neighbors)
                try:
                    new_path = nx.astar_path(G, current, dest, weight="weight")
                except nx.NetworkXNoPath:
                    new_path = None
            if not new_path:
                fallback = [n for n in nodes if G.nodes[n].get("type") != "obstacle"]
                if fallback:
                    new_node = random.choice(fallback)
                    new_path = [current, new_node]
        if new_path and len(new_path) > 1:
            energy_cost = compute_energy_cost(G, new_path[0], new_path[1])
        else:
            energy_cost = 0.0
        adaptation_data.append({
            "agv_id": agv_id,
            "current_node": current,
            "rerouted": new_path is not None,
            "new_path": new_path or [],
            "obstacle_detected": is_obstacle,
            "delay_time_s": delay_time,
            "energy_cost": round(energy_cost, 2)
        })
    return adaptation_data

if __name__ == "__main__":
    G = create_warehouse_graph(seed=42)
    info = adapt_to_real_time_conditions(G, num_agvs=5, seed=42)
    for agv_info in info:
        logger.info(agv_info)
