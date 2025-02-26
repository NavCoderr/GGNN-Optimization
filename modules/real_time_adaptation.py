import random
import networkx as nx
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost
from modules.warehouse import create_warehouse_graph

def adapt_to_real_time_conditions(G, num_agvs=10):
    adaptation_data = []
    
    for agv_id in range(num_agvs):
        current_node = random.choice(list(G.nodes))
        congestion = get_congestion_level(G, current_node)
        obstacle_detected = G.nodes[current_node].get("type", "") == "obstacle"

        new_path = None
        delay = congestion * 3  

        if congestion > 0.7 or obstacle_detected:
            neighbors = list(G.neighbors(current_node))
            if neighbors:
                try:
                
                    new_path = nx.astar_path(G, current_node, random.choice(neighbors), weight="weight")
                except nx.NetworkXNoPath:
                    new_path = None
            
          
            if not new_path:
                alternative_nodes = [n for n in G.nodes if "obstacle" not in G.nodes[n].get("type", "")]
                new_path = [current_node, random.choice(alternative_nodes)] if alternative_nodes else None

        #  Ensure energy cost is only computed if movement occurs
        energy_cost = compute_energy_cost(G, current_node, new_path[1] if new_path and len(new_path) > 1 else current_node)

        agv = {
            "agv_id": agv_id,
            "rerouted": bool(new_path),
            "new_path": new_path if new_path else "No change",
            "obstacle_detected": obstacle_detected,
            "delay_time": round(delay, 2),
            "energy_cost": energy_cost
        }
        adaptation_data.append(agv)

    return adaptation_data

if __name__ == "__main__":
    warehouse_graph = create_warehouse_graph()
    adaptation_info = adapt_to_real_time_conditions(warehouse_graph)
    print(" Real-Time Adaptation Sample:")
    for agv in adaptation_info[:5]:
        print(agv)
