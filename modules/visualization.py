import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import get_congestion_level

def visualize_warehouse(G, agvs, cycles=5, save_video=False):
    

    plt.ion()   
    fig, ax = plt.subplots(figsize=(10, 8))

   
    if not G.nodes:
        print("⚠️ Warning: No nodes found in the warehouse graph! Visualization skipped.")
        return

     
    pos = nx.spring_layout(G, seed=42)

  
    for node in G.nodes:
        if node not in pos:
            pos[node] = (random.uniform(-1, 1), random.uniform(-1, 1))

    frames = []   

    for cycle in range(cycles):
        ax.clear()   
        ax.set_title(f"Warehouse AGV Movement - Cycle {cycle + 1}")

        # Step 1 
        congestion_levels = [get_congestion_level(G, n) for n in G.nodes]
        node_colors = [
            (1, 0, 0, min(congestion, 1.0)) if congestion > 0 else "gray"
            for congestion in congestion_levels
        ]
        nx.draw(G, pos, node_color=node_colors, with_labels=False, edge_color="gray", node_size=50, ax=ax)

        # Step 2 
        agv_positions = {}
        for agv in agvs:
            if agv.position in G.nodes:
                agv_positions[agv.position] = pos[agv.position]
            else:
                print(f" Warning: AGV {agv.id} assigned to an invalid position ({agv.position}). Reassigning...")
                valid_nodes = list(G.nodes)
                if valid_nodes:
                    agv.position = random.choice(valid_nodes)   
                    agv_positions[agv.position] = pos[agv.position]
                else:
                    print(f" Critical Error: No valid nodes available for AGV {agv.id}. Skipping AGV.")

        # Step 3 
        for agv in agvs:
            neighbors = list(G.neighbors(agv.position))
            if neighbors:
                 
                best_neighbor = min(neighbors, key=lambda n: get_congestion_level(G, n))
                agv.position = best_neighbor

        # Step 4 
        if agv_positions:
            nx.draw_networkx_nodes(
                G, pos, nodelist=agv_positions.keys(), node_color="blue",
                node_size=150, label="AGVs", ax=ax
            )
        else:
            print(" Warning: No valid AGV positions found for visualization. Skipping visualization.")

        plt.pause(0.1)  

    plt.ioff()   
    plt.show()   
