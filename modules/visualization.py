import networkx as nx
import matplotlib.pyplot as plt
import random
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import update_congestion, get_congestion_level

def visualize_warehouse(
    G: nx.DiGraph,
    agvs: list,
    cycles: int = 5,
    seed: int = 42
) -> None:
    
    random.seed(seed)

    # Prepare interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()

    # Fixed layout so nodes don't jump
    pos = nx.spring_layout(G, seed=seed)

    # Main animation loop
    for cycle in range(cycles):
        ax.clear()
        ax.set_title(f"Warehouse AGV Movement — Cycle {cycle + 1}", fontsize=14)
        ax.set_axis_off()

        # 1) Update congestion values
        update_congestion(G)

        # 2) Color nodes by current congestion
        congestion_vals = [get_congestion_level(G, n) for n in G.nodes]
        node_colors = [
            (1, 0, 0, min(c, 1.0)) if c > 0 else (0.6, 0.6, 0.6, 0.3)
            for c in congestion_vals
        ]
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.4, ax=ax)
        nx.draw_networkx_nodes(
            G, pos,
            node_size=50,
            node_color=node_colors,
            ax=ax
        )

        # 3) Draw and validate AGV positions
        valid_positions = []
        for agv in agvs:
            if agv.position not in G.nodes:
                # Reassign to a random valid node
                agv.position = random.choice(list(G.nodes))
            valid_positions.append(agv.position)

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=valid_positions,
            node_size=150,
            node_color="blue",
            label="AGVs",
            ax=ax
        )

        # 4) Move each AGV to neighbor with lowest congestion
        for agv in agvs:
            neighbors = list(G.successors(agv.position)) + list(G.predecessors(agv.position))
            # Filter neighbors that remain connected
            neighbors = [n for n in neighbors if nx.has_path(G, agv.position, n)]
            if neighbors:
                best = min(neighbors, key=lambda n: get_congestion_level(G, n))
                agv.position = best

        # 5) Draw legend and pause
        ax.legend(loc="upper right")
        plt.pause(0.5)

    plt.ioff()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Create a graph and dummy AGV list for demonstration
    G = create_warehouse_graph(num_nodes=100, num_charging=5, num_rest=5, num_obstacles=10, num_edges=300, seed=42)
    class DummyAGV:
        def __init__(self, id, position):
            self.id = id
            self.position = position

    # Place 5 AGVs at random starting nodes
    agvs = [DummyAGV(i, random.choice(list(G.nodes))) for i in range(5)]

    visualize_warehouse(G, agvs, cycles=10, seed=123)
