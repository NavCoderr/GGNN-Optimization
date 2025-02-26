import random
import networkx as nx
from loguru import logger
 
def update_congestion(G, node):
     
    
    if node not in G.nodes:
        logger.warning(f"Attempted to update congestion for an invalid node {node}. Skipping.")
        return

     
    if "congestion" not in G.nodes[node]:
        G.nodes[node]["congestion"] = random.uniform(0.1, 0.3)

     
    G.nodes[node]["congestion"] += random.uniform(0.05, 0.1)
    G.nodes[node]["congestion"] = min(G.nodes[node]["congestion"], 1.0)  # Cap at max 1.0


def get_congestion_level(G, node):
     
    
    if node not in G.nodes:
        return 0  # Return zero congestion for invalid nodes
    
    return G.nodes[node].get("congestion", 0)


def decay_congestion(G):
     
    
    for node in list(G.nodes):
        if "congestion" in G.nodes[node]:
            G.nodes[node]["congestion"] *= 0.9  # Reduce congestion by 10% per cycle
            
            # If congestion is very low, reset to zero for cleanup
            if G.nodes[node]["congestion"] < 0.01:
                G.nodes[node]["congestion"] = 0


# Test the congestion system
if __name__ == "__main__":
    # Create a small test graph
    G = nx.DiGraph()
    G.add_nodes_from(range(5))  # Adding 5 nodes

    test_node = 2

    # Simulate congestion updates
    for _ in range(5):
        update_congestion(G, test_node)
        print(f"Congestion at node {test_node}: {get_congestion_level(G, test_node):.2f}")

    # Apply congestion decay
    decay_congestion(G)
    print(f"After decay: Congestion at node {test_node}: {get_congestion_level(G, test_node):.2f}")
