import networkx as nx
import random
from loguru import logger

def create_warehouse_graph(num_nodes=500, num_charging=10, num_rest=20, num_obstacles=30, num_edges=2000):
     
    G = nx.DiGraph()

    
    for i in range(num_nodes):
        G.add_node(i,
                   congestion=random.uniform(0, 1),
                   energy_cost=random.uniform(1, 10),
                   task_priority=random.choice(["Low", "Medium", "High"]))

   
    connect_isolated_nodes(G, num_nodes, num_edges)
 
    charging_stations = random.sample(range(num_nodes), min(num_charging, num_nodes))
    for node in charging_stations:
        G.nodes[node]['type'] = 'charging_station'
 
    rest_stations = random.sample(sorted(set(range(num_nodes)) - set(charging_stations)), min(num_rest, num_nodes))
    for node in rest_stations:
        G.nodes[node]['type'] = 'rest_station'

    place_obstacles(G, min(num_obstacles, num_nodes))

    return G

def connect_isolated_nodes(G, num_nodes, num_edges):
    
    for _ in range(num_edges):
        u, v = random.sample(range(num_nodes), 2)
        if not G.has_edge(u, v):
            weight = random.uniform(1, 5) * (1 + random.uniform(0, 0.5))
            G.add_edge(u, v, weight=weight)

    
    for node in range(num_nodes):
        if len(list(G.neighbors(node))) == 0:
            target = random.choice([n for n in G.nodes if n != node])
            G.add_edge(node, target, weight=random.uniform(1, 5))

        if len(list(G.predecessors(node))) == 0:
            source = random.choice([n for n in G.nodes if n != node])
            G.add_edge(source, node, weight=random.uniform(1, 5))

   
    largest_cc = max(nx.strongly_connected_components(G), key=len, default=set())
    isolated_nodes = set(G.nodes) - set(largest_cc)

    for node in isolated_nodes:
        target = random.choice(list(largest_cc))
        G.add_edge(node, target, weight=random.uniform(1, 5))
        G.add_edge(target, node, weight=random.uniform(1, 5))

def place_obstacles(G, num_obstacles):
    
    valid_nodes = [n for n in G.nodes if "type" not in G.nodes[n]]
    obstacles = []

    for _ in range(num_obstacles):
        if not valid_nodes:
            break

        node = random.choice(valid_nodes)

       
        G.nodes[node]["type"] = "obstacle"
        if not nx.is_strongly_connected(G):
            G.nodes[node]["type"] = None   
        else:
            obstacles.append(node)
            valid_nodes.remove(node)

    logger.info(f" Obstacles placed at: {obstacles}")

def update_congestion_and_obstacles(G):

    
    for node in G.nodes:
        old_congestion = G.nodes[node]['congestion']
        decay_factor = random.uniform(0.05, 0.2)   
        new_congestion = max(0, old_congestion - decay_factor)
        G.nodes[node]['congestion'] = new_congestion

     
    current_obstacles = [n for n in G.nodes if G.nodes[n].get("type") == "obstacle"]
    valid_nodes = [n for n in G.nodes if "type" not in G.nodes[n]]

    moved_obstacles = 0

    for obstacle in current_obstacles:
        G.nodes[obstacle]["type"] = None  

        
        alternative_nodes = [node for node in valid_nodes if nx.has_path(G, node, random.choice(valid_nodes))]
        if alternative_nodes:
            new_location = random.choice(alternative_nodes)
            G.nodes[new_location]["type"] = "obstacle"
            valid_nodes.remove(new_location)  
            moved_obstacles += 1

    logger.info(f" Obstacles moved dynamically: {moved_obstacles}/{len(current_obstacles)}")
 
if __name__ == "__main__":
    warehouse_graph = create_warehouse_graph()
    logger.info(f" Generated warehouse graph with {warehouse_graph.number_of_nodes()} nodes and {warehouse_graph.number_of_edges()} edges.")

    #  Update warehouse in real-time
    update_congestion_and_obstacles(warehouse_graph)
    logger.info(f" Updated congestion and obstacle locations dynamically.")
