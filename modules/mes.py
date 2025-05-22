import random
from loguru import logger
import networkx as nx

def generate_mes_tasks(num_tasks=100, valid_nodes=None, warehouse_graph=None, seed=None):
    
    if seed is not None:
        random.seed(seed)

    if not valid_nodes or len(valid_nodes) < 2:
        raise ValueError("MES Task Generation Failed: No valid nodes available!")

    tasks = []
    attempts = 0

    while len(tasks) < num_tasks and attempts < num_tasks * 3:
        s, e = random.sample(valid_nodes, 2)
        if warehouse_graph and not nx.has_path(warehouse_graph, s, e):
            attempts += 1
            continue

        priority = random.choices(["low","medium","high"], weights=[0.2,0.5,0.3])[0]
        congestion_factor = random.uniform(0.1, 1.0)
        task = {
            "task_id": random.randint(1000, 9999),
            "start_node": s,
            "end_node": e,
            "priority": "high" if congestion_factor > 0.7 else priority,
            "processing_time": round(random.uniform(5,20) - (5 if priority == "high" else 0), 2)
        }
        tasks.append(task)
        attempts += 1

    if not tasks:
        raise ValueError("Task generation failed after retries. Check MES logic!")

    # sort by priority: high > medium > low
    order = {"low":0, "medium":1, "high":2}
    tasks.sort(key=lambda t: order.get(t["priority"],1), reverse=True)
    return tasks


def fetch_dynamic_tasks(current_workload, warehouse_graph, base_tasks=5, seed=None):
    
    if seed is not None:
        random.seed(seed)

    if not isinstance(warehouse_graph, nx.Graph) and not isinstance(warehouse_graph, nx.DiGraph):
        raise TypeError("Invalid warehouse_graph provided. Expected a NetworkX graph.")

    valid_nodes = list(warehouse_graph.nodes)
    if len(valid_nodes) < 2:
        raise ValueError("No valid nodes available in warehouse_graph!")

    num = max(base_tasks + int(current_workload * 0.2), base_tasks)
    tasks = generate_mes_tasks(num, valid_nodes, warehouse_graph, seed=seed)
    logger.info(f"Generated {len(tasks)} dynamic MES tasks.")

    task_list = [(t["start_node"], t["end_node"]) for t in tasks]

    if not task_list:
        # fallback to any two nodes
        task_list = [(valid_nodes[0], valid_nodes[-1])]
        logger.error("No valid dynamic tasks. Using fallback task.")

    return task_list
