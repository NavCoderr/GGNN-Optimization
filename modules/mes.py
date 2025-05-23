import random
from loguru import logger
import networkx as nx

def generate_mes_tasks(num_tasks=100, valid_nodes=None, warehouse_graph=None, seed=None):
    """
    Generate a batch of MES tasks with unique IDs, priority, and processing times.
    Returns a list of dicts, each with:
      - task_id: int
      - start_node: node id
      - end_node: node id
      - priority: "low" | "medium" | "high"
      - processing_time: float (seconds)
    """
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

        # decide priority and processing time
        congestion_factor = random.uniform(0.1, 1.0)
        base_priority = random.choices(["low","medium","high"], weights=[0.2,0.5,0.3])[0]
        priority = "high" if congestion_factor > 0.7 else base_priority
        processing_time = round(random.uniform(5,20) - (5 if priority == "high" else 0), 2)

        task = {
            "task_id": random.randint(100000, 999999),
            "start_node": s,
            "end_node": e,
            "priority": priority,
            "processing_time": processing_time
        }
        tasks.append(task)
        attempts += 1

    if not tasks:
        raise ValueError("Task generation failed after retries. Check MES logic!")

    # sort by priority: high first, then medium, then low
    order = {"low":0, "medium":1, "high":2}
    tasks.sort(key=lambda t: order[t["priority"]], reverse=True)
    return tasks


def fetch_dynamic_tasks(current_workload, warehouse_graph, base_tasks=5, seed=None):
    """
    Fetch a new batch of MES tasks based on current workload.
    Returns a list of task dicts (same format as generate_mes_tasks).
    """
    if seed is not None:
        random.seed(seed)

    if not isinstance(warehouse_graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("Invalid warehouse_graph provided. Expected a NetworkX graph.")

    valid_nodes = list(warehouse_graph.nodes)
    if len(valid_nodes) < 2:
        raise ValueError("No valid nodes available in warehouse_graph!")

    # scale number of tasks with workload
    num = max(base_tasks + int(current_workload * 0.2), base_tasks)
    tasks = generate_mes_tasks(num, valid_nodes, warehouse_graph, seed=seed)
    logger.info(f"Generated {len(tasks)} dynamic MES tasks.")

    if not tasks:
        # fallback single task
        fallback = {
            "task_id": random.randint(100000, 999999),
            "start_node": valid_nodes[0],
            "end_node": valid_nodes[-1],
            "priority": "medium",
            "processing_time": 10.0
        }
        logger.error("No valid dynamic tasks. Using fallback task.")
        return [fallback]

    return tasks
