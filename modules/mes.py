import random
import networkx as nx
from loguru import logger
from modules.warehouse import create_warehouse_graph

def generate_mes_tasks(num_tasks=100, valid_nodes=None, warehouse_graph=None):
    if not valid_nodes or len(valid_nodes) < 2:
        raise ValueError("MES Task Generation Failed: No valid nodes available!")

    tasks = []
    attempts = 0

    while len(tasks) < num_tasks and attempts < num_tasks * 2:
        try:
            start_node, end_node = random.sample(valid_nodes, 2)

            if warehouse_graph and not nx.has_path(warehouse_graph, start_node, end_node):
                attempts += 1
                continue  # Skip invalid tasks

            priority = random.choices(["low", "medium", "high"], weights=[0.2, 0.5, 0.3])[0]
            congestion_factor = random.uniform(0.1, 1.0)

            task = {
                "task_id": random.randint(1000, 9999),
                "start_node": start_node,
                "end_node": end_node,
                "priority": "high" if congestion_factor > 0.7 else priority,
                "processing_time": round(random.uniform(5, 20) - (5 if priority == "high" else 0), 2)
            }
            tasks.append(task)
        except ValueError:
            logger.warning("Task generation failed. Retrying...")

        attempts += 1

    if not tasks:
        raise ValueError("Task generation failed after multiple retries. Check MES logic!")

    tasks.sort(key=lambda t: ["low", "medium", "high"].index(t["priority"]), reverse=True)
    return tasks

def fetch_dynamic_tasks(current_workload, warehouse_graph, base_tasks=5):
   
    if not isinstance(warehouse_graph, nx.Graph):
        raise TypeError("Invalid warehouse_graph provided. Expected a NetworkX graph.")

    valid_nodes = list(warehouse_graph.nodes)
    if len(valid_nodes) < 2:
        raise ValueError("No valid nodes available in warehouse graph!")

    dynamic_task_count = max(base_tasks + int(current_workload * 0.2), 5)
    tasks = generate_mes_tasks(dynamic_task_count, valid_nodes, warehouse_graph)

    logger.info(f"Fetching {len(tasks)} dynamic MES tasks...")

    task_list = [(task["start_node"], task["end_node"]) for task in tasks]

    retry_attempts = 10  # Increased retries
    while not task_list and retry_attempts > 0:
        logger.warning("No valid tasks found! Retrying task generation...")
        tasks = generate_mes_tasks(dynamic_task_count, valid_nodes, warehouse_graph)
        task_list = [(task["start_node"], task["end_node"]) for task in tasks]
        retry_attempts -= 1

    if not task_list:
        fallback_task = (valid_nodes[0], valid_nodes[-1])
        logger.error("No valid tasks found! Returning a fallback task.")
        task_list.append(fallback_task)

    return task_list

def update_task_priorities(tasks):
     
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise TypeError("Invalid tasks list provided. Expected a list of dictionaries.")

    for task in tasks:
        congestion_factor = random.uniform(0.1, 1.0)
        if congestion_factor > 0.8:
            task["priority"] = "high"

    tasks.sort(key=lambda t: (["low", "medium", "high"].index(t["priority"]), -random.random()), reverse=True)

    logger.info("Task priorities updated based on congestion levels.")

    return tasks

if __name__ == "__main__":
    warehouse_graph = create_warehouse_graph(num_nodes=500, num_edges=2000)
    valid_nodes = list(warehouse_graph.nodes)

    mes_tasks = generate_mes_tasks(num_tasks=10, valid_nodes=valid_nodes, warehouse_graph=warehouse_graph)
    print("MES Task Generation Completed. Sample Tasks:")
    for task in mes_tasks[:5]:
        print(task)

    dynamic_tasks = fetch_dynamic_tasks(current_workload=10, warehouse_graph=warehouse_graph)
    print("Real-time Dynamic Tasks from MES:")
    print(dynamic_tasks)

    updated_tasks = update_task_priorities(mes_tasks)
    print("Updated Task Priorities:")
    for task in updated_tasks[:5]:
        print(task)
