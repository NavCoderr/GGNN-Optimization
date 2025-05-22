import random
import networkx as nx
from loguru import logger
from modules.warehouse import create_warehouse_graph


def fetch_real_time_routes(warehouse_graph, mes_workload, num_routes=10):
    routes = []
    nodes = list(warehouse_graph.nodes)
    attempts = 0

    while len(routes) < num_routes and attempts < num_routes * 2:
        start, end = random.sample(nodes, 2)
        if not nx.has_path(warehouse_graph, start, end):
            attempts += 1
            continue

        congestion_factor = random.uniform(0.1, 1.0)
        congestion_factor = min(1.0, congestion_factor + mes_workload * 0.01)

        path = nx.shortest_path(warehouse_graph, start, end, weight="weight")
        route = {
            "start_node": start,
            "end_node": end,
            "congestion_factor": round(congestion_factor, 2),
            "priority": "high" if congestion_factor > 0.7 else random.choice(["low", "medium", "high"]),
            "optimized_path": path
        }
        routes.append(route)
        attempts += 1

    if len(routes) < num_routes:
        logger.warning(f"Only {len(routes)} TMS routes generated out of {num_routes} requested.")

    return routes


def get_dynamic_task_updates(warehouse_graph, mes_workload, num_tasks=5):
    tasks = []
    nodes = list(warehouse_graph.nodes)
    attempts = 0

    while len(tasks) < num_tasks and attempts < num_tasks * 2:
        start, end = random.sample(nodes, 2)
        if not nx.has_path(warehouse_graph, start, end):
            attempts += 1
            continue

        congestion_factor = min(1.0, random.uniform(0.1, 1.0) + mes_workload * 0.01)
        task = {
            "task_id": random.randint(1000, 9999),
            "start_node": start,
            "end_node": end,
            "priority": "high" if congestion_factor > 0.7 else random.choice(["low", "medium", "high"]),
            "estimated_time": round(random.uniform(5, 30), 2)
        }
        tasks.append(task)
        attempts += 1

    if len(tasks) < num_tasks:
        logger.warning(f"Only {len(tasks)} task updates generated out of {num_tasks} requested.")

    return tasks


def generate_tms_routes(num_routes=10):
    return [
        {
            "route_id": i,
            "start_node": random.randint(0, 100),
            "end_node": random.randint(101, 200)
        }
        for i in range(num_routes)
    ]


if __name__ == "__main__":
    G = create_warehouse_graph()
    tms = fetch_real_time_routes(G, mes_workload=15, num_routes=10)
    print("Real-Time Optimized Routes from TMS:")
    for r in tms[:5]:
        print(r)

    updates = get_dynamic_task_updates(G, mes_workload=15, num_tasks=5)
    print("Dynamic Task Updates from TMS:")
    for u in updates[:5]:
        print(u)
