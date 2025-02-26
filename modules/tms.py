import random
import networkx as nx
from loguru import logger

def fetch_real_time_routes(warehouse_graph, mes_workload, num_routes=10):
    
    routes = []
    nodes = list(warehouse_graph.nodes)
    retries = 0

    while len(routes) < num_routes and retries < num_routes * 2:   
        try:
            start, end = random.sample(nodes, 2)
            if nx.has_path(warehouse_graph, start, end):
                congestion_factor = round(random.uniform(0.1, 1.0), 2)
                congestion_factor += (mes_workload * 0.01)  
                congestion_factor = min(1.0, congestion_factor)

                optimized_path = nx.shortest_path(warehouse_graph, start, end, weight="weight")

                route = {
                    "start_node": start,
                    "end_node": end,
                    "congestion_factor": congestion_factor,
                    "priority": "high" if congestion_factor > 0.7 else random.choice(["low", "medium", "high"]),
                    "optimized_path": optimized_path
                }
                routes.append(route)
            else:
                retries += 1   
        except Exception as e:
            logger.error(f" Error generating real-time routes: {e}")
            retries += 1   

    if len(routes) < num_routes:
        logger.warning(f" Only {len(routes)} routes were generated out of requested {num_routes}.")

    return routes

def get_dynamic_task_updates(warehouse_graph, mes_workload, num_tasks=5):
  
    tasks = []
    nodes = list(warehouse_graph.nodes)
    retries = 0

    while len(tasks) < num_tasks and retries < num_tasks * 2:
        try:
            start, end = random.sample(nodes, 2)
            if nx.has_path(warehouse_graph, start, end):
                congestion_factor = round(random.uniform(0.1, 1.0), 2)
                congestion_factor += (mes_workload * 0.01)
                congestion_factor = min(1.0, congestion_factor)

                task = {
                    "task_id": random.randint(1000, 9999),
                    "start_node": start,
                    "end_node": end,
                    "priority": "high" if congestion_factor > 0.7 else random.choice(["low", "medium", "high"]),
                    "estimated_time": round(random.uniform(5, 30), 2)
                }
                tasks.append(task)
            else:
                retries += 1
        except Exception as e:
            logger.error(f" Error generating task updates: {e}")
            retries += 1

    if len(tasks) < num_tasks:
        logger.warning(f" Only {len(tasks)} tasks were generated out of requested {num_tasks}.")

    return tasks

def generate_tms_routes(num_routes=10):
     
    return [{"route_id": i, "start_node": random.randint(0, 100), "end_node": random.randint(101, 200)} for i in range(num_routes)]

 
if __name__ == "__main__":
    from modules.warehouse import create_warehouse_graph

    warehouse_graph = create_warehouse_graph()

    tms_routes = fetch_real_time_routes(warehouse_graph, mes_workload=15)
    print(" Real-Time Optimized Routes from TMS:")
    for route in tms_routes[:5]:
        print(route)

    task_updates = get_dynamic_task_updates(warehouse_graph, mes_workload=15)
    print(" Dynamic Task Updates from TMS:")
    for task in task_updates[:5]:
        print(task)
