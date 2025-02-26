import random
import networkx as nx
from modules.warehouse import create_warehouse_graph
from modules.mes import generate_mes_tasks, fetch_dynamic_tasks
from modules.tms import fetch_real_time_routes
from modules.congestion_management import update_congestion
from modules.energy_optimization import compute_energy_cost

def simulate_digital_twin(num_cycles=500, num_agvs=50, experiment_type="dynamic"):
    warehouse_graph = create_warehouse_graph()
    valid_nodes = list(warehouse_graph.nodes)

    if len(valid_nodes) < 2:
        raise ValueError("Digital Twin Failed: No valid nodes available in the warehouse graph!")

    if experiment_type == "static":
        mes_tasks = generate_mes_tasks(num_tasks=100, valid_nodes=valid_nodes, warehouse_graph=warehouse_graph)
    else:
        mes_tasks = fetch_dynamic_tasks(current_workload=10, warehouse_graph=warehouse_graph)

    tms_routes = fetch_real_time_routes(warehouse_graph, mes_workload=15)

    agvs = [
        {
            "id": i,
            "position": random.choice(valid_nodes),
            "battery": 100,
            "task": None,
            "charging": False
        }
        for i in range(num_agvs)
    ]

    for cycle in range(num_cycles):
        for agv in agvs:
            if agv["charging"]:
                agv["battery"] = min(100, agv["battery"] + 10)
                if agv["battery"] == 100:
                    agv["charging"] = False
                continue

            if not agv["task"]:
                if experiment_type == "static":
                    start, end = random.sample(valid_nodes, 2)
                    new_task = [(start, end)]
                else:
                    new_task = fetch_dynamic_tasks(current_workload=1, warehouse_graph=warehouse_graph)

                if not new_task:
                    continue

                first_task = new_task[0]
                if isinstance(first_task, tuple) and len(first_task) == 2:
                    agv["task"] = first_task
                elif isinstance(first_task, dict) and "start_node" in first_task and "end_node" in first_task:
                    agv["task"] = (first_task["start_node"], first_task["end_node"])
                else:
                    continue

            start, end = agv["task"]

            if agv["position"] != end:
                try:
                    path = nx.shortest_path(warehouse_graph, source=agv["position"], target=end, weight="weight")
                    if len(path) > 1:
                        agv["position"] = path[1]
                except nx.NetworkXNoPath:
                    agv["task"] = None
                continue

            update_congestion(warehouse_graph, agv["position"])
            energy_cost = compute_energy_cost(warehouse_graph, start, agv["position"])
            agv["battery"] = max(0, agv["battery"] - energy_cost)

            if agv["battery"] <= 5:
                charging_stations = [node for node in valid_nodes if warehouse_graph.nodes[node].get("type") == "charging_station"]
                if charging_stations:
                    agv["position"] = random.choice(charging_stations)
                    agv["charging"] = True
                else:
                    agv["task"] = None

    return {
        "warehouse_graph": warehouse_graph,
        "mes_tasks": mes_tasks,
        "tms_routes": tms_routes,
        "agvs": agvs
    }

if __name__ == "__main__":
    twin = simulate_digital_twin(num_cycles=500, num_agvs=50, experiment_type="dynamic")
    print(f"Digital Twin Initialized with {len(twin['agvs'])} AGVs.")

    for agv in twin["agvs"]:
        print(f"AGV {agv['id']} - Final Position: {agv['position']}, Battery: {agv['battery']:.2f}%, Charging: {agv['charging']}")
