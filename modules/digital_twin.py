import random
import networkx as nx
from modules.warehouse import create_warehouse_graph
from modules.mes import generate_mes_tasks, fetch_dynamic_tasks
from modules.tms import fetch_real_time_routes
from modules.congestion_management import update_congestion, decay_congestion
from modules.energy_optimization import compute_energy_cost


def simulate_digital_twin(
    num_cycles=2000,
    num_agvs=50,
    experiment_type="dynamic",
    seed=42
):
    """
    Simulate a digital twin of the warehouse for AGV movements.
    - num_cycles: number of time steps to run (e.g., 2000)
    - num_agvs: fleet size (e.g., 50)
    - experiment_type: "static", "dynamic", "obstacle", or "scalability"
    - seed: random seed for reproducibility
    Returns:
      dict with keys: warehouse_graph, mes_tasks, tms_routes, agvs
    """
    # Reproducibility
    random.seed(seed)

    # 1) Create warehouse graph
    warehouse_graph = create_warehouse_graph()
    valid_nodes = list(warehouse_graph.nodes)
    if len(valid_nodes) < 2:
        raise ValueError("Not enough nodes generated in warehouse_graph.")

    # 2) Fetch MES tasks
    if experiment_type == "static":
        mes_tasks = generate_mes_tasks(
            num_tasks=num_agvs,
            valid_nodes=valid_nodes,
            warehouse_graph=warehouse_graph,
            seed=seed
        )
    else:
        mes_tasks = fetch_dynamic_tasks(
            current_workload=num_agvs,
            warehouse_graph=warehouse_graph,
            seed=seed
        )

    # 3) Fetch TMS routes
    tms_routes = fetch_real_time_routes(
        warehouse_graph,
        mes_workload=num_agvs,
        num_routes=num_agvs
    )

    # 4) Initialize AGV fleet state
    agvs = []
    for i in range(num_agvs):
        agvs.append({
            "id": i,
            "position": random.choice(valid_nodes),
            "battery": 100.0,
            "task": None,
            "charging": False,
            "move_count": 0,
            "energy_consumed": 0.0,
            "completed_tasks": 0,
            "reroute_count": 0,
            "waiting_cycles": 0
        })

    # 5) Main simulation loop
    for cycle in range(num_cycles):
        for agv in agvs:
            # Recharge if at charging station
            if agv["charging"]:
                agv["battery"] = min(100.0, agv["battery"] + 10.0)
                if agv["battery"] >= 100.0:
                    agv["charging"] = False
                continue

            # Assign new task if none
            if agv["task"] is None:
                if experiment_type == "static":
                    start, end = random.sample(valid_nodes, 2)
                else:
                    dt = fetch_dynamic_tasks(
                        current_workload=num_agvs,
                        warehouse_graph=warehouse_graph,
                        seed=seed + cycle + agv["id"]
                    )
                    t0 = dt[0]
                    start = t0[0] if isinstance(t0, tuple) else t0["start_node"]
                    end   = t0[1] if isinstance(t0, tuple) else t0["end_node"]
                agv["task"] = (start, end)

            # Execute current task
            start, end = agv["task"]
            if agv["position"] != end:
                try:
                    path = nx.shortest_path(
                        warehouse_graph,
                        source=agv["position"],
                        target=end,
                        weight="weight"
                    )
                    if len(path) > 1:
                        next_pos = path[1]
                        cost = compute_energy_cost(
                            warehouse_graph,
                            agv["position"],
                            next_pos
                        )
                        # Update AGV state
                        agv["position"] = next_pos
                        agv["move_count"] += 1
                        agv["energy_consumed"] += cost
                        agv["battery"] = max(0.0, agv["battery"] - cost)
                        update_congestion(warehouse_graph, next_pos)
                    else:
                        agv["waiting_cycles"] += 1
                except nx.NetworkXNoPath:
                    agv["reroute_count"] += 1
                    agv["task"] = None
            else:
                # Task completed
                agv["completed_tasks"] += 1
                agv["task"] = None
                update_congestion(warehouse_graph, agv["position"])

            # If battery low, go to charging station
            if agv["battery"] < 20.0:
                stations = [n for n in valid_nodes if warehouse_graph.nodes[n].get("type") == "charging_station"]
                if stations:
                    try:
                        nearest = min(
                            stations,
                            key=lambda s: nx.shortest_path_length(
                                warehouse_graph,
                                source=agv["position"],
                                target=s,
                                weight="weight"
                            )
                        )
                        agv["position"] = nearest
                        agv["charging"] = True
                    except nx.NetworkXNoPath:
                        pass

        # Decay congestion globally
        decay_congestion(warehouse_graph)

    return {
        "warehouse_graph": warehouse_graph,
        "mes_tasks": mes_tasks,
        "tms_routes": tms_routes,
        "agvs": agvs
    }
