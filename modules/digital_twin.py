# -*- coding: utf-8 -*-

import random
import networkx as nx
import pandas as pd
from modules.warehouse import create_warehouse_graph
from modules.mes import generate_mes_tasks, fetch_dynamic_tasks
from modules.tms import fetch_real_time_routes
from modules.congestion_management import update_congestion, decay_congestion
from modules.energy_optimization import compute_energy_cost
from modules.real_time_adaptation import adapt_to_real_time_conditions
from modules.agv_fleet import AGV_Fleet

def update_edge_urgency(G, fleet):
    # reset urgencies
    for u, v in G.edges:
        G.edges[u, v]['urgency'] = 0
    # increment along each AGV's current path
    for agv in fleet.agvs:
        task = agv.current_task
        if task:
            try:
                path = nx.shortest_path(G, task['start'], task['goal'], weight='weight')
                for a, b in zip(path, path[1:]):
                    G.edges[a, b]['urgency'] += 1
            except nx.NetworkXNoPath:
                pass

def simulate_digital_twin(
    num_cycles=2000,
    num_agvs=50,
    experiment_type="dynamic",
    seed=42,
    export_movements=False,
    export_path="movements.csv"
):
    random.seed(seed)

    # 1) build warehouse graph & init urgencies
    G = create_warehouse_graph(seed=seed)
    for u, v in G.edges:
        G.edges[u, v]['urgency'] = 0

    # 2) init fleet
    fleet = AGV_Fleet(G, num_agvs)

    # 3) initial MES tasks
    if experiment_type == "static":
        initial = generate_mes_tasks(
            num_tasks=num_agvs,
            valid_nodes=list(G.nodes),
            warehouse_graph=G,
            seed=seed
        )
    else:
        initial = fetch_dynamic_tasks(
            current_workload=num_agvs,
            warehouse_graph=G,
            seed=seed
        )

    tasks = []
    for t in initial:
        if isinstance(t, dict):
            tasks.append((t["start_node"], t["end_node"], t.get("task_id")))
        else:
            tasks.append((t[0], t[1], None))
    fleet.assign_tasks(tasks)

    # 4) main simulation loop
    tms_routes = None
    for cycle in range(num_cycles):
        # if exporting, reassign all AGVs each cycle
        if export_movements:
            batch = generate_mes_tasks(
                num_tasks=num_agvs,
                valid_nodes=list(G.nodes),
                warehouse_graph=G,
                seed=seed + cycle
            )
            tasks = [
                (t["start_node"], t["end_node"], t.get("task_id")) if isinstance(t, dict)
                else (t[0], t[1], None)
                for t in batch
            ]
            fleet.assign_tasks(tasks)

        # normal dynamic mode (non-static & not exporting)
        elif experiment_type != "static":
            dt = fetch_dynamic_tasks(
                current_workload=num_agvs,
                warehouse_graph=G,
                seed=seed + cycle
            )
            new_tasks = []
            for t in dt:
                if isinstance(t, dict):
                    new_tasks.append((t["start_node"], t["end_node"], t.get("task_id")))
                else:
                    new_tasks.append((t[0], t[1], None))
            fleet.assign_tasks(new_tasks)

        # fetch TMS routes
        tms_routes = fetch_real_time_routes(
            warehouse_graph=G,
            mes_workload=num_agvs,
            num_routes=num_agvs
        )

        # real-time adaptation & urgency updates
        adapt_to_real_time_conditions(G, cycle, seed + cycle)
        update_edge_urgency(G, fleet)

        # choose path algorithm
        if experiment_type == "obstacle":
            algo = nx.astar_path
        elif experiment_type == "scalability":
            algo = nx.dijkstra_path
        else:
            algo = nx.shortest_path

        # move AGVs
        fleet.move_fleet(
            path_algorithm=algo,
            current_cycle=cycle,
            current_workload=num_agvs,
            current_route=tms_routes
        )

        # decay congestion
        decay_congestion(G)

    # 5) export movements
    if export_movements:
        all_moves = []
        for agv in fleet.agvs:
            all_moves.extend(agv.move_history)
        df = pd.DataFrame(all_moves)
        df.to_csv(export_path, index=False)
        print(f"[INFO] Exported {len(df)} movements to {export_path}")

    return {
        "warehouse_graph": G,
        "mes_tasks":       initial,
        "tms_routes":      tms_routes,
        "agv_fleet":       fleet
    }
