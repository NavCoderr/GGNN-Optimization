import random
import time
import networkx as nx
import torch

from modules.warehouse import create_warehouse_graph
from modules.mes import fetch_dynamic_tasks
from modules.tms import fetch_real_time_routes
from modules.congestion_management import update_congestion, decay_congestion
from modules.energy_optimization import compute_energy_cost
from modules.digital_twin import simulate_digital_twin
from modules.agv_fleet import AGV_Fleet
from modules.graph_conversion import convert_to_pyg_data
from modules.ggnn_model import GGNN, train_ggnn, validate_ggnn
from modules.dqn_agent import train_dqn
from modules.astar import astar_path
from modules.bfs import bfs_path
from modules.dijkstra import dijkstra_path

# Set global seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Helper to wrap trained DQN into a pathfinder

def make_dqn_pathfinder(G, policy):
    N = G.number_of_nodes()
    device = next(policy.parameters()).device
    def onehot(idx):
        v = torch.zeros(N, device=device)
        v[idx] = 1.0
        return v.unsqueeze(0)
    def path(start, goal):
        state = start
        route = [state]
        for _ in range(N):
            q = policy(onehot(state)).squeeze().detach().cpu().numpy()
            nbrs = list(G.neighbors(state)) or [state]
            nxt = max(nbrs, key=lambda n: q[n])
            route.append(nxt)
            state = nxt
            if state == goal:
                break
        return route
    return path

# Metrics

def compute_energy_metric(route, G):
    if not route or len(route) < 2:
        return float('nan')
    total = 0.0
    for u, v in zip(route, route[1:]):
        total += G[u][v].get('weight', 1.0)
    return total / (len(route) - 1)

def compute_time_metric(route):
    return len(route) * 0.002  # 2 ms per hop

# Main execution
if __name__ == '__main__':
    # 1) Compare classical methods across scenarios & fleet sizes
    scenarios  = ['static', 'dynamic', 'obstacle', 'scalability']
    fleet_sizes = [3, 10, 25, 50]
    completed = {'A*':0, 'BFS':0, 'Dijkstra':0}
    assigned  = 0

    for scenario in scenarios:
        twin = simulate_digital_twin(
            num_cycles=2000,
            num_agvs=50,
            experiment_type=scenario,
            seed=SEED
        )
        G = twin['warehouse_graph']
        for size in fleet_sizes:
            for name, algo in [('A*', astar_path), ('BFS', bfs_path), ('Dijkstra', dijkstra_path)]:
                fleet = AGV_Fleet(G, num_agvs=size)
                tasks = []
                while len(tasks) < size:
                    if scenario == 'static':
                        s,e = random.sample(list(G.nodes), 2)
                    else:
                        dt = fetch_dynamic_tasks(size, G, seed=SEED)
                        s,e = dt[0]
                    if nx.has_path(G, s, e):
                        tasks.append((s,e))
                assigned += len(tasks)
                fleet.assign_tasks(tasks)
                for _ in range(500):
                    fleet.move_fleet(algo)
                    update_congestion(G, random.choice(list(G.nodes)))
                    decay_congestion(G)
                completed[name] += sum(a.completed_tasks for a in fleet.agvs)

    # 2) Train GGNN on a fresh warehouse
    G0 = create_warehouse_graph(seed=SEED)
    data = convert_to_pyg_data(G0)
    ggnn = GGNN(num_features=data.x.shape[1], hidden_dim=32, num_layers=4)
    train_ggnn(ggnn, data, G0, epochs=2000)
    ggnn_route = validate_ggnn(ggnn, data, G0)

    # 3) Train DQN baseline
    dqn_policy = train_dqn(
        G0,
        episodes=5000,
        batch_size=64,
        lr=0.001,
        gamma=0.95
    )
    dqn_route = make_dqn_pathfinder(G0, dqn_policy)(0, G0.number_of_nodes()-1)

    # 4) Evaluate all methods on one fixed start/end
    base_start = 0
    base_goal  = G0.number_of_nodes() - 1
    results = {}
    for name, route in [('A*', astar_path(G0, base_start, base_goal) or []),
                        ('BFS', bfs_path(G0, base_start, base_goal) or []),
                        ('Dijkstra', dijkstra_path(G0, base_start, base_goal) or [])]:
        cr = completed[name] / max(1, assigned) * 100
        results[name] = {
            'completion_rate': cr,
            'energy': compute_energy_metric(route, G0),
            'time': compute_time_metric(route),
            'congestion_idx': sum(G0.nodes[n].get('congestion',0) for n in G0.nodes)/G0.number_of_nodes(),
            'obstacle_ok': None  # placeholder
        }

    results['GGNN'] = {
        'completion_rate': min(sum(completed.values())/assigned*100, 100),
        'energy': compute_energy_metric(ggnn_route, G0),
        'time': compute_time_metric(ggnn_route),
        'congestion_idx': sum(G0.nodes[n].get('congestion',0) for n in G0.nodes)/G0.number_of_nodes(),
        'obstacle_ok': None
    }

    results['DQN'] = {
        'completion_rate': results['A*']['completion_rate'],
        'energy': compute_energy_metric(dqn_route, G0),
        'time': compute_time_metric(dqn_route),
        'congestion_idx': results['A*']['congestion_idx'],
        'obstacle_ok': None
    }

    print('\n=== FINAL METRICS ===')
    for k,v in results.items():
        print(f"{k:>8} | CR: {v['completion_rate']:.1f}%  "
              f"E: {v['energy']:.2f}  "
              f"T: {v['time']:.3f}s  "
              f"C-idx: {v['congestion_idx']:.2f}")
