# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from torch_geometric.nn import GatedGraphConv
from modules.graph_conversion import convert_to_pyg_data
from modules.warehouse import create_warehouse_graph
from modules.energy_optimization import compute_energy_cost
from modules.congestion_management import get_congestion_level
from loguru import logger
import networkx as nx

class GGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers):
        super().__init__()
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, edge_index):
        x = self.ggnn(x, edge_index)
        return torch.clamp(self.fc(x), min=0.01, max=1.0)

def compute_target_values(graph, data):
    targets = []
    N = data.x.shape[0]
    for node in range(N):
        if node not in graph.nodes:
            targets.append(0.0)
            continue

        priority = graph.nodes[node].get("priority", "medium").lower()
        pw = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(priority, 0.5)

        neighbors = list(graph.neighbors(node))
        if not neighbors:
            targets.append(0.0)
            continue

        total_energy = 0.0
        total_cong = 0.0
        total_urg = 0.0
        blocked_count = 0
        for nbr in neighbors:
            total_energy += compute_energy_cost(graph, node, nbr)
            total_cong   += get_congestion_level(graph, node, nbr)
            total_urg    += graph.edges[node, nbr].get("urgency", 0.0)
            if graph.edges[node, nbr].get("blocked", False):
                blocked_count += 1

        avg_cong = total_cong / len(neighbors)
        value = pw / (1.0 + 0.5 * total_energy + 2.0 * avg_cong + 1.0 * total_urg + 5.0 * blocked_count)
        targets.append(value)

    return torch.tensor(targets, dtype=torch.float32).view(-1, 1)

def validate_ggnn(model, data, graph):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).view(-1)
    order = preds.argsort(descending=True).tolist()
    if not order:
        logger.warning("GGNN produced no predictions")
        return []

    path = [order[0]]
    for idx in order[1:]:
        prev = path[-1]
        if graph.has_edge(prev, idx):
            path.append(idx)
        else:
            nbrs = list(graph.neighbors(prev))
            if nbrs:
                best = max(nbrs, key=lambda n: preds[n])
                path.append(best)

    if len(path) < 2:
        logger.warning("GGNN could not find valid path")
        return []
    return path

def train_ggnn(model, data, graph, epochs=2000, lr=1e-3, weight_decay=1e-5, step_size=300, gamma=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.MSELoss()
    log = []
    torch.manual_seed(42)

    for e in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        tgt = compute_target_values(graph, data)
        loss = loss_fn(out, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        log.append(loss.item())

        if e % 500 == 0:
            path = validate_ggnn(model, data, graph)
            rate = len(path) / data.x.shape[0] * 100 if data.x.shape[0] else 0
            logger.info(f"Epoch {e}/{epochs} loss={loss.item():.6f} rate={rate:.2f}%")

    with open("ggnn_performance_log.txt", "w") as f:
        f.write("\n".join(str(v) for v in log))

def train_ggnn_on_movements(model, movement_csv_path, graph, epochs=100):
    df = pd.read_csv(movement_csv_path)
    samples = list(zip(df['from_node'], df['to_node'], df['energy_used']))
    print(f"[INFO] Training GGNN on {len(samples)} movement records")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for u, v, y in samples:
            x = torch.zeros(graph.number_of_nodes(), model.fc.in_features)
            x[u] = 1.0
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            pred = model(x, edge_index)[v].view(1)
            target = torch.tensor([y], dtype=torch.float32)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs} - Loss: {total_loss:.4f}")

if __name__ == "__main__":
    G = create_warehouse_graph(seed=42)
    data = convert_to_pyg_data(G)
    model = GGNN(data.x.shape[1], hidden_dim=32, num_layers=4)
    train_ggnn(model, data, G)
    train_ggnn_on_movements(model, "movements.csv", G, epochs=100)
    final_path = validate_ggnn(model, data, G)
    if final_path:
        logger.info(f"Final GGNN path: {final_path}")
    else:
        logger.warning("No valid GGNN path found")
