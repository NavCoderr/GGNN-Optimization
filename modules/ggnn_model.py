import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.nn import GatedGraphConv
from modules.graph_conversion import convert_to_pyg_data
from modules.warehouse import create_warehouse_graph
from modules.energy_optimization import compute_energy_cost
from modules.congestion_management import get_congestion_level
from loguru import logger
import networkx as nx

class GGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers):
        super(GGNN, self).__init__()
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, edge_index):
        x = self.ggnn(x, edge_index)
        return torch.clamp(self.fc(x), min=0.01, max=1.0)

def compute_target_values(graph, data):
    targets = []
    for node in range(data.x.shape[0]):
        if node not in graph.nodes:
            targets.append(0.0)
            continue
        congestion = get_congestion_level(graph, node)
        priority = graph.nodes[node].get("task_priority", "Medium")
        pw = 1.0 if priority == "High" else (0.5 if priority == "Medium" else 0.2)
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            targets.append(0.0)
            continue
        total_energy = sum(compute_energy_cost(graph, node, n) for n in neighbors) * len(neighbors)
        cp = 2.5 if congestion > 0.5 else 1.0
        op = 3.0 if graph.nodes[node].get("type") == "obstacle" else 1.0
        value = pw / (1 + total_energy * 0.5 + congestion * cp + op)
        targets.append(value)
    return torch.tensor(targets, dtype=torch.float32).view(-1, 1)

def validate_ggnn(model, data, graph):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
    order = torch.argsort(preds.view(-1), descending=True).tolist()
    path = [order[0]] if order else []
    for idx in order[1:]:
        if graph.has_edge(path[-1], idx):
            path.append(idx)
        else:
            nbrs = list(graph.neighbors(path[-1]))
            if nbrs:
                best = min(nbrs, key=lambda n: preds[n].item())
                path.append(best)
    if len(path) < 2:
        logger.warning("GGNN could not find valid path")
        return []
    return path

def train_ggnn(model, data, graph, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_fn = nn.MSELoss()
    log = []
    torch.manual_seed(42)
    for e in range(1, epochs+1):
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

if __name__ == "__main__":
    G = create_warehouse_graph(seed=42)
    data = convert_to_pyg_data(G)
    model = GGNN(data.x.shape[1], hidden_dim=32, num_layers=4)
    train_ggnn(model, data, G)
    final_path = validate_ggnn(model, data, G)
    if final_path:
        logger.info(f"Final GGNN path: {final_path}")
    else:
        logger.warning("No valid GGNN path found")
