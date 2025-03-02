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


def compute_target_values(warehouse_graph, data):
    targets = []
    for node in range(data.x.shape[0]):
        if node not in warehouse_graph.nodes:
            targets.append(0)
            continue
        
        congestion = get_congestion_level(warehouse_graph, node)
        priority = warehouse_graph.nodes[node].get("task_priority", "Medium")
        priority_weight = 1 if priority == "High" else (0.5 if priority == "Medium" else 0.2)

        connected_nodes = list(warehouse_graph.neighbors(node))
        if not connected_nodes:
            targets.append(0)
            continue

        total_task_energy = sum(compute_energy_cost(warehouse_graph, node, n) for n in connected_nodes) * len(connected_nodes)

        congestion_penalty = 2.5 if congestion > 0.5 else 1.0
        obstacle_penalty = 3.0 if warehouse_graph.nodes[node].get("type") == "obstacle" else 1.0

        target_value = priority_weight / (1 + total_task_energy * 0.5 + congestion * congestion_penalty + obstacle_penalty)

        targets.append(target_value)

    return torch.tensor(targets, dtype=torch.float32).view(-1, 1)


def validate_ggnn(model, data, warehouse_graph):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)

    sorted_nodes = torch.argsort(predictions.view(-1), descending=True).tolist()
    valid_path = [sorted_nodes[0]]

    for i in range(len(sorted_nodes) - 1):
        if warehouse_graph.has_edge(valid_path[-1], sorted_nodes[i + 1]):
            valid_path.append(sorted_nodes[i + 1])
        else:
            neighbors = list(warehouse_graph.neighbors(valid_path[-1]))
            if neighbors:
                closest_neighbor = min(neighbors, key=lambda n: predictions[n].item())
                valid_path.append(closest_neighbor)

    if len(valid_path) < 2:
        logger.warning("GGNN could not find a valid path.")
        return []

    return valid_path


def train_ggnn(model, data, warehouse_graph, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loss_fn = nn.MSELoss()
    performance_log = []
    torch.manual_seed(42)  

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        target = compute_target_values(warehouse_graph, data)

        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        performance_log.append(loss.item())

        if epoch % 500 == 0:
            completion_rate = validate_ggnn(model, data, warehouse_graph)
            logger.info(f" Epoch {epoch}, Loss: {loss.item():.6f}, Completion Rate: {len(completion_rate) / len(data.x) * 100:.2f}%")

    with open("ggnn_performance_log.txt", "w") as log_file:
        log_file.write("\n".join([str(val) for val in performance_log]))


if __name__ == "__main__":
    warehouse_graph = create_warehouse_graph()
    pyg_data = convert_to_pyg_data(warehouse_graph)
    model = GGNN(num_features=pyg_data.x.shape[1], hidden_dim=32, num_layers=4)

    train_ggnn(model, pyg_data, warehouse_graph)
    final_path = validate_ggnn(model, pyg_data, warehouse_graph)

    if final_path:
        logger.info(f"Final GGNN path found: {final_path}")
    else:
        logger.warning("No valid GGNN path found.")

