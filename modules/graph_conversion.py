import torch
from torch_geometric.data import Data

TASK_PRIORITY_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

def normalize(value, min_value, max_value):
    
    try:
        value = float(value)
        min_value = float(min_value)
        max_value = float(max_value)

        if max_value - min_value == 0:
            return 0   
        
        return (value - min_value) / (max_value - min_value + 1e-6)
    
    except (ValueError, TypeError) as e:
        print(f" Normalization Error: {e}, value={value}, min={min_value}, max={max_value}")
        return 0   

def convert_to_pyg_data(graph):
   
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Task priority mapping
    task_priority = [
        TASK_PRIORITY_MAP.get(graph.nodes[n].get("task_priority", "Low"), 1)
        for n in graph.nodes
    ]

 
    min_priority, max_priority = min(TASK_PRIORITY_MAP.values()), max(TASK_PRIORITY_MAP.values())
    task_priority = [normalize(t, min_priority, max_priority) for t in task_priority]

    # Add congestion feature
    congestion_values = [normalize(graph.nodes[n].get("congestion", 0), 0, 1) for n in graph.nodes]

    # Add node degree as an additional feature
    node_degree = [normalize(graph.degree[n], min(dict(graph.degree).values()), max(dict(graph.degree).values())) for n in graph.nodes]

    # Combine into feature tensor
    x = torch.tensor(list(zip(task_priority, congestion_values, node_degree)), dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

