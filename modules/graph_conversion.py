import torch
from torch_geometric.data import Data

TASK_PRIORITY = {"Low":1,"Medium":2,"High":3}

def normalize(v, lo, hi):
    if hi-lo==0: return 0.0
    return (float(v)-lo)/(hi-lo)

def convert_to_pyg_data(graph):
    import networkx as nx
    nodes = list(graph.nodes)
    edge_index = torch.tensor(list(graph.edges),dtype=torch.long).t().contiguous()
    priority = [TASK_PRIORITY.get(graph.nodes[n].get("task_priority","Low"),1) for n in nodes]
    min_p, max_p = min(TASK_PRIORITY.values()), max(TASK_PRIORITY.values())
    priority = [normalize(p,min_p,max_p) for p in priority]
    cong = [normalize(graph.nodes[n].get("congestion",0),0,1) for n in nodes]
    degs = dict(graph.degree)
    min_d, max_d = min(degs.values()), max(degs.values())
    degree = [normalize(degs[n],min_d,max_d) for n in nodes]
    x = torch.tensor(list(zip(priority,cong,degree)),dtype=torch.float)
    return Data(x=x, edge_index=edge_index)
