import random
import networkx as nx

def update_congestion(
    G: nx.DiGraph,
    node: int,
    init_range=(0.1, 0.3),
    increase_range=(0.05, 0.1),
    cap: float = 1.0
) -> None:
    if node not in G.nodes:
        return
    if "congestion" not in G.nodes[node]:
        G.nodes[node]["congestion"] = random.uniform(*init_range)
    G.nodes[node]["congestion"] = min(
        cap,
        G.nodes[node]["congestion"] + random.uniform(*increase_range)
    )

def get_congestion_level(
    G: nx.DiGraph,
    node: int
) -> float:
    if node not in G.nodes:
        return 0.0
    return float(G.nodes[node].get("congestion", 0.0))

def decay_congestion(
    G: nx.DiGraph,
    decay_factor: float = 0.9,
    threshold: float = 0.01
) -> None:
    for node, data in G.nodes(data=True):
        if "congestion" in data:
            data["congestion"] = max(
                0.0,
                data["congestion"] * decay_factor
            )
            if data["congestion"] < threshold:
                data["congestion"] = 0.0
