import random
import networkx as nx

def update_congestion(
    G: nx.Graph,
    u: int,
    v: int,
    init_range=(0.1, 0.3),
    increase_range=(0.05, 0.1),
    cap: float = 1.0
) -> None:
    """
    Update congestion on edge (u, v). If no 'congestion' attribute exists, initialize it
    to a random value in init_range; otherwise increase it by a random value in
    increase_range, capped at `cap`.
    """
    if not G.has_edge(u, v):
        return
    data = G.edges[u, v]
    if 'congestion' not in data:
        data['congestion'] = random.uniform(*init_range)
    data['congestion'] = min(cap, data['congestion'] + random.uniform(*increase_range))

def get_congestion_level(
    G: nx.Graph,
    u: int,
    v: int
) -> float:
    """
    Return the congestion level on edge (u, v), or 0.0 if no such edge or no attribute.
    """
    if not G.has_edge(u, v):
        return 0.0
    return float(G.edges[u, v].get('congestion', 0.0))

def decay_congestion(
    G: nx.Graph,
    decay_factor: float = 0.9,
    threshold: float = 0.01
) -> None:
    """
    Apply multiplicative decay to every edge's congestion each cycle.
    If it falls below `threshold`, reset to zero.
    """
    for u, v, data in G.edges(data=True):
        if 'congestion' in data:
            data['congestion'] = max(0.0, data['congestion'] * decay_factor)
            if data['congestion'] < threshold:
                data['congestion'] = 0.0
