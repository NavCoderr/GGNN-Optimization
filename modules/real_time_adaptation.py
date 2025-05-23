import random
import networkx as nx
from loguru import logger

def adapt_to_real_time_conditions(G, cycle, seed=None):
    """
    Every 50 cycles, randomly introduce or clear an obstacle on one warehouse edge.
    Sets G.edges[u,v]['blocked'] = True/False and logs the action.
    Returns a list of dicts describing what changed this cycle.
    """
    if seed is not None:
        random.seed(seed)

    events = []
    # initialize storage for blocked edges
    if 'blocked_edges' not in G.graph:
        G.graph['blocked_edges'] = set()

    # only act every 50 cycles
    if cycle % 50 == 0:
        # pick a random edge
        u, v = random.choice(list(G.edges))
        currently_blocked = (u, v) in G.graph['blocked_edges']

        if currently_blocked:
            # clear obstacle
            G.graph['blocked_edges'].remove((u, v))
            G.edges[u, v]['blocked'] = False
            events.append({'action': 'cleared', 'edge': (u, v), 'cycle': cycle})
            logger.info(f"Cleared obstacle on edge {(u, v)} at cycle {cycle}")
        else:
            # introduce obstacle
            G.graph['blocked_edges'].add((u, v))
            G.edges[u, v]['blocked'] = True
            events.append({'action': 'introduced', 'edge': (u, v), 'cycle': cycle})
            logger.info(f"Introduced obstacle on edge {(u, v)} at cycle {cycle}")

    return events