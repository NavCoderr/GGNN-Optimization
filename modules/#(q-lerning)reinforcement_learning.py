import random
import networkx as nx
import numpy as np
from loguru import logger
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

class RLAgent:
    def __init__(
        self,
        G: nx.DiGraph,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
        min_exploration: float = 0.01,
        exploration_decay: float = 0.995,
        seed: int = None
    ):
        """
        Simple Q-learning agent for AGV routing on warehouse_graph G.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.G = G
        self.nodes = list(G.nodes)
        # Initialize Q-table: Q[state][action] = value
        self.q_table = {
            state: {nbr: 0.0 for nbr in G.neighbors(state)}
            for state in self.nodes
        }
        self.lr = learning_rate
        self.df = discount_factor
        self.er = exploration_rate
        self.min_er = min_exploration
        self.er_decay = exploration_decay
        self.seed = seed

    def choose_action(self, state):
        nbrs = list(self.G.neighbors(state))
        if not nbrs:
            return None
        # ε‐greedy
        if random.random() < self.er:
            return random.choice(nbrs)
        # choose best available
        q_vals = self.q_table[state]
        max_q = max(q_vals.values())
        best = [n for n, q in q_vals.items() if q == max_q]
        return random.choice(best)

    def update_q(self, state, action, reward, next_state):
        if action is None:
            return
        old = self.q_table[state][action]
        # estimate of optimal future value
        future = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            future = max(self.q_table[next_state].values())
        # Q‐learning update
        self.q_table[state][action] = old + self.lr * (reward + self.df * future - old)

    def train(self, episodes: int = 500, max_steps: int = 100):
        """
        Train the agent for a number of episodes.
        """
        for ep in range(1, episodes + 1):
            state = random.choice(self.nodes)
            total_reward = 0.0

            for _ in range(max_steps):
                action = self.choose_action(state)
                if action is None:
                    break
                # compute reward
                cong_pen = -5 * get_congestion_level(self.G, action)
                energy_pen = -2 * compute_energy_cost(self.G, state, action)
                urgency = 10 if self.G.nodes[action].get("task_priority") == "High" else 0
                reward = cong_pen + energy_pen + urgency

                next_state = action
                self.update_q(state, action, reward, next_state)
                total_reward += reward
                state = next_state

            # decay exploration
            self.er = max(self.min_er, self.er * self.er_decay)

            if ep % 100 == 0 or ep == 1:
                logger.info(f"Episode {ep}/{episodes} — Total reward: {total_reward:.2f} — ε={self.er:.3f}")

        logger.info("RL training complete.")

    def get_best_path(self, start: int, goal: int, max_steps: int = None):
        """
        Extract path by following the highest‐Q actions until reaching goal or max_steps.
        """
        path = [start]
        current = start
        steps = 0
        max_steps = max_steps or len(self.nodes)

        while current != goal and steps < max_steps:
            nbrs = self.q_table.get(current)
            if not nbrs:
                break
            # pick best action
            best = max(nbrs, key=nbrs.get)
            if best in path:
                break
            path.append(best)
            current = best
            steps += 1

        return path if current == goal else None


if __name__ == "__main__":
    # Demo usage
    G = create_warehouse_graph(seed=42)
    agent = RLAgent(G, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, seed=42)
    agent.train(episodes=1000)

    start, goal = random.sample(agent.nodes, 2)
    best_path = agent.get_best_path(start, goal)
    if best_path:
        logger.info(f"Best RL path from {start} to {goal}: {best_path}")
    else:
        logger.warning(f"No RL path found from {start} to {goal}")
