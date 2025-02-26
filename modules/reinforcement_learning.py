import numpy as np
import random
import networkx as nx
from modules.warehouse import create_warehouse_graph
from modules.congestion_management import get_congestion_level
from modules.energy_optimization import compute_energy_cost

class RLAgent:
    def __init__(self, G, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.G = G   
        self.nodes = list(G.nodes)
        self.q_table = {node: {neighbor: 0 for neighbor in G.neighbors(node)} for node in G.nodes}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.loss_history = []   
        self.visited_nodes = set()

    def choose_action(self, state):
        """Selects action based on exploration-exploitation tradeoff, avoiding loops."""
        neighbors = list(self.G.neighbors(state))
        if not neighbors:
            return state   

        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(neighbors)  
        
        # Select highest Q-value, avoiding already visited nodes
        best_action = max(neighbors, key=lambda x: self.q_table[state].get(x, 0))
        if best_action in self.visited_nodes:  
            return random.choice(neighbors)
        return best_action

    def update_q_table(self, state, action, reward, next_state):
        
        if next_state in self.q_table and self.q_table[next_state]:
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get, default=0)
            self.q_table[state][action] += self.learning_rate * (
                reward + self.discount_factor * self.q_table[next_state].get(best_next_action, 0) - self.q_table[state][action]
            )

    def train(self, episodes=1000):
        
        for episode in range(episodes):
            state = random.choice(self.nodes)
            total_reward = 0  
            self.visited_nodes.clear()  

            for step in range(100):
                neighbors = list(self.G.neighbors(state))
                if not neighbors:
                    break  

                next_state = self.choose_action(state)
                self.visited_nodes.add(next_state)   

                 
                congestion_penalty = get_congestion_level(self.G, next_state) * -5
                energy_cost_penalty = compute_energy_cost(self.G, state, next_state) * -2
                task_urgency_reward = 10 if self.G.nodes[next_state].get("task_priority", "Low") == "High" else 0

              
                loop_penalty = -5 if next_state in self.visited_nodes else 0

                reward = congestion_penalty + energy_cost_penalty + task_urgency_reward + loop_penalty

                self.update_q_table(state, next_state, reward, next_state)
                total_reward += reward
                state = next_state

            self.loss_history.append(total_reward)
            self.exploration_rate *= 0.99  # Gradually reduce exploration

            if episode % 10 == 0:
                print(f" Epoch {episode}: Total Reward {total_reward:.2f}")

        print(" RL Training Complete.")

    def get_best_path(self, start, goal):
         
        path = [start]
        current_node = start

        while current_node != goal and current_node in self.q_table:
            next_nodes = self.q_table[current_node]
            if not next_nodes:
                break  

        
            next_node = max(next_nodes, key=next_nodes.get)
            if next_node in path:  # Prevent looping
                break  
            path.append(next_node)
            current_node = next_node

        return path if current_node == goal else None

if __name__ == "__main__":
    warehouse_graph = create_warehouse_graph()
    agent = RLAgent(warehouse_graph)
    agent.train(episodes=1000)
    
    # Test best path after training
    start_node, goal_node = random.sample(agent.nodes, 2)
    best_path = agent.get_best_path(start_node, goal_node)
    
    if best_path:
        print(f" RL Best Path from {start_node} to {goal_node}: {best_path}")
    else:
        print(f" No valid RL path found from {start_node} to {goal_node}.")
    
    print(" Trained Reinforcement Learning Agent for AGV Routing.")
