import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

# Transition tuple for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    """
    Simple feed-forward network mapping one-hot states to Q-values for each node.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = None):
        super(DQN, self).__init__()
        # If output_dim not set, assume same as input_dim
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dqn(
    G,
    episodes: int = 500,
    batch_size: int = 64,
    lr: float = 0.001,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    replay_size: int = 10000,
    hidden_dim: int = 128
):
    """
    Train a DQN agent on the graph G, treating nodes as states and edges as valid actions.
    Returns a trained policy network.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = G.number_of_nodes()

    # Initialize networks and optimizer
    policy_net = DQN(input_dim=N, hidden_dim=hidden_dim, output_dim=N).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Replay memory
    memory = deque(maxlen=replay_size)

    for ep in range(episodes):
        # Epsilon-greedy schedule
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** ep))
        # Random start/goal
        start = random.randrange(N)
        goal = random.randrange(N)
        while goal == start:
            goal = random.randrange(N)

        state = start
        for step in range(N * 2):  # limit steps
            # One-hot encode state
            state_vec = torch.zeros(N, device=device)
            state_vec[state] = 1.0

            # Choose action
            if random.random() < epsilon:
                neighbors = list(G.neighbors(state)) or [state]
                action = random.choice(neighbors)
            else:
                with torch.no_grad():
                    q_vals = policy_net(state_vec.unsqueeze(0)).squeeze(0)
                    # Mask invalid actions
                    neighbors = list(G.neighbors(state)) or [state]
                    # Select highest Q among neighbors
                    valid_q = q_vals[neighbors]
                    idx = torch.argmax(valid_q).item()
                    action = neighbors[idx]

            # Reward: +1 if goal reached, else small step penalty
            reward = 1.0 if action == goal else -0.01
            done = (action == goal)

            # One-hot encode next state
            next_vec = torch.zeros(N, device=device)
            next_vec[action] = 1.0

            # Store transition
            memory.append(Transition(state_vec.unsqueeze(0), action, reward, next_vec.unsqueeze(0), done))
            state = action
            if done:
                break

            # Learn from replay
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                batch = Transition(*zip(*batch))

                state_batch = torch.cat(batch.state)
                action_batch = torch.tensor(batch.action, device=device)
                reward_batch = torch.tensor(batch.reward, device=device)
                next_batch = torch.cat(batch.next_state)
                done_batch = torch.tensor(batch.done, dtype=torch.bool, device=device)

                # Q(s, a)
                q_values = policy_net(state_batch)
                q_s_a = q_values.gather(1, action_batch.view(-1, 1)).squeeze(1)

                # Q_max(s') for next states
                with torch.no_grad():
                    q_next = policy_net(next_batch)
                    q_next_max, _ = q_next.max(dim=1)
                    q_next_max[done_batch] = 0.0

                # Target = r + gamma * max Q(s')
                target = reward_batch + gamma * q_next_max

                loss = nn.functional.mse_loss(q_s_a, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return policy_net