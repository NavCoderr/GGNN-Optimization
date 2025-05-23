import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

# Transition tuple for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    """
    Feed-forward net mapping one-hot node states to Q-values for each node.
    """
    def __init__(self, input_dim: int, hidden_dim: int=128, output_dim: int=None):
        super().__init__()
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
    episodes: int=5000,
    batch_size: int=64,
    lr: float=1e-3,
    gamma: float=0.95,
    epsilon_start: float=1.0,
    epsilon_end: float=0.01,
    epsilon_decay: float=0.995,
    replay_size: int=10000,
    target_update: int=100
):
    """
    Train a DQN agent on graph G.
    - episodes, batch_size, lr, gamma from paper.
    - epsilon schedule: start→end with decay.
    - target_update: copy policy→target every target_update episodes.
    Returns the trained policy network.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = G.number_of_nodes()

    # networks and optimizer
    policy_net = DQN(input_dim=N, hidden_dim=128, output_dim=N).to(device)
    target_net = DQN(input_dim=N, hidden_dim=128, output_dim=N).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # replay memory
    memory = deque(maxlen=replay_size)

    epsilon = epsilon_start
    for ep in range(1, episodes+1):
        # update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # random start/goal
        start = random.randrange(N)
        goal  = random.randrange(N)
        while goal == start:
            goal = random.randrange(N)

        state = start
        for _ in range(N*2):
            # one-hot state
            state_vec = torch.zeros(N, device=device)
            state_vec[state] = 1.0

            # ε-greedy action
            if random.random() < epsilon:
                neighbors = list(G.neighbors(state)) or [state]
                action = random.choice(neighbors)
            else:
                with torch.no_grad():
                    q_vals = policy_net(state_vec.unsqueeze(0)).squeeze(0)
                    neighbors = list(G.neighbors(state)) or [state]
                    # pick neighbor with max Q
                    q_nbrs = q_vals[neighbors]
                    action = neighbors[q_nbrs.argmax().item()]

            # reward
            reward = 1.0 if action == goal else -0.01
            done = (action == goal)

            # next state vector
            next_vec = torch.zeros(N, device=device)
            next_vec[action] = 1.0

            # store transition
            memory.append(Transition(state_vec.unsqueeze(0), action, reward,
                                     next_vec.unsqueeze(0), done))
            state = action
            if done:
                break

            # learning step
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                batch = Transition(*zip(*batch))

                # prepare batches
                state_batch = torch.cat(batch.state)
                action_batch = torch.tensor(batch.action, device=device)
                reward_batch = torch.tensor(batch.reward, device=device)
                next_batch = torch.cat(batch.next_state)
                done_batch = torch.tensor(batch.done, device=device)

                # Q(s,a)
                q_values = policy_net(state_batch)
                q_s_a = q_values.gather(1, action_batch.view(-1,1)).squeeze(1)

                # Q_target = r + γ max_a' Q_target(s', a')
                with torch.no_grad():
                    q_next = target_net(next_batch)
                    q_next_max, _ = q_next.max(dim=1)
                    q_next_max[done_batch] = 0.0
                target = reward_batch + gamma * q_next_max

                loss = nn.functional.mse_loss(q_s_a, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # update target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net
