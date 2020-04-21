import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, not_done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, not_done))

    def sample(self, batch_size):
        state, action, reward, next_state, not_done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(np.concatenate(state), dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(np.concatenate(next_state), dtype=torch.float32, device=device),
            torch.tensor(not_done, dtype=torch.bool, device=device)
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def __init__(self, observation_dim, n_actions, hidden_size=128):
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float, device=device).view(1, -1)
            q_value = self.forward(state)
            action = q_value.max(1)[1].squeeze()
        else:
            action = torch.tensor(random.randrange(self.n_actions), device=device, dtype=torch.long)
        return action



