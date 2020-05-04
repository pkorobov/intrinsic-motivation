import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), torch.Tensor([reward]), torch.Tensor([done])
        self.memory.append(transition)
        if len(self.memory) > self.capacity: del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        batch_state = torch.cat(batch_state)
        batch_action = torch.LongTensor(batch_action)
        batch_next_state = torch.cat(batch_next_state)
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)

        return batch_state, batch_action, batch_next_state, batch_reward.unsqueeze(1), batch_done.unsqueeze(1)

    def __len__(self):
        return len(self.memory)


class QuantileRegressionDQN(nn.Module):

    def __init__(self, observation_dim, n_actions, hidden_size=256, num_quant=2):
        nn.Module.__init__(self)

        self.num_quant = num_quant
        self.num_actions = n_actions

        self.layer1 = nn.Linear(observation_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions * num_quant)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x.view(-1, self.num_actions, self.num_quant)

    def act(self, state, eps):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state])
        action = torch.randint(0, 2, (1,))
        if random.random() > eps:
            action = self.forward(state).mean(2).max(1)[1]
        return int(action)



