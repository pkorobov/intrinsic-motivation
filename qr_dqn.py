import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantileRegressionDQN(nn.Module):

    def __init__(self, observation_dim, n_actions, hidden_size=256, num_quant=5):
        nn.Module.__init__(self)

        self.num_quant = num_quant
        self.num_actions = n_actions

        self.layer1 = nn.Linear(observation_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions * num_quant)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x.view(-1, self.num_actions, self.num_quant)

    def act(self, state, eps):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state], device=device)
        action = torch.randint(0, 2, (1,))
        if random.random() > eps:
            action = self.forward(state).mean(2).max(1)[1]
        return int(action)



