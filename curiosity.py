import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ForwardModel:
    def __init__(self, state_dim, action_dim, device, hidden_dim=16):

        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU()),
        layers.append(nn.Linear(hidden_dim, hidden_dim)),
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers).to(device)
        self.optimizer = Adam(self.net.parameters(), lr=1e-4)
        self.device = device

    def intrinsic_reward(self, state, next_state, action):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action_ohe = torch.zeros(1, 3, device=self.device).scatter_(1, action, 1)

        return float(F.mse_loss(self.net(torch.cat([state, action_ohe.squeeze()])), next_state))

    def update(self, state, next_state, action):
        loss = F.mse_loss(self.net(torch.cat([state, action], dim=1)), next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class InverseModel:
    pass


class ICMModel:
    pass
