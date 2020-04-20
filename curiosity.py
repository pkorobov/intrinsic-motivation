import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ForwardModel:
    def __init__(self, state_dim, n_actions, device, hidden_dim=16):

        layers = []
        layers.append(nn.Linear(state_dim + n_actions, hidden_dim))
        layers.append(nn.ReLU()),
        layers.append(nn.Linear(hidden_dim, hidden_dim)),
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.n_actions = n_actions
        self.net = nn.Sequential(*layers).to(device)
        self.optimizer = Adam(self.net.parameters(), lr=1e-4)
        self.device = device

    # add one hots here
    def intrinsic_reward(self, state, next_state, action):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action_ohe = torch.zeros(action.shape[0], 3, device=self.device).scatter_(1, action, 1)

        return float(F.mse_loss(self.net(torch.cat([state, action_ohe.squeeze()], dim=-1)), next_state))
        # return F.mse_loss(self.net(torch.cat([state, action_ohe.squeeze()])), next_state)

    def loss(self, state, next_state, action):
        action_ohe = torch.zeros(len(action), self.n_actions, device=self.device).scatter_(1, action.unsqueeze(1), 1)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        # return float(F.mse_loss(self.net(torch.cat([state, action_ohe.squeeze()])), next_state))
        return F.mse_loss(self.net(torch.cat([state, action_ohe.squeeze()], dim=-1)), next_state)

    def update(self, state, next_state, action):
        loss = self.loss(state, next_state, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class InverseModel:
    def __init__(self, state_dim, n_actions, device, hidden_dim=32):

        layers = []
        layers.append(nn.Linear(2 * state_dim, hidden_dim))
        layers.append(nn.ReLU()),
        layers.append(nn.Linear(hidden_dim, hidden_dim)),
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, n_actions))
        layers.append(nn.Softmax())

        self.net = nn.Sequential(*layers).to(device)
        self.optimizer = Adam(self.net.parameters(), lr=1e-4)
        self.device = device

    def intrinsic_reward(self, state, next_state, action):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.loss(state, next_state, action))

    def loss(self, state, next_state, action):
        criterion = nn.CrossEntropyLoss()
        return criterion(self.net(torch.cat([state, next_state], dim=-1)), action.view(-1))

    def update(self, state, next_state, action):
        loss = self.loss(state, next_state, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ICMModel:
    pass
