import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, not_done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, not_done))

    def sample(self, batch_size):
        state, action, reward, next_state, not_done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.concatenate(state)).to(self.device),
            torch.LongTensor(action).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(np.concatenate(next_state)).to(self.device),
            torch.BoolTensor(not_done).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def __init__(self, observation_dim, actions_number, hidden_size=128):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions_number)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).to(device).view(1, -1)
            q_value = self.forward(state)
            action = q_value.max(1)[1].view(-1, 1)
        else:
            action = torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)
        return action



