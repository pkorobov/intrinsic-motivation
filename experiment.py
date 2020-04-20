import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from dqn import DQN, ReplayMemory
from curiosity import ForwardModel, InverseModel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

episodes = 10

# env = gym.envs.make("CartPole-v0")
env = gym.envs.make("MountainCar-v0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000
TARGET_UPDATE = 10

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

curiosity = ForwardModel(obs_dim, n_actions, device)
# curiosity = InverseModel(obs_dim, n_actions, device)

policy_net = DQN(observation_dim=obs_dim, actions_number=n_actions, hidden_size=32).to(device)
target_net = DQN(observation_dim=obs_dim, actions_number=n_actions, hidden_size=32).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(capacity=100000)
steps_done = 0

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    state, action, next_state, reward, not_done = memory.sample(BATCH_SIZE)

    state_action_values = policy_net(state).gather(1, action.view(-1, 1))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[not_done] = target_net(next_state[not_done]).max(1)[0].detach()
    # next_state_values = target_net(next_state).max(1)[0].detach()
    expected_state_action_values = reward + next_state_values * GAMMA

    # action_ohe = torch.zeros(len(action), n_actions, device=device).scatter_(1, action.unsqueeze(1), 1)
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    curiosity.update(state, next_state, action)

    # curiosity.update(state, next_state, action_ohe)


num_episodes = 500

cum_reward = 0
cum_i_reward = 0
for i_episode in range(num_episodes):
    state = env.reset()

    i_weight = 10

    for t in count():
        env.render()

        eps = EPS_END + (EPS_START - EPS_END) * \
              math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        action = policy_net.act(state, eps)
        next_state, reward, done, _ = env.step(action.item())
        cum_reward += reward

        action_ohe = torch.zeros(len(action), n_actions, device=device).scatter_(1, action, 1)
        i_reward = curiosity.intrinsic_reward(state, next_state, action)

        cum_i_reward += i_reward

        memory.add(state, action, next_state, reward + i_reward * i_weight, 1 - done)
        state = next_state
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            break

    print("__________________________")
    print(cum_reward, cum_i_reward * i_weight)
    print("__________________________")

    cum_reward = 0
    cum_i_reward = 0

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')

env.render()
env.close()
