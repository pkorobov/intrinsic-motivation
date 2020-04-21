import gym
from itertools import count
from dqn import DQN, ReplayMemory
from curiosity import ForwardModel, InverseModel, ICMModel
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000
TARGET_UPDATE = 10
STEPS_DONE = 0
ETA = 100.0
LEARNING_STARTS = 2000


def fix_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


env = gym.envs.make("MountainCar-v0")
fix_seed(0, env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# curiosity = ForwardModel(obs_dim, n_actions, encode_states=True, device=device).to(device)
# curiosity = InverseModel(obs_dim, n_actions, device, latent_state_dim=32, encode_states=True).to(device)
curiosity = ICMModel(obs_dim, n_actions, latent_state_dim=32).to(device)

policy_net = DQN(observation_dim=obs_dim, n_actions=n_actions, hidden_size=32).to(device)
target_net = DQN(observation_dim=obs_dim, n_actions=n_actions, hidden_size=32).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
curiosity_optimizer = optim.Adam(curiosity.parameters(), lr=1e-3)

memory = ReplayMemory(capacity=10000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    state, action, next_state, reward, not_done = memory.sample(BATCH_SIZE)

    loss = curiosity.loss(state, next_state, action)
    curiosity_optimizer.zero_grad()
    loss.backward()
    curiosity_optimizer.step()

    if STEPS_DONE < LEARNING_STARTS:
        return

    state_action_values = policy_net(state).gather(1, action.view(-1, 1))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[not_done] = target_net(next_state[not_done]).max(1)[0]
    expected_state_action_values = reward + next_state_values.detach() * GAMMA
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()


num_episodes = 500
cum_e_reward, cum_i_reward = 0, 0
flag_achieved = 0
episode_rewards = []

for i_episode in range(num_episodes):
    state = env.reset()

    for t in count():
        env.render()

        eps = EPS_END + (EPS_START - EPS_END) * 10.0 ** (-STEPS_DONE / EPS_DECAY)
        STEPS_DONE += 1

        action = policy_net.act(state, eps).item()
        next_state, e_reward, done, _ = env.step(action)
        i_reward = curiosity.reward(state, next_state, action) * ETA

        cum_e_reward += e_reward
        cum_i_reward += i_reward

        if STEPS_DONE < LEARNING_STARTS:
            i_reward = 0

        memory.add(state, action, next_state, e_reward + i_reward, 1 - done)
        state = next_state
        optimize_model()

        if done:
            print(f"Extrinsic and intrinsic rewards for the episode {i_episode}: {cum_e_reward, cum_i_reward}")
            episode_rewards.append(cum_e_reward)
            cum_e_reward, cum_i_reward = 0, 0
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.render()
env.close()

episode_rewards = np.array(episode_rewards)
print(f"Flag was achieved in {(episode_rewards[-100:] > -200).sum()} of last {100} episodes")
print(f"Average reward is {episode_rewards[-100:].mean()} for last {100} episodes")

plt.figure(figsize=(12, 8))
plt.plot(episode_rewards)
plt.xlabel("Episode number")
plt.ylabel("Reward")
plt.show()
plt.savefig("episode_rewards.pdf")
