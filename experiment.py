import gym
from itertools import count
from qr_dqn import QuantileRegressionDQN, ReplayMemory, huber
from int_motivation import ForwardModel, InverseModel, ICMModel, RND
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse


GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 500
TARGET_UPDATE = 100


def fix_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_episodes', default=501, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eta', default=1.0, type=float)
    parser.add_argument('--learning_starts', default=2000, type=int)
    parser.add_argument('--int_motivation_type', default=None, type=str)
    args = parser.parse_args()

    env = gym.envs.make("MountainCar-v0")
    fix_seed(args.seed, env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    intrinsic_motivation = None
    if args.int_motivation_type == 'ICM':
        intrinsic_motivation = ICMModel(obs_dim, n_actions, latent_state_dim=32).to(device)
    if args.int_motivation_type == 'Forward':
        intrinsic_motivation = ForwardModel(obs_dim, n_actions, encode_states=True, device=device).to(device)
    if args.int_motivation_type == 'Inverse':
        intrinsic_motivation = InverseModel(obs_dim, n_actions, device, latent_state_dim=32, encode_states=True).to(device)
    if args.int_motivation_type == 'RND':
        intrinsic_motivation = RND(obs_dim).to(device)


    policy_net = QuantileRegressionDQN(observation_dim=obs_dim, n_actions=n_actions).to(device)
    target_net = QuantileRegressionDQN(observation_dim=obs_dim, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    intrinsic_motivation_optimizer = optim.Adam(intrinsic_motivation.parameters(), lr=1e-3) \
        if intrinsic_motivation is not None else None

    memory = ReplayMemory(capacity=args.memory_size)

    eps = lambda steps: EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps / EPS_DECAY)

    tau = torch.Tensor((2 * np.arange(policy_net.num_quant) + 1) / (2.0 * policy_net.num_quant)).view(1, -1)
    flag_achieved = 0
    episode_rewards = []

    steps_done = 0
    for episode in range(args.num_episodes):
        sum_ext_reward, sum_int_reward = 0, 0
        state = env.reset()
        for t in count():
            env.render()

            steps_done += 1

            action = policy_net.act(torch.Tensor([state]), eps(steps_done))
            next_state, ext_reward, done, _ = env.step(action)
            sum_ext_reward += ext_reward

            if intrinsic_motivation is not None:
                # int_reward = np.clip(intrinsic_motivation.reward(state, next_state, action), 0, 10)
                int_reward = intrinsic_motivation.reward(state, next_state, action) if args.int_motivation_type != "RND" \
                                  else intrinsic_motivation.reward(next_state)
                reward = int_reward * args.eta + ext_reward
                sum_int_reward += int_reward * args.eta
            else:
                reward = ext_reward
            memory.push(state, action, next_state, reward, float(done))

            if len(memory) >= args.batch_size:
                states, actions, next_states, rewards, dones = memory.sample(args.batch_size)

                # intrinsic motivation training
                if intrinsic_motivation is not None:
                    loss = intrinsic_motivation.loss(state, next_state, action) if args.int_motivation_type != "RND" \
                                else intrinsic_motivation.loss(next_state)
                    intrinsic_motivation_optimizer.zero_grad()
                    loss.backward()
                    intrinsic_motivation_optimizer.step()

                # the main algorithm training
                if steps_done >= args.learning_starts:
                    theta = policy_net(states)[np.arange(args.batch_size), actions]
                    Znext = target_net(next_states).detach()
                    Znext_max = Znext[np.arange(args.batch_size), Znext.mean(2).max(1)[1]]
                    Ttheta = rewards + GAMMA * (1 - dones) * Znext_max

                    diff = Ttheta.t().unsqueeze(-1) - theta
                    loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
                    loss = loss.mean()

                    policy_optimizer.zero_grad()
                    loss.backward()
                    policy_optimizer.step()
            state = next_state

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                print(f"Extrinsic and intrinsic rewards for the episode {episode}: {sum_ext_reward, sum_int_reward}")
                episode_rewards.append(sum_ext_reward)
                break

    env.render()
    env.close()

    episode_rewards = np.array(episode_rewards)
    print(f"Flag was achieved in {(episode_rewards[-100:] > -200).sum()} of last {100} episodes")
    print(f"Average reward for last {100} episodes is {episode_rewards[-100:].mean()}")

    fig = plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards)
    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    plt.show()
    fig.savefig("episode_rewards.pdf")
