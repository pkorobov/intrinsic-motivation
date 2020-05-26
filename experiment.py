import gym
from itertools import count
from qr_dqn import QuantileRegressionDQN
from utils import ReplayMemory, huber, fix_seed, RunningMeanStd
from int_motivation import ForwardModel, InverseModel, ICMModel, RND
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import copy

GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_episodes', default=801, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eta', default=1.0, type=float)
    parser.add_argument('--learning_starts', default=2000, type=int)
    parser.add_argument('--int_motivation_type', default=None, type=str)
    parser.add_argument('--output_dir', default="results", type=str)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--num_quant', default=5, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--print', default=False, type=bool)
    parser.add_argument('--normalize_int_reward', default=False, type=bool)
    args = parser.parse_args()

    env = gym.envs.make("MountainCar-v0")
    fix_seed(args.seed, env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    int_reward_rms = RunningMeanStd()

    intrinsic_motivation = None
    if args.int_motivation_type == 'ICM':
        intrinsic_motivation = ICMModel(obs_dim, n_actions).to(device)
    if args.int_motivation_type == 'Forward':
        intrinsic_motivation = ForwardModel(obs_dim, n_actions, encode_states=True).to(device)
    if args.int_motivation_type == 'Inverse':
        intrinsic_motivation = InverseModel(obs_dim, n_actions, encode_states=True).to(device)
    if args.int_motivation_type == 'RND':
        intrinsic_motivation = RND(obs_dim).to(device)

    Z_net = QuantileRegressionDQN(observation_dim=obs_dim, n_actions=n_actions,
                                  hidden_size=args.hidden_size, num_quant=args.num_quant).to(device)
    Z_target_net = copy.deepcopy(Z_net)
    Z_target_net.eval()

    policy_optimizer = optim.Adam(Z_net.parameters(), lr=1e-3)
    intrinsic_motivation_optimizer = optim.Adam(intrinsic_motivation.parameters(), lr=1e-3) \
        if intrinsic_motivation is not None else None

    memory = ReplayMemory(capacity=args.memory_size)

    def eps(episode_number):
        return max(EPS_END, EPS_START * 0.9 ** episode_number)

    tau = torch.Tensor((2 * np.arange(Z_net.num_quant) + 1) / (2.0 * Z_net.num_quant)).view(1, -1).to(device)
    flag_achieved = 0
    episode_rewards = []
    episode_int_rewards = []

    steps_done = 0
    for episode in range(args.num_episodes):
        sum_ext_reward, sum_int_reward = 0, 0
        state = env.reset()

        for t in count():
            if args.render:
                env.render()

            steps_done += 1

            action = Z_net.act(torch.Tensor([state]).to(device), eps(episode))
            next_state, ext_reward, done, _ = env.step(action)
            sum_ext_reward += ext_reward

            if intrinsic_motivation is not None:
                if args.int_motivation_type != "RND":
                    int_reward = intrinsic_motivation.reward(state, next_state, action)
                else:
                    int_reward = intrinsic_motivation.reward(next_state)
                sum_int_reward += int_reward
            else:
                int_reward = 0.
            memory.push(state, action, next_state, ext_reward, int_reward, float(done))

            if len(memory) >= args.batch_size:
                states, actions, next_states, ext_rewards, int_rewards, dones = memory.sample(args.batch_size)

                if args.normalize_int_reward:
                    int_reward_rms.update(int_rewards.cpu().numpy())
                    int_rewards /= torch.from_numpy(np.sqrt(int_reward_rms.var)).to(device)

                if steps_done == args.learning_starts:
                    memory.clear()

                # the main algorithm training
                if steps_done >= args.learning_starts:
                    theta = Z_net(states)[np.arange(args.batch_size), actions]
                    Z_next = Z_target_net(next_states).detach()
                    Z_next_max = Z_next[np.arange(args.batch_size), Z_next.mean(2).max(1)[1]]
                    T_theta = (ext_rewards + args.eta * int_rewards) + GAMMA * (1 - dones) * Z_next_max

                    diff = T_theta.t().unsqueeze(-1) - theta
                    loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
                    loss = loss.mean()

                    policy_optimizer.zero_grad()
                    loss.backward()
                    policy_optimizer.step()

                # intrinsic motivation training
                if intrinsic_motivation is not None:
                    loss = intrinsic_motivation.loss(states, next_states, actions) if args.int_motivation_type != "RND" \
                                else intrinsic_motivation.loss(next_states)
                    intrinsic_motivation_optimizer.zero_grad()
                    loss.backward()
                    intrinsic_motivation_optimizer.step()
            state = next_state

            if steps_done % TARGET_UPDATE == 0:
                Z_target_net.load_state_dict(Z_net.state_dict())

            if done:
                if args.print:
                    print(f"Extrinsic and intrinsic rewards for the episode {episode}: {sum_ext_reward, sum_int_reward}")
                episode_rewards.append(sum_ext_reward)
                episode_int_rewards.append(sum_int_reward)
                break

    if args.render:
        env.render()
    env.close()

    episode_rewards = np.array(episode_rewards)
    episode_int_rewards = np.array(episode_int_rewards)

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    torch.save(Z_net.state_dict(), f"{args.output_dir}/QR_DQN_{args.int_motivation_type}_{args.seed}_weights")
    np.save(f"{args.output_dir}/{args.int_motivation_type}_{args.seed}", episode_rewards)
    np.save(f"{args.output_dir}/{args.int_motivation_type}_int_{args.seed}", episode_int_rewards)
    print(f"Flag was achieved in {(episode_rewards[-100:] > -200).sum()} of last {100} episodes")
    print(f"Average reward for last {100} episodes is {episode_rewards[-100:].mean()}")

    fig = plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards)
    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    fig.savefig(f"{args.output_dir}/{args.int_motivation_type}_{args.seed}.png")
