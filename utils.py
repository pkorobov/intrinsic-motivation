import random
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, ext_reward, int_reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), \
                     torch.Tensor([ext_reward]), torch.Tensor([int_reward]), torch.Tensor([done])
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_ext_reward, batch_int_reward, batch_done = zip(*sample)

        batch_state = torch.cat(batch_state).to(device)
        batch_action = torch.LongTensor(batch_action).to(device)
        batch_next_state = torch.cat(batch_next_state).to(device)
        batch_ext_reward = torch.cat(batch_ext_reward).to(device)
        batch_int_reward = torch.cat(batch_int_reward).to(device)
        batch_done = torch.cat(batch_done).to(device)

        return batch_state, batch_action, batch_next_state, batch_ext_reward.unsqueeze(1), \
            batch_int_reward.unsqueeze(1), batch_done.unsqueeze(1)

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
