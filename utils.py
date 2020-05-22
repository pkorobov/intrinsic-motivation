import random
import torch

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

        batch_state = torch.cat(batch_state).to(device)
        batch_action = torch.LongTensor(batch_action).to(device)
        batch_next_state = torch.cat(batch_next_state).to(device)
        batch_reward = torch.cat(batch_reward).to(device)
        batch_done = torch.cat(batch_done).to(device)

        return batch_state, batch_action, batch_next_state, batch_reward.unsqueeze(1), batch_done.unsqueeze(1)

    def __len__(self):
        return len(self.memory)
