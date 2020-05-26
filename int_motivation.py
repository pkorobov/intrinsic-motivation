import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def network_block(input_dim, output_dim, hidden_dim=32, mid_layers=0):
    layers = list()
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for i in range(mid_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    block = nn.Sequential(*layers)
    return block


class ForwardModel(nn.Module):
    def __init__(self, state_dim, n_actions, encode_states=False,
                 latent_state_dim=64, action_embedding_dim=64):

        super(ForwardModel, self).__init__()

        if encode_states:
            self.encoder = network_block(state_dim, latent_state_dim, hidden_dim=128)
            self.action_embedding = nn.Embedding(n_actions, action_embedding_dim)
            self.head = network_block(latent_state_dim + action_embedding_dim, state_dim, hidden_dim=128)
        else:
            self.action_embedding = nn.Embedding(n_actions, action_embedding_dim)
            self.head = network_block(state_dim + action_embedding_dim, state_dim)

        self.n_actions = n_actions
        self.device = device
        self.encode_states = encode_states

    def forward(self, state, action):
        if self.encode_states:
            state_emb = self.encoder(state)
        else:
            state_emb = state

        action_emb = self.action_embedding(action)
        return self.head(torch.cat([state_emb, action_emb], dim=-1))

    def loss(self, state, next_state, action):
        if not (torch.is_tensor(state) and torch.is_tensor(next_state) and torch.is_tensor(action)):
            state = torch.tensor(state, dtype=torch.float, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            action = torch.tensor(action, dtype=torch.long, device=device)
        predicted_next_state = self.forward(state, action)
        return 0.5 * (predicted_next_state - next_state).pow(2).mean(0).sum()

    def reward(self, state, next_state, action):
        return float(self.loss(state, next_state, action))


class InverseModel(nn.Module):
    def __init__(self, state_dim, n_actions, encode_states=False, latent_state_dim=64):

        super(InverseModel, self).__init__()

        if encode_states:
            self.encoder = network_block(state_dim, latent_state_dim, hidden_dim=128)
            self.head = network_block(2 * latent_state_dim, n_actions)
        else:
            self.head = network_block(2 * state_dim, n_actions, hidden_dim=128)
        self.n_actions = n_actions
        self.encode_states = encode_states

    def forward(self, state, next_state):
        if self.encode_states:
            state_emb = self.encoder(state)
            next_state_emb = self.encoder(next_state)
        else:
            state_emb = state
            next_state_emb = next_state
        return self.head(torch.cat([state_emb, next_state_emb], dim=-1))

    def loss(self, state, next_state, action):
        if not (torch.is_tensor(state) and torch.is_tensor(next_state) and torch.is_tensor(action)):
            state = torch.tensor(state, dtype=torch.float, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            action = torch.tensor(action, dtype=torch.long, device=device)
        predicted_action = self.forward(state, next_state)
        return nn.CrossEntropyLoss()(predicted_action.view(-1, self.n_actions), action.view(-1))

    def reward(self, state, next_state, action):
        return float(self.loss(state, next_state, action))


class ICMModel(nn.Module):
    def __init__(self, state_dim, n_actions, latent_state_dim=64, eta=1.0):

        super(ICMModel, self).__init__()

        self.encoder = network_block(state_dim, latent_state_dim, hidden_dim=128)
        self.forward_model = ForwardModel(latent_state_dim, n_actions)
        self.inverse_model = InverseModel(latent_state_dim, n_actions)
        self.eta = eta
        self.n_actions = n_actions

    def forward(self, state, next_state, action):
        state_emb = self.encoder(state)
        next_state_emb = self.encoder(next_state)

        predicted_action = self.inverse_model(state_emb, next_state_emb)
        predicted_next_state = self.forward_model(state_emb, action)

        loss1 = nn.CrossEntropyLoss()(predicted_action.view(-1, self.n_actions), action.view(-1))
        loss2 = 0.5 * (predicted_next_state - next_state_emb).pow(2).mean(0).sum()
        return loss2, loss1 + self.eta * loss2

    def loss(self, state, next_state, action):
        if not (torch.is_tensor(state) and torch.is_tensor(next_state) and torch.is_tensor(action)):
            state = torch.tensor(state, dtype=torch.float, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            action = torch.tensor(action, dtype=torch.long, device=device)
        return self.forward(state, next_state, action)[1]

    def reward(self, state, next_state, action):
        if not (torch.is_tensor(state) and torch.is_tensor(next_state) and torch.is_tensor(action)):
            state = torch.tensor(state, dtype=torch.float, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            action = torch.tensor(action, dtype=torch.long, device=device)
        return float(self.forward(state, next_state, action)[0])


class RND(nn.Module):

    def __init__(self, state_dim, output_dim=8, hidden_dim=256):

        super(RND, self).__init__()

        self.target = network_block(state_dim, output_dim, hidden_dim)
        self.predictor = network_block(state_dim, output_dim, hidden_dim)
        for param in self.target.parameters():
            param.requires_grad = False

    def loss(self, next_state):
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        pred = self.predictor(next_state)
        target = self.target(next_state)
        return 0.5 * (pred - target).pow(2).mean(0).sum()

    def reward(self, next_state):
        return float(self.loss(next_state))
