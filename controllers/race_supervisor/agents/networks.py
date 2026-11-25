"""Redes neuronales compartidas para los agentes de aprendizaje."""
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(input_dim, output_dim, hidden=(128, 128)):
    layers = []
    last = int(input_dim)
    for width in hidden:
        layers.extend([nn.Linear(last, int(width)), nn.ReLU()])
        last = int(width)
    layers.append(nn.Linear(last, int(output_dim)))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        self.shared = mlp(obs_dim, hidden[-1], hidden[:-1]) if len(hidden) > 1 else nn.Sequential(nn.Linear(obs_dim, hidden[0]), nn.ReLU())
        width = int(hidden[-1])
        self.actor = nn.Linear(width, n_actions)
        self.critic = nn.Linear(width, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features).squeeze(-1)


def obs_tensor(obs):
    return torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).reshape(1, -1)
