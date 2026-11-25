"""Deep Q-Network para acciones discretas."""
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.base import Agent
from agents.networks import DEVICE, QNetwork, obs_tensor


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, obs_dim, n_actions, seed=0, params=None):
        super().__init__(obs_dim, n_actions, seed, params)
        p = self.params
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.gamma = float(p.get("gamma", 0.99))
        self.lr = float(p.get("learning_rate", 1e-3))
        self.batch_size = int(p.get("batch_size", 64))
        self.capacity = int(p.get("replay_capacity", 20000))
        self.min_replay = int(p.get("min_replay", 256))
        self.target_interval = int(p.get("target_interval", 250))
        self.epsilon = float(p.get("epsilon_start", 1.0))
        self.epsilon_min = float(p.get("epsilon_min", 0.05))
        self.epsilon_decay = float(p.get("epsilon_decay", 0.995))
        self.online = QNetwork(obs_dim, n_actions).to(DEVICE)
        self.target = QNetwork(obs_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.capacity)
        self.steps = 0
        self._training = True
        self.loss_value = None

    def act(self, obs, training=True):
        self._training = bool(training)
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return int(self.online(obs_tensor(obs)).argmax(dim=1).item())

    def observe(self, obs, action, reward, next_obs, done):
        if not self._training:
            return
        self.memory.append((np.asarray(obs, dtype=np.float32), int(action),
                            float(reward), np.asarray(next_obs, dtype=np.float32), float(done)))
        self.steps += 1
        if len(self.memory) >= max(self.batch_size, self.min_replay):
            self._update()
        if self.steps % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE)
        q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            target = rewards + self.gamma * self.target(next_states).max(dim=1).values * (1.0 - dones)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.loss_value = float(loss.item())

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"algorithm": self.name, "online": self.online.state_dict(),
                    "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "epsilon": self.epsilon, "steps": self.steps}, path)

    def load(self, path):
        data = torch.load(path, map_location=DEVICE)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data.get("target", data["online"]))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.steps = int(data.get("steps", 0))
