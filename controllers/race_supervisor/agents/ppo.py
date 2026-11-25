"""Proximal Policy Optimization (PPO) discreto con GAE."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from agents.base import Agent
from agents.networks import DEVICE, ActorCritic, obs_tensor


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, obs_dim, n_actions, seed=0, params=None):
        super().__init__(obs_dim, n_actions, seed, params)
        torch.manual_seed(seed)
        p = self.params
        self.gamma = float(p.get("gamma", 0.99))
        self.gae_lambda = float(p.get("gae_lambda", 0.95))
        self.clip = float(p.get("clip", 0.2))
        self.value_coef = float(p.get("value_coef", 0.5))
        self.entropy_coef = float(p.get("entropy_coef", 0.01))
        self.epochs = int(p.get("epochs", 4))
        self.model = ActorCritic(obs_dim, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(p.get("learning_rate", 3e-4)))
        self.rollout = []
        self._training = True
        self.last_loss = None

    def act(self, obs, training=True):
        self._training = bool(training)
        with torch.no_grad():
            logits, value = self.model(obs_tensor(obs))
            dist = Categorical(logits=logits)
            action = dist.sample() if training else torch.argmax(logits, dim=-1)
            log_prob = dist.log_prob(action)
        self._last_action_data = (float(log_prob.item()), float(value.item()))
        return int(action.item())

    def observe(self, obs, action, reward, next_obs, done):
        if self._training:
            log_prob, value = getattr(self, "_last_action_data", (0.0, 0.0))
            self.rollout.append((np.asarray(obs, dtype=np.float32), int(action),
                                 float(reward), np.asarray(next_obs, dtype=np.float32),
                                 bool(done), log_prob, value))

    def end_episode(self, summary):
        if not self._training or not self.rollout:
            self.rollout.clear()
            return
        rewards = [x[2] for x in self.rollout]
        dones = [x[4] for x in self.rollout]
        values = [x[6] for x in self.rollout]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.asarray(values, dtype=np.float32)
        states = torch.as_tensor(np.asarray([x[0] for x in self.rollout]), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor([x[1] for x in self.rollout], dtype=torch.int64, device=DEVICE)
        old_log_probs = torch.as_tensor([x[5] for x in self.rollout], dtype=torch.float32, device=DEVICE)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        last_loss = 0.0
        for _ in range(self.epochs):
            logits, values_now = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
            actor_loss = -torch.min(s1, s2).mean()
            critic_loss = F.mse_loss(values_now, targets)
            entropy = dist.entropy().mean()
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            last_loss = float(loss.item())
        self.last_loss = last_loss
        self.rollout.clear()

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"algorithm": self.name, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(data["model"])
