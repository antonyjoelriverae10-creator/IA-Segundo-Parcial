"""A2C episodico con GAE y manejo explicito de intervenciones del Safety Shield."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from agents.base import Agent
from agents.networks import DEVICE, ActorCritic, obs_tensor


class A2CAgent(Agent):
    name = "a2c"

    def __init__(self, obs_dim, n_actions, seed=0, params=None):
        super().__init__(obs_dim, n_actions, seed, params)
        torch.manual_seed(seed)
        p = self.params
        self.gamma = float(p.get("gamma", 0.99))
        self.gae_lambda = float(p.get("gae_lambda", 0.95))
        self.value_coef = float(p.get("value_coef", 0.5))
        self.entropy_coef = float(p.get("entropy_coef", 0.01))
        self.max_grad_norm = float(p.get("max_grad_norm", 0.5))

        self.model = ActorCritic(obs_dim, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(p.get("learning_rate", 3e-4))
        )
        self.rollout = []
        self._training = True
        self.last_loss = None
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.last_actor_samples = 0
        self.last_intervened_samples = 0

    def act(self, obs, training=True):
        self._training = bool(training)
        with torch.no_grad():
            logits, _ = self.model(obs_tensor(obs))
            dist = Categorical(logits=logits)
            action = dist.sample() if training else torch.argmax(logits, dim=-1)
        return int(action.item())

    def observe(self, obs, action, reward, next_obs, done,
                intervened=False, executed_action=None):
        """Guarda transición. La acción intervenida no genera gradiente del actor.

        El crítico sí aprende de todas las transiciones, incluidas las modificadas
        por el shield, porque sus recompensas y estados siguientes son reales.
        """
        if not self._training:
            return
        self.rollout.append({
            "obs": np.asarray(obs, dtype=np.float32),
            "action": int(action),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "done": bool(done),
            "intervened": bool(intervened),
            "executed_action": int(action if executed_action is None else executed_action),
        })

    def end_episode(self, summary):
        if not self._training or not self.rollout:
            self.rollout.clear()
            return

        states_np = np.asarray([x["obs"] for x in self.rollout], dtype=np.float32)
        next_states_np = np.asarray([x["next_obs"] for x in self.rollout], dtype=np.float32)
        actions_np = np.asarray([x["action"] for x in self.rollout], dtype=np.int64)
        rewards_np = np.asarray([x["reward"] for x in self.rollout], dtype=np.float32)
        dones_np = np.asarray([x["done"] for x in self.rollout], dtype=np.float32)
        intervened_np = np.asarray([x["intervened"] for x in self.rollout], dtype=np.bool_)

        states = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=DEVICE)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=DEVICE)
        actor_mask = torch.as_tensor(~intervened_np, dtype=torch.bool, device=DEVICE)

        # GAE: el crítico aprovecha todas las transiciones; los terminales no bootstrap.
        with torch.no_grad():
            _, values_old = self.model(states)
            _, next_values = self.model(next_states)
            deltas = rewards + self.gamma * next_values * (1.0 - dones) - values_old

            advantages = torch.zeros_like(rewards)
            gae = torch.zeros((), dtype=torch.float32, device=DEVICE)
            for t in reversed(range(len(self.rollout))):
                continuation = 1.0 - dones[t]
                gae = deltas[t] + self.gamma * self.gae_lambda * continuation * gae
                advantages[t] = gae
            returns = advantages + values_old

        logits, values = self.model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        if bool(actor_mask.any()):
            actor_advantages = advantages[actor_mask]
            if actor_advantages.numel() > 1:
                actor_advantages = (
                    actor_advantages - actor_advantages.mean()
                ) / (actor_advantages.std(unbiased=False) + 1e-8)
            actor_loss = -(log_probs[actor_mask] * actor_advantages.detach()).mean()
            entropy = dist.entropy()[actor_mask].mean()
        else:
            # Si todo el episodio fue intervenido, se entrena solo el crítico.
            actor_loss = logits.sum() * 0.0
            entropy = logits.sum() * 0.0

        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.last_loss = float(loss.detach().item())
        self.last_actor_loss = float(actor_loss.detach().item())
        self.last_critic_loss = float(critic_loss.detach().item())
        self.last_actor_samples = int(actor_mask.sum().item())
        self.last_intervened_samples = int((~actor_mask).sum().item())
        self.rollout.clear()

    def save(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        torch.save({
            "algorithm": self.name,
            "format_version": 2,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path, load_optimizer=False):
        """Carga checkpoints antiguos o nuevos; por defecto reinicia el optimizador."""
        try:
            data = torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:  # compatibilidad con versiones anteriores de PyTorch
            data = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(data["model"])
        if load_optimizer and "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
