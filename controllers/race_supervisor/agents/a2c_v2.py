"""A2C v2: actor/critic independientes, minibatches y Safety Shield explícito."""
from collections import deque
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class A2CV2:
    name = "a2c_v2"

    def __init__(self, obs_dim, n_actions, seed=0, params=None):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.params = dict(params or {})
        self.gamma = float(self.params.get("gamma", 0.99))
        self.gae_lambda = float(self.params.get("gae_lambda", 0.95))
        self.value_coef = float(self.params.get("value_coef", 0.5))
        self.entropy_coef = float(self.params.get("entropy_coef", 0.01))
        self.learning_rate = float(self.params.get("learning_rate", 3e-4))
        self.minibatch_size = max(1, int(self.params.get("minibatch_size", 32)))
        self.max_grad_norm = float(self.params.get("max_grad_norm", 0.5))

        torch.manual_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))
        self.actor = PolicyNetwork(self.obs_dim, self.n_actions).to(DEVICE)
        self.critic = ValueNetwork(self.obs_dim).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.rollout = []
        self.metrics_history = []
        self.last_metrics = self._empty_metrics()

    @staticmethod
    def _empty_metrics():
        return {
            "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
            "actor_samples": 0, "critic_samples": 0,
            "intervened_samples": 0, "minibatches": 0,
            "mean_actor_advantage": 0.0, "std_actor_advantage": 0.0,
            "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
        }

    def act(self, obs, training=True):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=DEVICE).view(1, -1)
        with torch.no_grad():
            dist = Categorical(logits=self.actor(x))
            action = dist.sample() if training else torch.argmax(dist.logits, dim=-1)
        return int(action.item())

    def observe(self, obs, action, reward, next_obs, done,
                intervened=False, executed_action=None):
        self.rollout.append({
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "action": int(action),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
            "done": bool(done),
            "intervened": bool(intervened),
            "executed_action": int(executed_action if executed_action is not None else action),
        })

    def _calculate_advantages(self, obs, next_obs, rewards, dones, intervened):
        with torch.no_grad():
            values = self.critic(obs).cpu().numpy()
            next_values = self.critic(next_obs).cpu().numpy()
        critic_adv = np.zeros_like(rewards, dtype=np.float32)
        actor_adv = np.zeros_like(rewards, dtype=np.float32)
        gae_critic = 0.0
        gae_actor = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_values[t] * nonterminal - values[t]
            gae_critic = delta + self.gamma * self.gae_lambda * nonterminal * gae_critic
            critic_adv[t] = gae_critic

            # Una intervención en t+1 corta la atribución de esa recompensa a la acción de t.
            next_was_intervened = float(intervened[t + 1]) if t + 1 < len(rewards) else 0.0
            trace_mask = nonterminal * (1.0 - next_was_intervened)
            gae_actor = delta + self.gamma * self.gae_lambda * trace_mask * gae_actor
            actor_adv[t] = gae_actor
        return actor_adv, critic_adv, values

    def end_episode(self, summary=None):
        if not self.rollout:
            self.last_metrics = self._empty_metrics()
            return

        batch = self.rollout
        self.rollout = []
        obs = torch.as_tensor(np.stack([x["obs"] for x in batch]), dtype=torch.float32, device=DEVICE)
        next_obs = torch.as_tensor(np.stack([x["next_obs"] for x in batch]), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor([x["action"] for x in batch], dtype=torch.long, device=DEVICE)
        rewards = np.asarray([x["reward"] for x in batch], dtype=np.float32)
        dones = np.asarray([x["done"] for x in batch], dtype=np.bool_)
        intervened = np.asarray([x["intervened"] for x in batch], dtype=np.bool_)

        actor_adv, critic_adv, values = self._calculate_advantages(
            obs, next_obs, rewards, dones, intervened
        )
        returns = critic_adv + values
        actor_mask = ~intervened
        valid_adv = actor_adv[actor_mask]
        if valid_adv.size > 1:
            valid_adv = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
            actor_adv = actor_adv.copy()
            actor_adv[actor_mask] = valid_adv

        actor_adv_t = torch.as_tensor(actor_adv, dtype=torch.float32, device=DEVICE)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
        order = self.rng.permutation(len(batch))
        metric = self._empty_metrics()
        actor_losses, critic_losses, entropies = [], [], []
        actor_grad_norms, critic_grad_norms = [], []
        minibatches = 0

        for start in range(0, len(order), self.minibatch_size):
            ids_np = order[start:start + self.minibatch_size]
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=DEVICE)
            minibatches += 1

            # El crítico utiliza todas las transiciones, incluidas las intervenidas.
            value_pred = self.critic(obs[ids])
            critic_loss = nn.functional.mse_loss(value_pred, returns_t[ids])
            self.critic_optimizer.zero_grad(set_to_none=True)
            (self.value_coef * critic_loss).backward()
            c_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))
            critic_grad_norms.append(float(c_norm.detach().cpu()))

            valid_local = ~torch.as_tensor(intervened[ids_np], dtype=torch.bool, device=DEVICE)
            if bool(valid_local.any()):
                logits = self.actor(obs[ids][valid_local])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions[ids][valid_local])
                entropy = dist.entropy().mean()
                adv = actor_adv_t[ids][valid_local]
                actor_loss = -(log_probs * adv.detach()).mean()
                total_actor_loss = actor_loss - self.entropy_coef * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                a_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                actor_grad_norms.append(float(a_norm.detach().cpu()))

        metric.update({
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "actor_samples": int(actor_mask.sum()),
            "critic_samples": int(len(batch)),
            "intervened_samples": int(intervened.sum()),
            "minibatches": int(minibatches),
            "mean_actor_advantage": float(actor_adv[actor_mask].mean()) if actor_mask.any() else 0.0,
            "std_actor_advantage": float(actor_adv[actor_mask].std()) if actor_mask.sum() > 1 else 0.0,
            "actor_grad_norm": float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            "critic_grad_norm": float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0,
        })
        self.last_metrics = metric
        record = dict(metric)
        if isinstance(summary, dict):
            for key in ("episode", "lane", "outcome", "return", "decisions"):
                if key in summary:
                    record[key] = summary[key]
        self.metrics_history.append(record)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "algorithm": "a2c_v2",
            "format_version": 1,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "params": self.params,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }, path)

    def load(self, path):
        try:
            data = torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:  # Compatibilidad con versiones de PyTorch sin weights_only.
            data = torch.load(path, map_location=DEVICE)
        if data.get("algorithm") != "a2c_v2":
            raise ValueError("Checkpoint incompatible: se esperaba un checkpoint a2c_v2")
        if int(data.get("obs_dim", -1)) != self.obs_dim or int(data.get("n_actions", -1)) != self.n_actions:
            raise ValueError("Dimensiones del checkpoint a2c_v2 incompatibles")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.metrics_history = list(data.get("metrics_history", []))
