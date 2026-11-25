"""Interfaz comun de los agentes (Taller 2, PPO, DQN, Actor-Critic y baselines)."""


class Agent(object):
    name = "base"

    def __init__(self, obs_dim, n_actions, seed=0, params=None):
        self.obs_dim, self.n_actions, self.seed = obs_dim, n_actions, seed
        self.params = params or {}

    def act(self, obs, training=True):
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, done):
        """Transicion (s, a, r, s', done). Los agentes que aprenden la usan aqui."""

    def end_episode(self, summary):
        """Llamado al terminar el episodio del carril (summary = metricas)."""

    def save(self, path):
        pass

    def load(self, path):
        pass
