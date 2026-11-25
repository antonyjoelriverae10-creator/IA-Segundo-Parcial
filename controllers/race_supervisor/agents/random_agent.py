import random

from agents.base import Agent


class RandomAgent(Agent):
    """Baseline de exploracion pura (no es uno de los 4 agentes del parcial)."""
    name = "random"

    def __init__(self, *a, **k):
        super(RandomAgent, self).__init__(*a, **k)
        self.rng = random.Random(self.seed)

    def act(self, obs, training=True):
        return self.rng.randrange(self.n_actions)
