from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from agents.a2c_v2 import A2CV2
from agents.ppo import PPOAgent


def build_agent(name, obs_dim, n_actions, seed=0, params=None):
    key = str(name).lower()
    if key == "random":
        return RandomAgent(obs_dim, n_actions, seed, params)
    if key == "heuristic":
        return HeuristicAgent(obs_dim, n_actions, seed, params)
    if key == "dqn":
        return DQNAgent(obs_dim, n_actions, seed, params)
    if key == "a2c":
        return A2CAgent(obs_dim, n_actions, seed, params)
    if key == "a2c_v2":
        return A2CV2(obs_dim, n_actions, seed, params)
    if key == "ppo":
        return PPOAgent(obs_dim, n_actions, seed, params)
    raise ValueError("Agente desconocido: %s" % name)
