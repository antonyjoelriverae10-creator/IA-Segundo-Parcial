"""Prueba de la LOGICA Python (rangos, recompensas, escudo, bucle de episodios) con un
Supervisor falso cinematico. NO prueba Webots, ni la marcha real, ni colisiones fisicas.
Uso: python tests/test_env_mock.py"""
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "controllers", "race_supervisor"))
from agents import build_agent          # noqa: E402
from race_env import N_ACTIONS, OBS_DIM, RaceEnv  # noqa: E402


class Field(object):
    def __init__(self, node, name):
        self.n, self.name = node, name

    def setSFVec3f(self, v): self.n.pos = list(v)
    def setSFRotation(self, r): self.n.yaw = r[3]
    def setSFString(self, s): self.n.custom = s


class Node(object):
    def __init__(self, sup):
        self.sup, self.pos, self.yaw, self.custom, self.boot = sup, [0, 0, 0.33], 0.0, "", 0.0

    def getField(self, name): return Field(self, name)
    def getPosition(self): return list(self.pos)
    def getOrientation(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [c, -s, 0, s, c, 0, 0, 0, 1]
    def resetPhysics(self): pass
    def restartController(self): self.boot = self.sup.t + 1.0


class FakeSup(object):
    def __init__(self, n):
        self.t, self.nodes = 0.0, {}
        for i in range(n):
            self.nodes["ROBOT_%d" % i], self.nodes["GOAL_%d" % i] = Node(self), Node(self)

    def getFromDef(self, name): return self.nodes.get(name)
    def getBasicTimeStep(self): return 16.0
    def getTime(self): return self.t

    def step(self, ms):
        dt = ms / 1000.0
        self.t += dt
        for k, nd in self.nodes.items():
            if k.startswith("ROBOT") and self.t >= nd.boot:
                try:
                    _, f, t = nd.custom.split(",")
                    f, t = float(f), float(t)
                except ValueError:
                    continue
                nd.yaw += t * 0.6 * dt
                nd.pos[0] += f * 0.12 * dt * math.cos(nd.yaw)
                nd.pos[1] += f * 0.12 * dt * math.sin(nd.yaw)
        return 0


def run(agent_name, episodes=2):
    with open(os.path.join(ROOT, "config", "scenario.json")) as fh:
        cfg = json.load(fh)
    cfg["episode"]["count"] = episodes
    sup = FakeSup(cfg["world"]["laneCount"])
    env = RaceEnv(sup, cfg)
    agents = [build_agent(agent_name, OBS_DIM, N_ACTIONS, seed=i) for i in range(len(env.lanes))]
    results = []
    for _ in range(episodes):
        obs = env.reset()
        assert len(obs[0]) == OBS_DIM
        while not env.all_done():
            acts = [agents[i].act(obs[i]) for i in range(len(agents))]
            obs, rew, done, _ = env.step(acts)
        results.append([(l.outcome, round(l.ret, 1), l.interventions, l.collisions) for l in env.lanes])
    return results


if __name__ == "__main__":
    h = run("heuristic")
    print("heuristic:", h)
    r = run("random")
    print("random   :", r)
    assert any(o[0] == "goal" for ep in h for o in ep), "el heuristico no llego a la meta en el doble cinematico"
    print("OK (solo logica Python con doble cinematico)")
