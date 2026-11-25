import math

from agents.base import Agent


class HeuristicAgent(Agent):
    """Controlador reactivo escrito a mano: sirve para validar entorno + locomocion
    (si ESTE no llega a la meta, el fallo esta en el entorno o la marcha, no en RL).
    Indices de obs: [dist, sin(rumbo), cos(rumbo), r(-60), r(-30), r(0), r(+30), r(+60)].
    Solo corrige hacia la meta por un lado si ese lado esta despejado (evita oscilar
    frente a un obstaculo: girar para esquivar y volver a girar hacia el)."""
    name = "heuristic"
    FRONT_BLOCKED = 0.35   # ~0.5 m
    SIDE_CLEAR = 0.6       # ~0.9 m

    def act(self, obs, training=True):
        bearing = math.atan2(obs[1], obs[2])
        r = obs[3:]
        if min(r[1], r[2], r[3]) < self.FRONT_BLOCKED:      # obstaculo de frente
            return 3 if (r[3] + r[4]) >= (r[0] + r[1]) else 4
        left_ok = min(r[3], r[4]) > self.SIDE_CLEAR
        right_ok = min(r[0], r[1]) > self.SIDE_CLEAR
        if bearing > 0.5 and left_ok:
            return 3
        if bearing < -0.5 and right_ok:
            return 4
        if bearing > 0.15 and left_ok:
            return 1
        if bearing < -0.15 and right_ok:
            return 2
        return 0
