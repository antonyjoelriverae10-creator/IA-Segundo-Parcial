"""Entorno de carrera multi-carril para DARwIn-OP (un robot/agente por carril).

Cada carril es una copia de la misma pista, separada por paredes: los robots no
pueden chocar entre si (seguridad hacia otros robots) y la carrera es justa.
Un 'paso' del entorno = una decision del agente = decisionMs de simulacion.
Todos los carriles avanzan en lockstep; un episodio termina cuando todos acaban.

NO depende del modulo 'controller': recibe el objeto Supervisor (o un doble de prueba).
Acciones discretas -> (forward, turn) para el controlador C++ de locomocion.
"""
import math
import random

import geometry as geo
import layout

ACTIONS = [(1.0, 0.0), (0.6, 0.6), (0.6, -0.6), (0.0, 1.0), (0.0, -1.0)]
ACTION_NAMES = ["forward", "fwd_left", "fwd_right", "turn_left", "turn_right"]
RANGE_ANGLES = [math.radians(a) for a in (-60, -30, 0, 30, 60)]  # + = izquierda
OBS_DIM = 3 + len(RANGE_ANGLES)  # dist_meta, sin(rumbo), cos(rumbo), 5 rangos
N_ACTIONS = len(ACTIONS)
FORWARD_ACTIONS = (0, 1, 2)
ROBOT_START_Z = 0.33  # altura de caida inicial (verificar con el modelo)


class SimulationEnded(Exception):
    pass


class Lane(object):
    def __init__(self, index, robot, goal, origin_y, boxes_walls, boxes_obs):
        self.index = index
        self.robot, self.goal = robot, goal
        self.pos_f = robot.getField("translation")
        self.rot_f = robot.getField("rotation")
        self.custom_f = robot.getField("customData")
        self.goal_f = goal.getField("translation")
        for name, f in (("translation", self.pos_f), ("rotation", self.rot_f), ("customData", self.custom_f)):
            if f is None:
                raise RuntimeError("Robot ROBOT_%d no expone el campo '%s'" % (index, name))
        self.origin_y = origin_y
        self.boxes = boxes_walls + boxes_obs
        self.obstacle_boxes = boxes_obs
        self.reset_state()

    def reset_state(self):
        self.done = False
        self.outcome = ""
        self.ret = 0.0
        self.decisions = 0
        self.collisions = 0
        self.interventions = 0
        self.finish_time = None
        self.prev_dist = 0.0
        self.state = None
        self.start_y = 0.0
        self.goal_x = 0.0


class RaceEnv(object):
    def __init__(self, sup, cfg, seed=None):
        self.sup, self.cfg = sup, cfg
        self.world, self.track, self.ep = cfg["world"], cfg["track"], cfg["episode"]
        self.rw, self.safety, self.obs_cfg = cfg["reward"], cfg["safety"], cfg["observation"]
        self.dt = int(round(sup.getBasicTimeStep()))
        self.rng = random.Random(cfg.get("seed", 0) if seed is None else seed)
        self.seq = 0
        self.t0 = 0.0
        self.lanes = []
        for i in range(self.world["laneCount"]):
            robot, goal = sup.getFromDef("ROBOT_%d" % i), sup.getFromDef("GOAL_%d" % i)
            if robot is None or goal is None:
                raise RuntimeError("Faltan DEF ROBOT_%d / GOAL_%d en el mundo (regenerar con tools/generate_world.py)" % (i, i))
            self.lanes.append(Lane(i, robot, goal, layout.lane_origin_y(self.world, i),
                                   layout.lane_walls(self.world, i), layout.lane_obstacles(cfg, i)))

    # ------------------------------------------------------------ utilidades
    def _advance(self, ms):
        for _ in range(max(1, int(round(ms / float(self.dt))))):
            if self.sup.step(self.dt) == -1:
                raise SimulationEnded()

    def _send(self, lane, forward, turn):
        self.seq += 1
        lane.custom_f.setSFString("%d,%.3f,%.3f" % (self.seq, forward, turn))

    def _read_state(self, lane):
        x, y, z = lane.robot.getPosition()
        R = lane.robot.getOrientation()
        yaw = math.atan2(R[3], R[0])          # eje local x proyectado (adelante)
        up_z = R[8]                            # componente z del eje local z
        dx, dy = lane.goal_x - x, lane.origin_y - y
        ranges = geo.scan(x, y, yaw, RANGE_ANGLES, lane.boxes, self.obs_cfg["maxRange"])
        d_obs = min(geo.point_box_distance(x, y, b) for b in lane.boxes)
        return {"x": x, "y": y, "z": z, "yaw": yaw, "up_z": up_z,
                "dist": math.hypot(dx, dy), "bearing": geo.wrap_angle(math.atan2(dy, dx) - yaw),
                "ranges": ranges, "d_obs": d_obs,
                "fallen": up_z < 0.7 or z < 0.15}

    def _obs(self, lane):
        s = lane.state
        mr = self.obs_cfg["maxRange"]
        return [min(s["dist"] / self.world["laneLength"], 1.5), math.sin(s["bearing"]), math.cos(s["bearing"])] + \
               [r / mr for r in s["ranges"]]

    def _shield(self, action, s):
        """Escudo de seguridad: con un obstaculo cerca de frente solo se permite girar."""
        if not self.safety["enabled"]:
            return action, False
        r = s["ranges"]
        if min(r[1], r[2], r[3]) < self.safety["dSafe"] and action in FORWARD_ACTIONS:
            left, right = r[3] + r[4], r[0] + r[1]
            return (3 if left >= right else 4), True
        return action, False

    # ------------------------------------------------------------------- API
    def reset(self):
        for lane in self.lanes:
            lane.reset_state()
            lane.start_y = self.rng.uniform(*self.ep["startYRange"])
            lane.goal_x = self.rng.uniform(*self.ep["goalXRange"])
            lane.custom_f.setSFString("")   # antes de reiniciar: evita ordenes obsoletas
            lane.pos_f.setSFVec3f([self.track["startX"], lane.origin_y + lane.start_y, ROBOT_START_Z])
            lane.rot_f.setSFRotation([0, 0, 1, 0])
            lane.robot.resetPhysics()
            lane.robot.restartController()
            lane.goal_f.setSFVec3f([lane.goal_x, lane.origin_y, 0.005])
        self._advance(self.ep["warmupMs"])   # el controlador adopta la postura inicial
        self.t0 = self.sup.getTime()
        for lane in self.lanes:
            lane.state = self._read_state(lane)
            lane.prev_dist = lane.state["dist"]
            if lane.state["fallen"]:
                lane.done, lane.outcome = True, "fall_at_reset"
        return [self._obs(l) for l in self.lanes]

    def all_done(self):
        return all(l.done for l in self.lanes)

    def step(self, actions):
        """actions: lista (una por carril; ignorada si el carril ya termino)."""
        active = [l for l in self.lanes if not l.done]
        intervened = {}
        diag = {}  # solo diagnostico: no influye en el comportamiento
        for lane in active:
            a, iv = self._shield(int(actions[lane.index]), lane.state)
            intervened[lane.index] = iv
            diag[lane.index] = {"decision": lane.decisions, "pre_state": lane.state,
                                "agent_action": int(actions[lane.index]), "final_action": a}
            lane.interventions += int(iv)
            self._send(lane, *ACTIONS[a])
        self._advance(self.ep["decisionMs"])

        rw = self.rw
        rewards, infos = [0.0] * len(self.lanes), [{} for _ in self.lanes]
        for lane in active:
            s = self._read_state(lane)
            lane.decisions += 1
            r = rw["progressGain"] * (lane.prev_dist - s["dist"]) - rw["stepPenalty"]
            collided = s["d_obs"] < self.obs_cfg["robotRadius"]
            if collided:
                lane.collisions += 1
                r -= rw["collisionPenalty"]
            elif s["d_obs"] < self.safety["dSafe"]:
                r -= rw["proximityGain"] * (self.safety["dSafe"] - s["d_obs"]) / self.safety["dSafe"]
            if intervened[lane.index]:
                r -= rw["shieldPenalty"]
            lane.prev_dist, lane.state = s["dist"], s
            if s["fallen"]:
                r -= rw["fallPenalty"]
                lane.done, lane.outcome = True, "fall"
            elif s["dist"] < self.ep["goalRadius"]:
                r += rw["goalBonus"] + rw["timeBonus"] * (1.0 - lane.decisions / float(self.ep["maxDecisions"]))
                lane.done, lane.outcome = True, "goal"
                lane.finish_time = self.sup.getTime() - self.t0
            elif lane.decisions >= self.ep["maxDecisions"]:
                lane.done, lane.outcome = True, "timeout"
            lane.ret += r
            rewards[lane.index] = r
            infos[lane.index] = {"intervened": intervened[lane.index], "collided": collided, "outcome": lane.outcome}
            infos[lane.index].update(diag[lane.index])
            infos[lane.index]["done"] = lane.done
            if lane.done:
                self._send(lane, 0.0, 0.0)   # parar al terminar
        return [self._obs(l) for l in self.lanes], rewards, [l.done for l in self.lanes], infos

    def summary(self, lane):
        return {"lane": lane.index, "outcome": lane.outcome, "return": round(lane.ret, 3),
                "decisions": lane.decisions, "collisions": lane.collisions,
                "interventions": lane.interventions,
                "finish_time_s": "" if lane.finish_time is None else round(lane.finish_time, 2),
                "final_dist": round(lane.state["dist"], 3) if lane.state else "",
                "start_y": round(lane.start_y, 3), "goal_x": round(lane.goal_x, 3)}
