"""Registro de diagnostico por decision (solo lectura: no influye en agentes, entorno ni C++).

Escribe results/<run>/decisions.csv. Una fila por (episodio, carril, decision) con el
estado que el agente VIO al decidir (antes de actuar), las acciones y el resultado tras
ejecutar esa decision. Ademas, una fila por carril al inicio de cada episodio
(decision = -1, tras el calentamiento) con la pose inicial y sin acciones.
Unidades: x, y en metros (mundo); y_lane relativo al centro del carril; yaw y bearing
en radianes; rangos en metros (la observacion los normaliza dividiendo por maxRange)."""
import csv
import os

from race_env import ACTION_NAMES, RANGE_ANGLES  # noqa: F401  (RANGE_ANGLES documenta el orden)

COLUMNS = [
    "run_id", "episode", "lane", "agent", "decision", "sim_time_s",
    "x", "y", "y_lane", "yaw", "z", "up_z", "fallen", "dist", "bearing",
    "r_m60", "r_m30", "r_0", "r_p30", "r_p60",
    "agent_action", "agent_action_name", "final_action", "final_action_name",
    "shield_intervened", "lane_done", "outcome",
]


def _f(v):
    return "%.4f" % v


def _name(a):
    return ACTION_NAMES[a] if isinstance(a, int) and 0 <= a < len(ACTION_NAMES) else ""


class DecisionLogger(object):
    def __init__(self, out_dir, run_id, agent_names, filename="decisions.csv"):
        self.run_id, self.agent_names = run_id, list(agent_names)
        self.path = os.path.join(out_dir, filename)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._w.writeheader()
        self._fh.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def _row(self, episode, lane, decision, t, state, agent_action, final_action,
             intervened, done, outcome):
        r = state["ranges"]
        return {
            "run_id": self.run_id, "episode": episode, "lane": lane.index,
            "agent": self.agent_names[lane.index], "decision": decision, "sim_time_s": _f(t),
            "x": _f(state["x"]), "y": _f(state["y"]), "y_lane": _f(state["y"] - lane.origin_y),
            "yaw": _f(state["yaw"]), "z": _f(state["z"]), "up_z": _f(state["up_z"]),
            "fallen": int(bool(state["fallen"])),
            "dist": _f(state["dist"]), "bearing": _f(state["bearing"]),
            "r_m60": _f(r[0]), "r_m30": _f(r[1]), "r_0": _f(r[2]), "r_p30": _f(r[3]), "r_p60": _f(r[4]),
            "agent_action": "" if agent_action is None else agent_action,
            "agent_action_name": "" if agent_action is None else _name(agent_action),
            "final_action": "" if final_action is None else final_action,
            "final_action_name": "" if final_action is None else _name(final_action),
            "shield_intervened": "" if intervened is None else int(bool(intervened)),
            "lane_done": int(bool(done)), "outcome": outcome,
        }

    def log_reset(self, episode, lanes, t):
        """Pose inicial de cada carril tras reset()+calentamiento (decision = -1)."""
        for lane in lanes:
            self._w.writerow(self._row(episode, lane, -1, t, lane.state, None, None, None,
                                       lane.done, lane.outcome))
        self._fh.flush()

    def log_step(self, episode, t_before, lanes, infos):
        """Una fila por carril activo en esta decision. t_before = tiempo al decidir."""
        for lane, info in zip(lanes, infos):
            if not info:          # carril ya terminado antes de esta decision
                continue
            self._w.writerow(self._row(episode, lane, info["decision"], t_before, info["pre_state"],
                                       info["agent_action"], info["final_action"], info["intervened"],
                                       info["done"], info["outcome"]))
        self._fh.flush()
