"""Supervisor de Webots con entrenamiento y persistencia de agentes."""
import csv
import json
import os
import time

from controller import Supervisor

from agents import build_agent
from diagnostics import DecisionLogger
from race_env import N_ACTIONS, OBS_DIM, RaceEnv, SimulationEnded

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIELDS = ["episode", "lane", "agent", "outcome", "return", "decisions", "collisions",
          "interventions", "finish_time_s", "final_dist", "start_y", "goal_x", "winner"]
LEARNABLE = {"dqn", "a2c", "a2c_v2", "ppo"}


def make_run_dir(base, run_id):
    candidate, k = run_id, 1
    while True:
        path = os.path.join(base, candidate)
        try:
            os.makedirs(path)
            return candidate, path
        except FileExistsError:
            k += 1
            candidate = "%s_%d" % (run_id, k)


def resolve_path(path):
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def save_agents(agents, out_dir, episode=None):
    """Guarda checkpoint por carril para poder reanudar o evaluar."""
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        if agent.name not in LEARNABLE:
            continue
        filename = "robot_%d_%s.pt" % (i, agent.name)
        agent.save(os.path.join(model_dir, filename))
    state = {"episode_saved": episode, "agents": [
        {"lane": i, "name": a.name,
         "checkpoint": ("models/robot_%d_%s.pt" % (i, a.name)) if a.name in LEARNABLE else None}
        for i, a in enumerate(agents)]}
    with open(os.path.join(out_dir, "checkpoint_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def load_agents(agents, cfg):
    """Carga rutas indicadas en modelLoad, cuyas claves son Ã­ndices de carril."""
    requested = cfg.get("modelLoad", {})
    if isinstance(requested, list):
        requested = {str(i): p for i, p in enumerate(requested) if p}
    for i, agent in enumerate(agents):
        path = resolve_path(requested.get(str(i), requested.get(i)))
        if path:
            if agent.name not in LEARNABLE:
                raise ValueError("No se puede cargar checkpoint para baseline en carril %d" % i)
            if not os.path.isfile(path):
                raise FileNotFoundError("Checkpoint no encontrado: %s" % path)
            agent.load(path)
            print("[supervisor] cargado carril %d (%s): %s" % (i, agent.name, path))


def main():
    cfg_path = os.environ.get("SCENARIO", os.path.join(ROOT, "config", "scenario.json"))
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base_id = cfg.get("runId") or time.strftime("run_%Y%m%d_%H%M%S")
    run_id, out_dir = make_run_dir(os.path.join(ROOT, "results"), base_id)
    print("[supervisor] resultados en", out_dir)
    with open(os.path.join(out_dir, "scenario_snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    sup = Supervisor()
    env = RaceEnv(sup, cfg)
    agent_params = cfg.get("agentParams", {})
    agents = [build_agent(n, OBS_DIM, N_ACTIONS, seed=cfg["seed"] + i,
                          params=agent_params.get(n, {}))
              for i, n in enumerate(cfg["agents"][:len(env.lanes)])]
    if len(agents) != len(env.lanes):
        raise RuntimeError("scenario.agents debe tener un agente por carril")
    load_agents(agents, cfg)
    training = cfg["mode"] == "train"
    save_enabled = bool(cfg.get("saveModels", True))

    try:
        with open(os.path.join(out_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as fh, \
                DecisionLogger(out_dir, run_id, [a.name for a in agents]) as dlog:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            for ep in range(cfg["episode"]["count"]):
                obs = env.reset()
                dlog.log_reset(ep, env.lanes, sup.getTime())
                while not env.all_done():
                    actions = [agents[i].act(obs[i], training) if not env.lanes[i].done else 0
                               for i in range(len(agents))]
                    was_done = [l.done for l in env.lanes]
                    t_before = sup.getTime()
                    nxt, rewards, dones, infos = env.step(actions)
                    dlog.log_step(ep, t_before, env.lanes, infos)
                    for i, ag in enumerate(agents):
                        if not was_done[i]:
                            info = infos[i] or {}
                            if ag.name in ("a2c", "a2c_v2"):
                                ag.observe(
                                    obs[i], actions[i], rewards[i], nxt[i], dones[i],
                                    intervened=bool(info.get("intervened", False)),
                                    executed_action=info.get("final_action", actions[i])
                                )
                            else:
                                ag.observe(obs[i], actions[i], rewards[i], nxt[i], dones[i])
                    obs = nxt
                finishers = [l for l in env.lanes if l.finish_time is not None]
                winner = min(finishers, key=lambda l: l.finish_time).index if finishers else ""
                for i, lane in enumerate(env.lanes):
                    row = env.summary(lane)
                    row.update(episode=ep, agent=agents[i].name,
                               winner=int(winner == i) if winner != "" else 0)
                    writer.writerow({k: row.get(k, "") for k in FIELDS})
                    agents[i].end_episode(row)
                fh.flush()
                if save_enabled and training:
                    save_agents(agents, out_dir, episode=ep)
                print("[supervisor] episodio %d: %s" %
                      (ep, [(l.outcome, round(l.ret, 1)) for l in env.lanes]))
    except SimulationEnded:
        print("[supervisor] simulacion terminada por Webots")
    finally:
        if save_enabled and any(a.name in LEARNABLE for a in agents):
            save_agents(agents, out_dir, episode="final")
    print("[supervisor] listo; resultados en", out_dir)
    if cfg.get("quitWhenFinished"):
        sup.simulationQuit(0)
    else:
        sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()

