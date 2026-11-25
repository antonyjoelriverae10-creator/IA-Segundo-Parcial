"""Prueba (sin Webots) del registro de diagnostico: ejecuta el supervisor REAL con un modulo
'controller' falso y un Supervisor cinematico, y valida decisions.csv contra episodes.csv.
Escribe temporalmente en results/ y borra lo que crea. Uso: python tests/test_diagnostics.py"""
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "controllers", "race_supervisor"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_env_mock as T  # noqa: E402  (FakeSup)


class FakeSupervisor(T.FakeSup):
    SIMULATION_MODE_PAUSE = 0

    def __init__(self):
        T.FakeSup.__init__(self, 4)

    def simulationSetMode(self, m): pass
    def simulationQuit(self, c): pass


fake = types.ModuleType("controller")
fake.Supervisor = FakeSupervisor
sys.modules["controller"] = fake
import diagnostics  # noqa: E402
import race_supervisor  # noqa: E402


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def run_supervisor(scn_path):
    os.environ["SCENARIO"] = scn_path
    race_supervisor.main()


def main():
    base = json.load(open(os.path.join(ROOT, "config", "scenario.json")))
    base["episode"]["count"] = 2
    base["runId"] = "unittest_diag_%d" % os.getpid()
    tmp = tempfile.mkdtemp()
    scn = os.path.join(tmp, "scenario.json")
    json.dump(base, open(scn, "w"))
    created = []
    try:
        run_supervisor(scn)
        run_supervisor(scn)          # misma runId: no debe sobrescribir
        dirs = sorted(d for d in os.listdir(os.path.join(ROOT, "results")) if d.startswith(base["runId"]))
        created = [os.path.join(ROOT, "results", d) for d in dirs]
        assert dirs == [base["runId"], base["runId"] + "_2"], dirs

        for d in created:
            dec, epi = read(os.path.join(d, "decisions.csv")), read(os.path.join(d, "episodes.csv"))
            assert list(dec[0].keys()) == diagnostics.COLUMNS
            assert all(r["run_id"] == os.path.basename(d) for r in dec)
            for e in epi:
                ep, lane = e["episode"], e["lane"]
                rows = [r for r in dec if r["episode"] == ep and r["lane"] == lane]
                reset = [r for r in rows if r["decision"] == "-1"]
                steps = [r for r in rows if r["decision"] != "-1"]
                assert len(reset) == 1 and reset[0]["agent_action"] == "" and reset[0]["shield_intervened"] == ""
                assert len(steps) == int(e["decisions"]), (ep, lane, len(steps), e["decisions"])
                assert [int(r["decision"]) for r in steps] == list(range(len(steps)))
                assert steps[-1]["lane_done"] == "1" and steps[-1]["outcome"] == e["outcome"]
                assert all(r["lane_done"] == "0" and r["outcome"] == "" for r in steps[:-1])
                assert all(r["agent"] == e["agent"] for r in rows)
                n_iv = 0
                for r in steps:
                    a, f, iv = int(r["agent_action"]), int(r["final_action"]), int(r["shield_intervened"])
                    assert 0 <= a <= 4 and 0 <= f <= 4
                    assert (a != f) == bool(iv), r           # intervino <=> la accion cambio
                    n_iv += iv
                    assert abs(float(r["bearing"])) <= math.pi + 1e-3 and abs(float(r["yaw"])) <= math.pi + 1e-3
                    assert all(0.0 <= float(r[k]) <= 1.5 + 1e-3 for k in ("r_m60", "r_m30", "r_0", "r_p30", "r_p60"))
                    gx = float(e["goal_x"])
                    assert abs(math.hypot(gx - float(r["x"]), float(r["y_lane"])) - float(r["dist"])) < 0.02, r
                assert n_iv == int(e["interventions"]), (n_iv, e["interventions"])
        # La segunda ejecucion es identica a la primera (determinista): mismos resultados
        a, b = read(os.path.join(created[0], "episodes.csv")), read(os.path.join(created[1], "episodes.csv"))
        assert a == b
        print("decisions.csv: %d filas; episodes.csv: %d filas; carpetas: %s" %
              (len(read(os.path.join(created[0], "decisions.csv"))), len(a), dirs))
        print("OK diagnostico (logica Python con doble cinematico; sin Webots)")
    finally:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
