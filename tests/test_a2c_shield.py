"""Pruebas unitarias del A2C con transiciones intervenidas. No requiere Webots."""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "controllers", "race_supervisor"))

from agents.a2c import A2CAgent  # noqa: E402


def make_episode(agent, all_intervened=False):
    for i in range(24):
        obs = [0.1 * i, 0.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4]
        nxt = [0.1 * (i + 1), 0.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4]
        intervened = all_intervened or (i % 3 == 0)
        agent.observe(
            obs, i % 5, 0.2 if i < 23 else 1.0, nxt, i == 23,
            intervened=intervened,
            executed_action=3 if intervened else i % 5
        )


def main():
    agent = A2CAgent(8, 5, seed=123)
    make_episode(agent, all_intervened=False)
    agent.end_episode({"outcome": "test"})
    assert not agent.rollout
    assert agent.last_actor_samples == 16
    assert agent.last_intervened_samples == 8
    assert agent.last_loss is not None
    assert agent.last_critic_loss >= 0.0

    # Incluso si todas las acciones fueron intervenidas, el crítico debe actualizarse
    # y el actor debe recibir cero muestras de política.
    make_episode(agent, all_intervened=True)
    agent.end_episode({"outcome": "test"})
    assert agent.last_actor_samples == 0
    assert agent.last_intervened_samples == 24
    assert agent.last_loss is not None

    # Checkpoint compatible: carga de pesos sin restaurar estado del optimizador.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a2c_test.pt")
        agent.save(path)
        restored = A2CAgent(8, 5, seed=999)
        restored.load(path)
        assert all(
            (a == b).all()
            for a, b in zip(agent.model.state_dict().values(),
                            restored.model.state_dict().values())
        )

    print("OK A2C shield: transiciones mixtas, episodio totalmente intervenido y checkpoint")


if __name__ == "__main__":
    main()
