import numpy as np

from src.agents.classical_ctde import ClassicalCTDE
from src.config import load_config
from src.envs.factory import make_scheduling_env, validate_state_contract
from src.envs.topology import load_fixed_topologies


def test_yaml_topologies_include_shared_sta_conditions():
    topologies = load_fixed_topologies()
    assert set(topologies) == {f"T{i}" for i in range(1, 7)}
    for name in ("T3", "T6"):
        group = topologies[name].shared_sta_groups[0]
        first_ap, first_sta = group[0]
        reference = topologies[name].sta_positions[first_ap][first_sta]
        assert all(np.allclose(reference, topologies[name].sta_positions[ap][sta]) for ap, sta in group[1:])


def test_dynamic_state_contract_and_classical_ctde_update():
    cfg = load_config("configs/classical_ctde.yaml")
    topo = load_fixed_topologies()["T3"]
    env = make_scheduling_env(topo, cfg, episode_horizon=1)
    contract = validate_state_contract(env, cfg)
    assert contract.local_dim == 2 * cfg["env"]["k_stas"] + 2
    assert contract.global_dim == contract.local_dim * topo.n_aps

    learner = ClassicalCTDE(contract.local_dim, contract.n_aps, cfg["env"]["k_stas"], hidden_dim=16)
    obs, _ = env.reset(seed=0)
    action, log_prob = learner.select_actions(obs, [len(stas) for stas in topo.sta_positions])
    next_obs, reward, done, _, _ = env.step(action)
    result = learner.update(obs, log_prob, reward, next_obs, done)
    assert np.isfinite(result.actor_loss)
    assert np.isfinite(result.critic_loss)
