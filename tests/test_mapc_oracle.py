"""Mute action, real-STA traffic, exact oracle and PPO learners on env v2 (qh_v2)."""
import itertools
from dataclasses import replace

import numpy as np
import pytest

from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.env_factory import make_env
from src.envs.topology_registry import get_registry
from src.training.baseline_policies import BaselinePolicy
from src.training.oracle_policy import exact_oracle_actions, prospective_harsh, schedule_value


def make(topo_id="T6", load=1.0, steps=20, allow_mute=True, overlap_mode="hard", fading="none"):
    experiment = ExperimentConfig.from_dict(load_config("configs/qh_ctde.yaml"))
    traffic = dict(experiment.env.traffic, traffic_load_ratio=0.0,
                   poisson_lambda_min=load, poisson_lambda_max=load)
    overlap = dict(experiment.env.overlap, mode=overlap_mode)
    cfg = replace(experiment.env, impl="qh_v2", max_episodes=steps, traffic=traffic,
                  allow_mute=allow_mute, overlap=overlap, fading={"model": fading})
    return make_env(cfg, topology=get_registry().get(topo_id), topology_file=experiment.topology_file)


def test_mute_extends_action_space_and_silences_ap():
    env = make()
    assert env.action_space.nvec[0] == env.k_stas + 1 and env.mute_index == env.k_stas
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([env.mute_index, 0] * env.n_aps))
    assert info["system_throughput"] == 0
    assert make(allow_mute=False).mute_index is None


def test_padding_stas_receive_no_traffic():
    env = make("T1", load=5.0)  # one real STA per AP, K = 4
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.array([env.mute_index, 0] * env.n_aps))
    assert (env.queues()[:, 1:] == 0).all()
    assert (env.queues()[:, 0] > 0).all()


def test_peek_arrivals_does_not_consume_rng():
    env = make()
    env.reset(seed=3)
    before = env.queues().copy()
    peeked = env.peek_arrivals()
    env.step(np.array([env.mute_index, 0] * env.n_aps))
    np.testing.assert_allclose(env.queues(), np.clip(before + peeked, 0, env.queue_max_size))


@pytest.mark.parametrize("overlap_mode,fading", [("hard", "none"), ("soft", "none"), ("hard", "rayleigh")])
def test_exact_oracle_matches_independent_brute_force(overlap_mode, fading):
    env = make("T3", overlap_mode=overlap_mode, fading=fading)
    env.reset(seed=1)
    queues = np.clip(env.queues() + env.peek_arrivals(), 0, env.queue_max_size)
    harsh = prospective_harsh(env)
    actions = exact_oracle_actions(env)
    stas = [-1 if a >= len(env.topo.sta_positions[i]) else int(a) for i, a in enumerate(actions[0::2])]
    value, _ = schedule_value(env, stas, queues, harsh)

    # Brute force over STA sets scanning every MCS, independent of best_mcs_for().
    best = 0.0
    for combo in itertools.product(*[[-1] + list(range(n)) for n in env.valid_sta_counts]):
        tx = [i for i, s in enumerate(combo) if s >= 0]
        total = 0.0
        for i in tx:
            r = env.interference.compute(i, combo[i], tx, is_harsh=harsh)
            if r.blocked_by_overlap:
                continue
            total += max([min(queues[i, combo[i]], env.mcs_table.capacity(m))
                          for m in range(env.mcs_table.n_mcs) if env.mcs_table.decodable(r.sinr_db, m)] or [0.0])
        best = max(best, total)
    assert value == pytest.approx(best)


@pytest.mark.parametrize("topo_id", ["T1", "T3", "T5", "T6"])
def test_exact_oracle_dominates_baselines_on_same_arrivals(topo_id):
    def run(name):
        env, policy = make(topo_id), BaselinePolicy(name, seed=0)
        env.reset(seed=7)
        total, done = 0.0, False
        while not done:
            _, _, done, _, info = env.step(policy.act(env))
            policy.observe(info)
            total += info["system_throughput"]
        return total

    exact = run("exact_oracle")
    for name in ("random", "greedy", "round_robin", "oracle", "co_tdma"):
        assert exact >= run(name), name


def test_ctde_mask_keeps_mute_available():
    torch = pytest.importorskip("torch")
    from src.agents.ctde_ppo import CTDEActorCritic

    env = make("T6")
    model = CTDEActorCritic(env.obs_dim, env.n_aps, env.k_stas + 1, 16, 16, 4,
                            torch.ones(env.n_aps, env.n_aps), mute_index=env.mute_index)
    sta_logits, _ = model.actor_logits(torch.zeros(1, env.n_aps, env.obs_dim),
                                       torch.as_tensor([env.valid_sta_counts]))
    allowed = sta_logits[0] > -1e8
    for i, n_valid in enumerate(env.valid_sta_counts):
        assert allowed[i, :n_valid].all() and allowed[i, env.mute_index]
        assert not allowed[i, n_valid:env.k_stas].any()


@pytest.mark.parametrize("learner", ["MAPPO", "CentralPPO"])
def test_ppo_learners_smoke(learner):
    pytest.importorskip("torch")
    from src.agents import mappo

    env = make("T6", steps=8)
    cls = getattr(mappo, learner)
    agent = cls(env.obs_dim, env.n_aps, env.k_stas, env.valid_sta_counts, hidden_dim=16)
    assert len(mappo.train_mappo(env, agent, episodes=2, episodes_per_update=2)) == 2
    obs, _ = env.reset(seed=0)
    action, log_prob, _ = agent.act(obs)
    assert action.shape == (2 * env.n_aps,) and np.isfinite(log_prob)
    for i, n_valid in enumerate(env.valid_sta_counts):
        assert action[2 * i] < n_valid or action[2 * i] == env.mute_index


@pytest.mark.parametrize("wrapper", ["arf", "genie"])
def test_mcs_control_overrides_learner_mcs(wrapper):
    from src.envs.mcs_control import MCS_CONTROL

    env = MCS_CONTROL[wrapper](make("T6", load=5.0))
    env.reset(seed=0)
    # One AP transmits, the rest mute: ARF's interference-free starting MCS must decode.
    # The learner proposes MCS 0 (1 packet/slot); link adaptation must raise it.
    action = np.array([env.mute_index, 0] * env.n_aps)
    action[0:2] = [0, 0]
    _, _, _, _, info = env.step(action)
    assert info["system_throughput"] > env.mcs_table.capacity(0)


def test_genie_mcs_matches_exact_value_for_chosen_set():
    from src.envs.mcs_control import GenieMCS

    env = GenieMCS(make("T3", load=5.0))
    env.reset(seed=2)
    stas, queues = env._transmit_set(np.array([0, 0] * env.n_aps))
    expected, _ = schedule_value(env.env, stas, queues, prospective_harsh(env.env))
    _, _, _, _, info = env.step(np.array([0, 0] * env.n_aps))
    assert info["system_throughput"] == pytest.approx(expected)


def test_no_fading_is_deterministic_path_loss():
    env = make("T6")
    env.reset(seed=0)
    assert env.interference.fading_gain is None


def test_rayleigh_fading_redraws_each_slot_without_touching_arrivals():
    plain, faded = make("T6", load=5.0, steps=50), make("T6", load=5.0, steps=50, fading="rayleigh")
    plain.reset(seed=4)
    faded.reset(seed=4)
    gains = []
    silence = np.array([faded.mute_index, 0] * faded.n_aps)
    for _ in range(40):
        gains.append(faded.interference.fading_gain.copy())
        plain.step(silence)
        faded.step(silence)
    # Arrivals use their own RNG, so queues match the no-fading run exactly.
    np.testing.assert_allclose(plain.queues(), faded.queues())
    gains = np.stack(gains)
    assert not np.allclose(gains[0], gains[1])
    assert 0.85 < gains.mean() < 1.15  # Exp(1) power gain


def test_fading_does_not_change_mean_snr_observation():
    plain, faded = make("T3"), make("T3", fading="rayleigh")
    np.testing.assert_allclose(plain.reset(seed=1)[0], faded.reset(seed=1)[0])


def make_random(topo_id, load=8.0, steps=30, overlap_mode="hard", fading="none"):
    experiment = ExperimentConfig.from_dict(load_config("configs/qh_ctde.yaml"))
    traffic = dict(experiment.env.traffic, traffic_load_ratio=0.0,
                   poisson_lambda_min=load, poisson_lambda_max=load)
    overlap = dict(experiment.env.overlap, mode=overlap_mode)
    cfg = replace(experiment.env, impl="qh_v2", max_episodes=steps, traffic=traffic,
                  allow_mute=True, overlap=overlap, fading={"model": fading})
    return make_env(cfg, topology=get_registry("configs/topologies_random.yaml").get(topo_id))


@pytest.mark.parametrize("overlap_mode,fading", [("hard", "none"), ("soft", "rayleigh")])
def test_vectorized_oracle_matches_reference_on_large_topologies(overlap_mode, fading):
    from src.training.oracle_policy import exact_oracle_actions_reference

    registry = get_registry("configs/topologies_random.yaml")
    big = [t for t in registry.ids() if registry.get(t).n_aps == 6 and registry.get(t).n_overlap_stas() > 0][:1]
    for topo_id in big:
        env = make_random(topo_id, steps=8, overlap_mode=overlap_mode, fading=fading)
        env.reset(seed=0)
        done = False
        while not done:
            fast, ref = exact_oracle_actions(env), exact_oracle_actions_reference(env)
            np.testing.assert_array_equal(fast, ref)
            _, _, done, _, _ = env.step(ref)


def test_random_topologies_follow_design_variables():
    import yaml
    from src.envs.random_topology import RANGES, build_spec, design_points

    raw = yaml.safe_load(open("configs/topologies_random.yaml"))
    registry = get_registry("configs/topologies_random.yaml")
    assert len(raw["groups"]["stage1"]) == 200 and len(raw["groups"]["stage2"]) == 20
    for topo_id in registry.ids():
        design, topo = raw["topologies"][topo_id]["design"], registry.get(topo_id)
        aps = topo.ap_positions
        d_min = min(np.linalg.norm(aps[i] - aps[j]) for i in range(topo.n_aps) for j in range(i))
        assert d_min == pytest.approx(design["d"], abs=0.05)
        assert topo.n_aps == design["n_aps"]
        assert all(len(s) == design["stas_per_ap"] for s in topo.sta_positions)
        for name, (lo, hi) in RANGES.items():
            assert lo <= design[name] <= hi
    # Same seed -> same topology.
    point = design_points(5, seed=1)[3]
    assert build_spec(point, seed=9) == build_spec(point, seed=9)
