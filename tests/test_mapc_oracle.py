import itertools

import numpy as np
import pytest

from src.agents.mapc_baselines import CoTDMA, GenieOracle, LocalGreedy, RandomScheduler, best_mcs
from src.envs.qh_obss_env_mod1 import MCS_CAPACITY, OBSSEnv
from src.envs.topology import load_fixed_topologies


def make_env(name="T6", **kwargs):
    defaults = dict(mode="qh", max_episodes=20, k_stas=4, poisson_lambda=1.0, allow_mute=True)
    defaults.update(kwargs)
    return OBSSEnv(load_fixed_topologies()[name], **defaults)


def test_harsh_mode_is_independent_of_one_step_horizon():
    # Before the fix, max_episodes=1 made `step >= 1 // 2` true from step 0 (always harsh).
    env = make_env(max_episodes=1, harsh_start_step=100)
    env.reset(seed=0)
    assert not env.is_harsh
    assert make_env(max_episodes=200).harsh_start_step == 100  # legacy default kept


def test_mute_action_removes_interference_and_transmission():
    env = make_env("T6")
    env.reset(seed=0)
    all_tx = env.link_sinr_db([0, 0, 0, 0])
    solo = env.link_sinr_db([0, -1, -1, -1])
    assert solo[0] > all_tx[0]
    assert np.isneginf(solo[1:]).all()

    env.reset(seed=0)
    mute = env.mute_index
    _, _, _, _, info = env.step(np.array([mute, 0] * env.n_aps))
    assert info["system_throughput"] == 0


def test_padding_stas_receive_no_traffic():
    env = make_env("T1", poisson_lambda=5.0)  # T1 has one real STA per AP
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.array([env.mute_index, 0] * env.n_aps))
    assert (env.queues()[:, 1:] == 0).all()
    assert (env.queues()[:, 0] > 0).all()


def test_padding_sinr_observation_uses_mw():
    env = make_env("T1")
    obs, _ = env.reset(seed=0)
    k = env.k_stas
    # -110 dBm vs -95 dBm noise floor => -15 dB SNR, normalised by 40.
    assert obs[0, k + 1] == pytest.approx(-15.0 / 40.0, abs=1e-4)


def test_peek_arrivals_does_not_consume_rng():
    env = make_env()
    env.reset(seed=3)
    q0 = env.queues().copy()
    peeked = env.peek_arrivals()
    env.step(np.array([env.mute_index, 0] * env.n_aps))
    np.testing.assert_allclose(env.queues(), np.clip(q0 + peeked, 0, env.queue_max_size))


def test_oracle_matches_brute_force_over_flat_actions():
    env = make_env("T3")
    env.reset(seed=1)
    queues = np.clip(env.queues() + env.peek_arrivals(), 0, env.queue_max_size)
    _, _, value = GenieOracle(env).solve(queues)

    # Brute force over every joint (STA | mute, MCS) is too large; instead enumerate STA
    # choices and pick each link's MCS by scanning all 16 — independent of best_mcs().
    best = 0.0
    options = [[-1] + list(range(n)) for n in env.valid_sta_counts]
    for stas in itertools.product(*options):
        sinr = env.link_sinr_db(list(stas))
        total = 0.0
        for i, sta in enumerate(stas):
            if sta == -1:
                continue
            total += max([min(queues[i, sta], MCS_CAPACITY[m]) for m in range(16)
                          if sinr[i] >= 2.0 + 1.5 * m] or [0.0])
        best = max(best, total)
    assert value == pytest.approx(best)


@pytest.mark.parametrize("name", ["T1", "T3", "T5", "T6"])
def test_oracle_dominates_heuristics_on_same_arrivals(name):
    def run(policy_cls, **kw):
        env = make_env(name)
        policy = policy_cls(env, **kw)
        obs, _ = env.reset(seed=7)
        total, done = 0.0, False
        while not done:
            obs, _, done, _, info = env.step(policy.act(obs))
            total += info["system_throughput"]
        return total

    oracle = run(GenieOracle)
    for cls in (LocalGreedy, CoTDMA):
        assert oracle >= run(cls)
    assert oracle >= run(RandomScheduler, seed=0)


def test_best_mcs_edges():
    assert best_mcs(1.9) is None
    assert best_mcs(2.0) == 0
    assert best_mcs(100.0) == 15


def test_mappo_rollout_and_update_smoke():
    torch = pytest.importorskip("torch")
    from src.agents.mappo import MAPPO, train_mappo

    env = make_env("T6", max_episodes=8)
    learner = MAPPO(env.observation_space.shape[-1], env.n_aps, env.k_stas, env.valid_sta_counts, hidden_dim=16)
    history = train_mappo(env, learner, episodes=2)
    assert len(history) == 2
    obs, _ = env.reset(seed=0)
    action, log_prob, _ = learner.act(obs)
    assert action.shape == (2 * env.n_aps,)
    # Padding STAs (T6 AP2 has one real STA) are never sampled; mute (index K) is allowed.
    for i, n_valid in enumerate(env.valid_sta_counts):
        assert action[2 * i] < n_valid or action[2 * i] == env.mute_index
    assert np.isfinite(log_prob)
