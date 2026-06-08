"""tests/test_env.py — OBSSEnv 기본 동작 검증"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.envs.topology import topo_cw1, topo_cw3, random_topology
from src.envs.obss_env import OBSSEnv


def test_reset_shape():
    env = OBSSEnv(topo_cw1(), mode="cw", interact_slots=1000, max_episodes=5)
    obs, info = env.reset()
    assert obs.shape == (3, 6), f"Expected (3,6), got {obs.shape}"


def test_step_cw():
    env = OBSSEnv(topo_cw1(), mode="cw", interact_slots=1000, max_episodes=5)
    env.reset()
    actions = np.array([0, 0, 0], dtype=np.int32)
    obs, rew, done, _, info = env.step(actions)
    assert obs.shape == (3, 6)
    assert isinstance(rew, float)
    assert "d95" in info


def test_step_cca():
    env = OBSSEnv(topo_cw3(), mode="cca", interact_slots=1000, max_episodes=5)
    env.reset()
    actions = np.array([2, 2, 2, 2], dtype=np.int32)
    obs, rew, done, _, info = env.step(actions)
    assert obs.shape == (4, 6)


def test_random_topo():
    rng  = np.random.default_rng(0)
    topo = random_topology(rng=rng)
    assert topo.n_aps == 4
    env  = OBSSEnv(topo, mode="cw", interact_slots=500, max_episodes=3)
    env.reset()
    actions = np.zeros(4, dtype=np.int32)
    obs, rew, done, _, info = env.step(actions)
    assert obs.shape == (4, 6)
