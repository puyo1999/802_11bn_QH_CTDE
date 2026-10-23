"""Non-learned MAPC schedulers for the qh scheduling environment (allow_mute=True).

All policies return the flat env action ``[sta_0, mcs_0, sta_1, mcs_1, ...]`` where
``sta == env.mute_index`` means the AP stays silent for the slot.

* GenieOracle      — exact per-slot throughput optimum over every joint (STA | mute)
                     choice, with the best feasible MCS per link. It knows the next
                     slot's arrivals (genie), so it upper-bounds any per-slot scheduler.
* RandomScheduler  — uniform valid STA and MCS, never mutes (legacy "Random").
* LocalGreedy      — 0-bit heuristic: largest queue, MCS from the noise-only SNR.
                     Ignores interference entirely (uncoordinated link adaptation).
* CoTDMA           — one AP per slot, round-robin over APs with traffic, everyone
                     else muted; MCS from the noise-only SNR (no interference exists).
"""
from __future__ import annotations

import itertools

import numpy as np

from src.envs.qh_obss_env_mod1 import MCS_CAPACITY, MCS_SINR_THRESH, OBSSEnv

_THRESH = np.asarray(MCS_SINR_THRESH)


def best_mcs(sinr_db: float) -> int | None:
    """Highest MCS whose SINR threshold is met (capacity is monotone in MCS)."""
    feasible = np.flatnonzero(_THRESH <= sinr_db)
    return int(feasible[-1]) if feasible.size else None


def _flat_action(env: OBSSEnv, stas: list[int], mcs: list[int]) -> np.ndarray:
    action = []
    for sta, m in zip(stas, mcs):
        action.extend((env.mute_index if sta == -1 else sta, m))
    return np.asarray(action, dtype=np.int32)


def _snr_only_mcs(env: OBSSEnv, stas: list[int]) -> list[int]:
    """MCS each AP would pick if it assumed nobody else transmits."""
    mcs = []
    for i, sta in enumerate(stas):
        if sta == -1:
            mcs.append(0)
            continue
        solo = [-1] * env.n_aps
        solo[i] = sta
        m = best_mcs(env.link_sinr_db(solo)[i])
        mcs.append(0 if m is None else m)
    return mcs


class GenieOracle:
    name = "Genie Oracle"

    def __init__(self, env: OBSSEnv):
        self.env = env

    def solve(self, queues: np.ndarray) -> tuple[list[int], list[int], float]:
        env = self.env
        options = []
        for i in range(env.n_aps):
            n_valid = min(env.valid_sta_counts[i], env.k_stas)
            options.append([-1] + [k for k in range(n_valid) if queues[i, k] > 0])

        best = ([-1] * env.n_aps, [0] * env.n_aps, 0.0)
        for stas in itertools.product(*options):
            sinr = env.link_sinr_db(list(stas))
            mcs, value = [], 0.0
            for i, sta in enumerate(stas):
                m = None if sta == -1 else best_mcs(sinr[i])
                mcs.append(0 if m is None else m)
                if m is not None:
                    value += min(queues[i, sta], MCS_CAPACITY[m])
            if value > best[2]:
                best = (list(stas), mcs, value)
        return best

    def act(self, obs: np.ndarray) -> np.ndarray:
        queues = np.clip(self.env.queues() + self.env.peek_arrivals(), 0, self.env.queue_max_size)
        stas, mcs, _ = self.solve(queues)
        return _flat_action(self.env, stas, mcs)


class RandomScheduler:
    name = "Random"

    def __init__(self, env: OBSSEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = []
        for n_valid in self.env.valid_sta_counts:
            action.extend((int(self.rng.integers(min(n_valid, self.env.k_stas))), int(self.rng.integers(16))))
        return np.asarray(action, dtype=np.int32)


class LocalGreedy:
    name = "Local Greedy (0-bit)"

    def __init__(self, env: OBSSEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> np.ndarray:
        queues = self.env.queues()
        stas = []
        for i in range(self.env.n_aps):
            n_valid = min(self.env.valid_sta_counts[i], self.env.k_stas)
            k = int(np.argmax(queues[i, :n_valid]))
            stas.append(k if queues[i, k] > 0 else -1)
        return _flat_action(self.env, stas, _snr_only_mcs(self.env, stas))


class CoTDMA:
    name = "Co-TDMA (RR)"

    def __init__(self, env: OBSSEnv):
        self.env = env
        self.turn = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        queues = self.env.queues()
        stas = [-1] * self.env.n_aps
        for offset in range(self.env.n_aps):
            i = (self.turn + offset) % self.env.n_aps
            n_valid = min(self.env.valid_sta_counts[i], self.env.k_stas)
            if queues[i, :n_valid].sum() > 0:
                stas[i] = int(np.argmax(queues[i, :n_valid]))
                self.turn = i + 1
                break
        return _flat_action(self.env, stas, _snr_only_mcs(self.env, stas))
