"""
src/envs/mcs_control.py
Link-adaptation wrappers that take MCS selection away from the learner.

The learner then decides only *who transmits to whom* (STA index or mute); the MCS half of
each AP's action is ignored and replaced here. This isolates the coordination decision from
rate selection, which in 802.11 is the PHY/MAC link adaptation's job anyway.

  ARFMCS   : Auto Rate Fallback per (AP, STA) link from ACK success/failure — realistic.
  GenieMCS : best decodable MCS for the transmit set the APs actually chose — perfect link
             adaptation, a reference that needs the other APs' decisions (not implementable).
"""
from __future__ import annotations

import numpy as np

from src.training.oracle_policy import prospective_harsh, schedule_value

ARF_UP = 10
ARF_DOWN = 2


class _MCSWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _transmit_set(self, action: np.ndarray) -> tuple[list[int], np.ndarray]:
        """STA per AP that will actually transmit this step (-1 = silent) and post-arrival queues."""
        queues = np.clip(self.env.queues() + self.env.peek_arrivals(), 0, self.env.queue_max_size)
        stas = []
        for i, sta in enumerate(np.asarray(action).reshape(-1)[0::2]):
            sta = int(sta)
            real = sta < len(self.env.topo.sta_positions[i])
            stas.append(sta if real and queues[i, sta] > 0 else -1)
        return stas, queues

    def _with_mcs(self, action: np.ndarray, mcs: list[int]) -> np.ndarray:
        action = np.asarray(action, dtype=np.int32).reshape(-1).copy()
        action[1::2] = mcs
        return action

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GenieMCS(_MCSWrapper):
    def step(self, action):
        stas, queues = self._transmit_set(action)
        saved = (self.env.overlap_policy.blocked_count, self.env.overlap_policy.soft_penalty_count)
        try:
            _, mcs = schedule_value(self.env, stas, queues, prospective_harsh(self.env))
        finally:
            self.env.overlap_policy.blocked_count, self.env.overlap_policy.soft_penalty_count = saved
        return self.env.step(self._with_mcs(action, mcs))


class ARFMCS(_MCSWrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.mcs: dict[tuple[int, int], int] = {}
        self.n_ok: dict[tuple[int, int], int] = {}
        self.n_fail: dict[tuple[int, int], int] = {}
        self.probing: dict[tuple[int, int], bool] = {}
        return out

    def _initial_mcs(self, ap: int, sta: int) -> int:
        """Start from the interference-free SINR, as the local ARF baseline does."""
        saved = (self.env.overlap_policy.blocked_count, self.env.overlap_policy.soft_penalty_count)
        try:
            result = self.env.interference.compute(ap, sta, {ap}, is_harsh=prospective_harsh(self.env))
        finally:
            self.env.overlap_policy.blocked_count, self.env.overlap_policy.soft_penalty_count = saved
        return self.env.mcs_table.best_mcs_for(result.sinr_db)

    def step(self, action):
        stas, _ = self._transmit_set(action)
        mcs = []
        for ap, sta in enumerate(stas):
            if sta >= 0 and (ap, sta) not in self.mcs:
                key = (ap, sta)
                self.mcs[key], self.n_ok[key], self.n_fail[key], self.probing[key] = self._initial_mcs(ap, sta), 0, 0, False
            mcs.append(self.mcs[(ap, sta)] if sta >= 0 else 0)

        out = self.env.step(self._with_mcs(action, mcs))
        thp = out[4]["thp"]
        top = self.env.mcs_table.n_mcs - 1
        for ap, sta in enumerate(stas):
            if sta < 0:
                continue
            key = (ap, sta)
            if thp[ap] > 0:
                self.n_ok[key] += 1
                self.n_fail[key] = 0
                self.probing[key] = False
                if self.n_ok[key] >= ARF_UP:
                    self.mcs[key] = min(self.mcs[key] + 1, top)
                    self.n_ok[key] = 0
                    self.probing[key] = True
            else:
                self.n_fail[key] += 1
                self.n_ok[key] = 0
                if self.probing[key] or self.n_fail[key] >= ARF_DOWN:
                    self.mcs[key] = max(self.mcs[key] - 1, 0)
                    self.n_fail[key] = 0
                self.probing[key] = False
        return out


MCS_CONTROL = {"learned": None, "arf": ARFMCS, "genie": GenieMCS}
