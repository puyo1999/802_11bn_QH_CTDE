"""
src/cellular/marl_env.py
Multi-agent environment of paper3 for the learned schedulers (MAPPO and Central PPO of src/agents/mappo.py).

Agents. Every pair of a cluster cell c and a subband b is an agent, ordered as i = k * B + b for the k-th cluster
cell. The action of an agent is (UE slot, CQI): the UE slot j < U selects the j-th UE of the cell, j = U is silence;
the CQI entry is used only with learned CQI control (index q - 1 for CQI q in 1..15).

Observation of agent (c, b) (paper3 eq. obs): reported CQI indices of the U UEs of cell c on subband b divided by 15,
their PF weights divided by their maximum, their backlogs divided by their maximum under finite buffer, the mean
received power from the other cluster cells at the UEs of cell c in dB mapped to [0, 1], and the one-hot subband.

CQI control:
    learned   the actor selects the CQI
    olla      OLLA on the delayed report (linkadapt.LinkAdapter "local")
    ideal     the highest CQI decodable at the SINR of the slot for the selected targets (linkadapt.genie_cqi)

Reward: the slot utility (paper3 eq. utility) divided by the number of agents.
"""
from __future__ import annotations

import numpy as np

from src.cellular import cqi
from src.cellular.linkadapt import LinkAdapter, genie_cqi
from src.cellular.sim import CellularConfig, CellularSim

CQI_MODES = ("learned", "olla", "ideal")


class CellularMARLEnv:
    def __init__(self, cfg: CellularConfig, drop: int, cqi_mode: str = "ideal"):
        if cqi_mode not in CQI_MODES:
            raise ValueError(f"cqi_mode must be one of {CQI_MODES}")
        self.cfg, self.drop, self.cqi_mode = cfg, drop, cqi_mode
        self.sim = CellularSim(cfg, drop_seed=drop)
        sim = self.sim
        self.C, self.B, self.U = len(sim.cluster), sim.n_sub, sim.n_ue_cell
        self.n_aps = self.C * self.B                      # name used by src/agents/mappo.py
        self.k_stas = self.U
        self.valid_sta_counts = [self.U] * self.n_aps
        self.n_mcs = cqi.N_CQI if cqi_mode == "learned" else 1
        self.ues = np.array([np.nonzero(sim.serving == c)[0] for c in sim.cluster])     # (C, U) global UE index
        # mean received power from the other cluster cells at the UEs of each cell (dB), mapped to [0, 1]
        inter = []
        for k, c in enumerate(sim.cluster):
            others = [o for o in sim.cluster if o != c]
            inter.append(10 * np.log10(sim.rx_mean[np.ix_(others, self.ues[k])].mean() + 1e-30))
        self.inter = np.clip((np.array(inter) + 130.0) / 80.0, 0.0, 1.0)
        self.obs_dim = 2 * self.U + (0 if cfg.full_buffer else self.U) + 1 + self.B
        self.adapter = None

    # ------------------------------------------------------------------ gym-like interface
    def reset(self, seed: int = 0):
        self.sim.reset(seed=seed)
        self.adapter = LinkAdapter(self.sim, "local") if self.cqi_mode == "olla" else None
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        sim = self.sim
        rep = cqi.best_cqi(sim.reported_sinr()) / cqi.N_CQI                    # (n_ues, B)
        w = sim.weights()
        obs = np.zeros((self.n_aps, self.obs_dim), dtype=np.float32)
        for k in range(self.C):
            ue = self.ues[k]
            wk = w[ue] / w[ue].max()
            parts_common = [wk]
            if not self.cfg.full_buffer:
                q = sim.backlog_now()[ue]
                parts_common.append(q / max(q.max(), 1e-9))
            for b in range(self.B):
                onehot = np.zeros(self.B)
                onehot[b] = 1.0
                obs[k * self.B + b] = np.concatenate([rep[ue, b], *parts_common, [self.inter[k]], onehot])
        return obs

    def targets_from_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat action (n_agents * 2,) -> targets (C, B) global UE index or -1, learned CQI (C, B)."""
        a = np.asarray(action).reshape(self.C, self.B, 2)
        slot, q = a[..., 0], a[..., 1]
        targets = np.where(slot < self.U, self.ues[np.arange(self.C)[:, None], np.minimum(slot, self.U - 1)], -1)
        return targets.astype(np.int64), (q + 1).astype(np.int64)

    def cqis_for(self, targets: np.ndarray, learned: np.ndarray) -> np.ndarray:
        if self.cqi_mode == "learned":
            return np.where(targets >= 0, learned, 0)
        if self.cqi_mode == "olla":
            return self.adapter.select(targets)
        return genie_cqi(self.sim, targets)

    def step(self, action: np.ndarray):
        targets, learned = self.targets_from_action(action)
        cqis = self.cqis_for(targets, learned)
        out = self.sim.step(targets, cqis)
        if self.adapter is not None:
            self.adapter.update(targets, cqis, out["success"])
        reward = out["utility"] / self.n_aps
        info = {"system_throughput": out["throughput"], "targets": targets, "cqis": cqis, "out": out}
        return self._obs(), reward, self.sim.done, False, info
