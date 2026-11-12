"""
src/cellular/sim.py
Slot simulator of coordinated scheduling in a cluster of 5G NR cells (paper3, Section III).

Units: deliveries and backlogs are in units of W * T_slot bits, so that a link with CQI q on one of B subbands
delivers EFFICIENCY[q-1] / B per slot and cluster TDMA delivers at most EFFICIENCY[-1] per slot.

Slot order (as in env v2 of paper1): arrivals join the backlog, the decisions are applied to the channel of the
slot, deliveries leave the backlog, PF averages are updated, the CQI report of the slot is recorded, and the
fading advances. Cells outside the cluster transmit on every subband in every slot (background interference).

UE indices are global: UE u of cell c has index c * U + u.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.cellular import cqi, layout
from src.cellular.channel import GaussMarkovFading, doppler_rho, large_scale_gain_db


@dataclass
class CellularConfig:
    isd: float = 200.0
    fc_ghz: float = 3.5
    bandwidth_hz: float = 20e6
    n_subbands: int = 4
    slot_s: float = 1e-3
    ptx_dbm: float = 44.0
    noise_figure_db: float = 9.0
    h_bs: float = 10.0
    h_ut: float = 1.5
    ues_per_cell: int = 4
    cluster_size: int = 3
    speed_kmh: float = 3.0
    cqi_delay: int = 4
    objective: str = "pf"            # pf | throughput
    pf_beta: float = 0.01
    full_buffer: bool = True
    load: float = 0.0                # lambda: mean offered load per UE in units of e_15 * W * T_slot per slot
    packets_per_unit: int = 20       # packet size e_15 / packets_per_unit (finite buffer)
    episode_len: int = 200

    def __post_init__(self):
        if self.objective not in ("pf", "throughput"):
            raise ValueError(f"objective must be pf or throughput, got {self.objective}")


class CellularSim:
    def __init__(self, cfg: CellularConfig, drop_seed: int):
        self.cfg = cfg
        n_cells, n_ue = 7, cfg.ues_per_cell
        self.n_cells, self.n_ue_cell, self.n_ues = n_cells, n_ue, n_cells * n_ue
        self.n_sub = cfg.n_subbands
        self.cluster = layout.cluster_cells(cfg.cluster_size)
        self.background = [c for c in range(n_cells) if c not in self.cluster]
        self.serving = np.repeat(np.arange(n_cells), n_ue)                       # (n_ues,)
        self.in_cluster = np.isin(self.serving, self.cluster)

        self.centers = layout.cell_centers(cfg.isd)
        self._drop(np.random.default_rng((drop_seed, 1)))

        self.p_sub_mw = 10 ** (cfg.ptx_dbm / 10) / self.n_sub
        self.noise_mw = 10 ** ((-174 + 10 * np.log10(cfg.bandwidth_hz / self.n_sub) + cfg.noise_figure_db) / 10)
        self.rx_mean = self.p_sub_mw * 10 ** (self.gain_db / 10)                # (n_cells, n_ues) mW
        self.rho = doppler_rho(cfg.speed_kmh, cfg.fc_ghz, cfg.slot_s)
        self.packet = cqi.EFFICIENCY[-1] / cfg.packets_per_unit

    def _drop(self, rng: np.random.Generator) -> None:
        """Drop UEs uniformly over the layout and attach each to the cell of largest large-scale gain
        (TR 38.901 Table 7.8-1, attachment based on path loss), until every cell serves U UEs."""
        cfg, n_ue = self.cfg, self.n_ue_cell
        kept = [[] for _ in range(self.n_cells)]                                 # (position, gain_db, los)
        while min(len(k) for k in kept) < n_ue:
            pos = layout.drop_ues(rng, n_ue, cfg.isd).reshape(-1, 2)
            d2d = layout.wrapped_distance_2d(self.centers, pos, cfg.isd)
            gain, los = large_scale_gain_db(rng, d2d, cfg.fc_ghz, cfg.h_bs, cfg.h_ut)
            for k, cell in enumerate(np.argmax(gain, axis=0)):
                if len(kept[cell]) < n_ue:
                    kept[cell].append((pos[k], gain[:, k], los[:, k]))
        flat = [entry for cell in kept for entry in cell]
        self.ue_pos = np.array([e[0] for e in flat])                             # (n_ues, 2)
        self.gain_db = np.array([e[1] for e in flat]).T                          # (n_cells, n_ues)
        self.los = np.array([e[2] for e in flat]).T
        self.d2d = layout.wrapped_distance_2d(self.centers, self.ue_pos, cfg.isd)

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng((seed, 2))
        self.fading = GaussMarkovFading((self.n_cells, self.n_ues, self.n_sub), self.rho, self.rng)
        self.backlog = np.zeros(self.n_ues)
        self.avg_rate = np.ones(self.n_ues)
        self.reports = deque(maxlen=self.cfg.cqi_delay + 1)
        self.t = 0
        self._draw_arrivals()

    def _draw_arrivals(self) -> None:
        if self.cfg.full_buffer:
            self.arrivals = np.zeros(self.n_ues)
        else:
            # Only the UEs of the cluster carry traffic; the background cells transmit regardless.
            lam = self.cfg.load * self.cfg.packets_per_unit
            self.arrivals = self.rng.poisson(lam, self.n_ues) * self.packet * self.in_cluster

    # ------------------------------------------------------------------ views
    def rx_power(self) -> np.ndarray:
        """(n_cells, n_ues, n_sub) received power of every base station at every UE in this slot (mW)."""
        return self.rx_mean[:, :, None] * self.fading.power()

    def backlog_now(self) -> np.ndarray:
        """Backlog after this slot's arrivals (genie view, as the oracle of paper1); inf under full buffer."""
        if self.cfg.full_buffer:
            return np.full(self.n_ues, np.inf)
        return self.backlog + self.arrivals

    def weights(self) -> np.ndarray:
        if self.cfg.objective == "throughput":
            return np.ones(self.n_ues)
        return 1.0 / self.avg_rate

    def background_interference(self, rx: np.ndarray) -> np.ndarray:
        """(n_ues, n_sub) interference from the cells outside the cluster plus noise."""
        bg = rx[self.background].sum(axis=0) if self.background else np.zeros(rx.shape[1:])
        return bg + self.noise_mw

    def all_transmit_sinr(self, rx: np.ndarray | None = None) -> np.ndarray:
        """(n_ues, n_sub) SINR of every UE from its serving cell when all cells transmit."""
        rx = self.rx_power() if rx is None else rx
        signal = rx[self.serving, np.arange(self.n_ues)]
        return signal / (rx.sum(axis=0) - signal + self.noise_mw)

    def sinr(self, targets: np.ndarray, rx: np.ndarray | None = None) -> np.ndarray:
        """(C, n_sub) SINR of each scheduled link; targets (C, n_sub) global UE index or -1 (silence)."""
        rx = self.rx_power() if rx is None else rx
        bg = self.background_interference(rx)
        out = np.zeros(targets.shape)
        for b in range(self.n_sub):
            for i, c in enumerate(self.cluster):
                u = targets[i, b]
                if u < 0:
                    continue
                inter = bg[u, b] + sum(rx[self.cluster[j], u, b] for j in range(len(self.cluster))
                                       if j != i and targets[j, b] >= 0)
                out[i, b] = rx[c, u, b] / inter
        return out

    # ------------------------------------------------------------------ step
    def step(self, targets: np.ndarray, cqi_idx: np.ndarray) -> dict:
        """Apply targets (C, n_sub) and CQI indices (C, n_sub); returns the slot statistics."""
        targets, cqi_idx = np.asarray(targets), np.asarray(cqi_idx)
        for i, c in enumerate(self.cluster):
            owned = targets[i][targets[i] >= 0]
            if np.any(self.serving[owned] != c):
                raise ValueError(f"cell {c} scheduled a UE it does not serve")
        rx = self.rx_power()
        weights = self.weights()
        self.backlog = self.backlog + self.arrivals
        gamma = self.sinr(targets, rx)
        success = (targets >= 0) & (cqi_idx > 0) & (gamma >= cqi.THRESHOLDS[np.clip(cqi_idx, 1, cqi.N_CQI) - 1])
        offered = np.where(success, cqi.efficiency(cqi_idx) / self.n_sub, 0.0)
        delivered = np.zeros(self.n_ues)
        for b in range(self.n_sub):
            for i in range(len(self.cluster)):
                u = targets[i, b]
                if u >= 0 and success[i, b]:
                    room = np.inf if self.cfg.full_buffer else self.backlog[u] - delivered[u]
                    delivered[u] += min(offered[i, b], room)
        if not self.cfg.full_buffer:
            self.backlog = self.backlog - delivered
        self.avg_rate = (1 - self.cfg.pf_beta) * self.avg_rate + self.cfg.pf_beta * delivered
        self.reports.append(self.all_transmit_sinr(rx))
        self.fading.step()
        self.t += 1
        self._draw_arrivals()
        return {"throughput": float(delivered.sum()), "utility": float(weights @ delivered),
                "delivered": delivered, "success": success, "sinr": gamma}

    def reported_sinr(self) -> np.ndarray:
        """(n_ues, n_sub) all-transmit SINR reported cqi_delay slots ago (oldest available before that)."""
        if not self.reports:
            return self.all_transmit_sinr()
        return self.reports[0]

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.episode_len
