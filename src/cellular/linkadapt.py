"""
src/cellular/linkadapt.py
CQI selection for given targets (docs/paper3_dynamic_quantization.md, Sections 3.3 and 7).

    genie   highest CQI decodable at the SINR of this slot (the oracle's rate rule)
    local   CQI from the delayed all-transmit report of the UE, corrected by an OLLA offset
    intent  expected SINR from the large-scale gains of the cells that announced a transmission on the
            subband, times a fading correction estimated from the delayed report, corrected by an OLLA offset

The report is the all-transmit SINR recorded at the end of slot t - d - 1 (sim.reported_sinr), i.e., its
age is the CQI delay d plus the slot in which it was measured.
"""
from __future__ import annotations

import numpy as np

from src.cellular import cqi

MODES = ("genie", "local", "intent")


def genie_cqi(sim, targets: np.ndarray) -> np.ndarray:
    return np.where(targets >= 0, cqi.best_cqi(sim.sinr(targets)), 0)


def large_scale_all_transmit_sinr(sim) -> np.ndarray:
    """(n_ues,) all-transmit SINR of every UE from its serving cell without fading."""
    rx = sim.rx_mean
    signal = rx[sim.serving, np.arange(sim.n_ues)]
    return signal / (rx.sum(axis=0) - signal + sim.noise_mw)


def expected_sinr(sim, targets: np.ndarray) -> np.ndarray:
    """(C, n_sub) SINR predicted from announced transmissions, large-scale gains and the delayed report."""
    rx, report = sim.rx_mean, sim.reported_sinr()
    kappa = report / large_scale_all_transmit_sinr(sim)[:, None]               # fading correction (n_ues, n_sub)
    bg = rx[sim.background].sum(axis=0) if sim.background else np.zeros(sim.n_ues)
    out = np.zeros(targets.shape)
    for b in range(sim.n_sub):
        for i, c in enumerate(sim.cluster):
            u = targets[i, b]
            if u < 0:
                continue
            inter = bg[u] + sim.noise_mw + sum(rx[sim.cluster[j], u] for j in range(len(sim.cluster))
                                               if j != i and targets[j, b] >= 0)
            out[i, b] = rx[c, u] / inter * kappa[u, b]
    return out


class LinkAdapter:
    """CQI selection for one episode; keeps the per-UE OLLA offsets of the local and intent modes."""

    def __init__(self, sim, mode: str, step_db: float = 0.5, target_bler: float = 0.1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode}")
        self.sim, self.mode = sim, mode
        self.olla = cqi.OLLA(sim.n_ues, step_db, target_bler)

    def select(self, targets: np.ndarray) -> np.ndarray:
        if self.mode == "genie":
            return genie_cqi(self.sim, targets)
        if self.mode == "local":
            gamma = self.sim.reported_sinr()[np.maximum(targets, 0), np.arange(self.sim.n_sub)[None, :]]
        else:
            gamma = expected_sinr(self.sim, targets)
        gamma_db = 10 * np.log10(np.maximum(gamma, 1e-30))
        offset = self.olla.offset[np.maximum(targets, 0)]
        return np.where(targets >= 0, cqi.best_cqi(10 ** ((gamma_db + offset) / 10)), 0)

    def update(self, targets: np.ndarray, cqis: np.ndarray, success: np.ndarray) -> None:
        if self.mode == "genie":
            return
        for u, q, ok in zip(targets.ravel(), cqis.ravel(), success.ravel()):
            if u >= 0 and q > 0:
                self.olla.update(int(u), bool(ok))
