"""
src/cellular/cqi.py
Rate set of the coordinated scheduler.

Spectral efficiencies: four-bit CQI table, ETSI TS 138 214 V16.2.0, Table 5.2.2.1-2 (CQI 1-15).
SINR thresholds: inversion of the attenuated Shannon bound e = alpha log2(1 + SINR) with the downlink
attenuation alpha = 0.6 of ETSI TR 136 942 V15.0.0, Annex A, Table A.1, i.e., theta_q = 2^(e_q / alpha) - 1.
"""
from __future__ import annotations

import numpy as np

EFFICIENCY = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
                       2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])
ALPHA = 0.6
THRESHOLDS = 2.0 ** (EFFICIENCY / ALPHA) - 1.0          # linear, index q - 1
THRESHOLDS_DB = 10 * np.log10(THRESHOLDS)
N_CQI = len(EFFICIENCY)


def best_cqi(sinr):
    """Highest CQI index (1-15) whose threshold the SINR meets, 0 if none (out of range)."""
    return np.searchsorted(THRESHOLDS, np.asarray(sinr), side="right")


def efficiency(cqi):
    """Spectral efficiency of CQI index (0 -> 0)."""
    cqi = np.asarray(cqi)
    return np.where(cqi > 0, EFFICIENCY[np.clip(cqi, 1, N_CQI) - 1], 0.0)


class OLLA:
    """Outer-loop link adaptation: per-UE offset in dB on the reported SINR, driven by ACK/NACK.

    offset <- offset - step after a failure, offset + step * target / (1 - target) after a success,
    so that the long-run error rate converges to `target`.
    """

    def __init__(self, n_ues: int, step_db: float = 0.5, target: float = 0.1):
        self.offset = np.zeros(n_ues)
        self.step, self.up = step_db, step_db * target / (1 - target)

    def select(self, ue: int, reported_sinr_db: float) -> int:
        return int(best_cqi(10 ** ((reported_sinr_db + self.offset[ue]) / 10)))

    def update(self, ue: int, success: bool) -> None:
        self.offset[ue] += self.up if success else -self.step
