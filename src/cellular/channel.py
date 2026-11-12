"""
src/cellular/channel.py
Large-scale and small-scale channel of the UMi-Street Canyon scenario.

Path loss and LOS probability follow ETSI TR 138 901 V17.0.0, Tables 7.4.1-1 and 7.4.2-1 (fc in GHz, distances
in meters). Shadowing is log-normal with 4 dB (LOS) / 7.82 dB (NLOS), drawn independently per link and drop.
Small-scale fading is a complex Gauss-Markov process per link and subband, h(t) = rho h(t-1) + sqrt(1-rho^2) n(t),
with rho = J0(2 pi f_D T_slot), so that |h|^2 has unit mean and is exponentially distributed.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0

SPEED_OF_LIGHT = 3.0e8
SHADOW_STD_DB = {True: 4.0, False: 7.82}


def breakpoint_distance(fc_ghz: float, h_bs: float, h_ut: float, h_e: float = 1.0) -> float:
    """d'_BP = 4 h'_BS h'_UT fc / c with effective heights h' = h - h_E (h_E = 1 m for UMi)."""
    return 4 * (h_bs - h_e) * (h_ut - h_e) * fc_ghz * 1e9 / SPEED_OF_LIGHT


def umi_los_pathloss_db(d2d, d3d, fc_ghz: float, h_bs: float = 10.0, h_ut: float = 1.5):
    d2d, d3d = np.asarray(d2d, dtype=float), np.asarray(d3d, dtype=float)
    d_bp = breakpoint_distance(fc_ghz, h_bs, h_ut)
    pl1 = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc_ghz)
    pl2 = (32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc_ghz)
           - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def umi_nlos_pathloss_db(d2d, d3d, fc_ghz: float, h_bs: float = 10.0, h_ut: float = 1.5):
    nlos = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return np.maximum(umi_los_pathloss_db(d2d, d3d, fc_ghz, h_bs, h_ut), nlos)


def umi_los_probability(d2d):
    d2d = np.asarray(d2d, dtype=float)
    d = np.maximum(d2d, 1e-9)
    return np.where(d2d <= 18.0, 1.0, 18.0 / d + np.exp(-d / 36.0) * (1.0 - 18.0 / d))


def large_scale_gain_db(rng: np.random.Generator, d2d: np.ndarray, fc_ghz: float,
                        h_bs: float = 10.0, h_ut: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Gain -(PL + X) in dB for every link of d2d, with LOS state and shadowing drawn once per drop.

    Returns (gain_db, los). d2d below 10 m is clipped to the 10 m lower end of the model's range.
    """
    d2d = np.maximum(d2d, 10.0)
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    los = rng.random(d2d.shape) < umi_los_probability(d2d)
    pl = np.where(los, umi_los_pathloss_db(d2d, d3d, fc_ghz, h_bs, h_ut),
                  umi_nlos_pathloss_db(d2d, d3d, fc_ghz, h_bs, h_ut))
    shadow = rng.standard_normal(d2d.shape) * np.where(los, SHADOW_STD_DB[True], SHADOW_STD_DB[False])
    # Positive shadowing means more received power (TR 38.901 sign convention), i.e., gain = -PL + X.
    return -pl + shadow, los


def doppler_rho(speed_kmh: float, fc_ghz: float, slot_s: float) -> float:
    f_d = speed_kmh / 3.6 * fc_ghz * 1e9 / SPEED_OF_LIGHT
    return float(j0(2 * np.pi * f_d * slot_s))


class GaussMarkovFading:
    """Complex Gauss-Markov fading with unit-mean power, one coefficient per entry of `shape`."""

    def __init__(self, shape: tuple[int, ...], rho: float, rng: np.random.Generator):
        self.rho, self.rng, self.shape = rho, rng, shape
        self.h = self._noise()

    def _noise(self) -> np.ndarray:
        return (self.rng.standard_normal(self.shape) + 1j * self.rng.standard_normal(self.shape)) / np.sqrt(2)

    def step(self) -> None:
        self.h = self.rho * self.h + np.sqrt(1 - self.rho ** 2) * self._noise()

    def power(self) -> np.ndarray:
        return np.abs(self.h) ** 2
