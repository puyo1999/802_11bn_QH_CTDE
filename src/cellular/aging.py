"""
src/cellular/aging.py
Representative implementations of established link-adaptation remedies for aged CQI reports, used to measure how
much of the aging cost they recover (docs/paper3_aging_aware_la.md, option A of Section 8.3).

Model. The fading power of a link, observed a slots before use, has the power correlation c = rho^(2a) with the
current one (complex AR(1) fading of src/cellular/channel.py). The report gives the ratio kappa of the reported
all-transmit SINR to its value without fading, an estimate of the past fading power. Given kappa = g, the current
fading power is distributed as |sqrt(c g) + sqrt(1 - c) n|^2, n ~ CN(0, 1). The current SINR is modeled as the
all-transmit SINR without fading times this power, times a calibration factor 10^(Delta/10).

Remedies (all choose the CQI for given targets and use only reports and ACK/NACK):
  ExpectedThroughputCQI(mode="known")      CQI maximizing e_q P(gamma >= theta_q) with the true c (analysis bound)
  ExpectedThroughputCQI(mode="estimated")  the same with c estimated from the report history (model-based
                                           selection for outdated CSI, cf. Kang et al. 2014, Pellaco et al. 2020)
  ExpectedThroughputCQI(mode="calibrated") estimated c plus a calibration loop that matches the predicted success
                                           probability to the observed ACK rate (probability model refined by
                                           transmission outcomes, cf. Saxena et al., LTSLA)
Target-error-rate adaptation (cf. Park et al. 2015, SALAD) is represented by OLLA with a tuned target in
linkadapt.LinkAdapter.

Aging estimation (paper3 extension A and B):
  estimator="lag1"      power correlation from consecutive reports (AgingEstimator)
  estimator="multilag"  least-squares fit of rho^(2k) to the report autocorrelation at lags 1, 2, 4, 8
                        (MultiLagEstimator), more robust to measurement noise
  source="sinr"         past fading from the all-transmit SINR report (kappa), which mixes the fading of the serving
                        link with that of the interferers; current SINR modeled as gamma_ls times the fading factor
  fresh=True            the all-transmit SINR of the current slot instead of the aged report, with c = 1; with
                        the same rule on aged reports it splits a remedy's gain into a reduction of the operating
                        cost of link adaptation and a reduction of the cost of aging
  source="signal"       past fading of the serving link alone, from a received-power report of the serving cell
                        (sim.reported_signal_fading); current SINR modeled with the serving-link fading from the
                        conditional law and independent Rayleigh fading of every interferer
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.cellular import cqi
from src.cellular.linkadapt import fading_correction, large_scale_all_transmit_sinr


class AgingEstimator:
    """Per-UE estimate of the one-slot power correlation rho^2 from consecutive reports.

    Exponential moving averages of kappa, kappa^2 and kappa(t) kappa(t-1) over all subbands give the lag-one
    correlation of the reported fading power; c = (lag-one correlation)^a for a report of age a.
    """

    def __init__(self, n_ues: int, beta: float = 0.01):
        self.beta = beta
        self.m1 = np.ones(n_ues)
        self.m2 = np.full(n_ues, 2.0)          # E[X^2] = 2 for unit-mean exponential power
        self.m11 = np.ones(n_ues)              # uncorrelated start: E[X(t) X(t-1)] = 1
        self.prev = None

    def update(self, kappa: np.ndarray) -> None:
        """kappa: (n_ues, n_sub) fading-power estimates of the report that became available this slot."""
        if self.prev is not None:
            b = self.beta
            self.m1 = (1 - b) * self.m1 + b * kappa.mean(axis=1)
            self.m2 = (1 - b) * self.m2 + b * (kappa ** 2).mean(axis=1)
            self.m11 = (1 - b) * self.m11 + b * (kappa * self.prev).mean(axis=1)
        self.prev = kappa.copy()

    def lag_one(self) -> np.ndarray:
        var = np.maximum(self.m2 - self.m1 ** 2, 1e-9)
        return np.clip((self.m11 - self.m1 ** 2) / var, 0.0, 0.9999)

    def correlation(self, age: int) -> np.ndarray:
        return self.lag_one() ** age


class MultiLagEstimator:
    """Per-UE estimate of the one-slot power correlation rho^2 from the report autocorrelation at several lags.

    For complex AR(1) fading the power autocorrelation at lag k is rho^(2k); a least-squares fit of
    log corr(k) = k log rho^2 over the lags with a positive correlation gives rho^2.
    """

    def __init__(self, n_ues: int, lags: tuple[int, ...] = (1, 2, 4, 8), beta: float = 0.01):
        self.lags, self.beta = np.array(lags), beta
        self.m1 = np.ones(n_ues)
        self.m2 = np.full(n_ues, 2.0)
        self.mk = np.ones((len(lags), n_ues))
        self.history = deque(maxlen=max(lags) + 1)

    def update(self, kappa: np.ndarray) -> None:
        self.history.append(kappa.copy())
        if len(self.history) <= max(self.lags):
            return
        b = self.beta
        self.m1 = (1 - b) * self.m1 + b * kappa.mean(axis=1)
        self.m2 = (1 - b) * self.m2 + b * (kappa ** 2).mean(axis=1)
        for i, k in enumerate(self.lags):
            self.mk[i] = (1 - b) * self.mk[i] + b * (kappa * self.history[-1 - k]).mean(axis=1)

    def lag_one(self) -> np.ndarray:
        var = np.maximum(self.m2 - self.m1 ** 2, 1e-9)
        corr = (self.mk - self.m1[None, :] ** 2) / var[None, :]                  # (n_lags, n_ues)
        valid = corr > 0.05
        logc = np.log(np.where(valid, corr, 1.0))
        k = self.lags[:, None] * valid
        slope = (k * logc).sum(axis=0) / np.maximum((k ** 2).sum(axis=0), 1e-12)
        return np.where(valid.any(axis=0), np.clip(np.exp(slope), 0.0, 0.9999), 0.0)

    def correlation(self, age: int) -> np.ndarray:
        return self.lag_one() ** age


def probabilities_from_samples(gamma: np.ndarray) -> np.ndarray:
    """(k, 15) share of SINR samples gamma (k, m) that reach each CQI threshold."""
    return (gamma[:, :, None] >= cqi.THRESHOLDS[None, None, :]).mean(axis=1)


def success_probabilities(gamma_ls: np.ndarray, g: np.ndarray, c: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(k, 15) probability that each CQI decodes, from samples of the current fading power.

    gamma_ls, g, c: (k,) all-transmit SINR without fading, past fading power, power correlation.
    normals: (m,) complex standard normal samples shared by all links.
    """
    power = np.abs(np.sqrt(c * g)[:, None] + np.sqrt(1 - c)[:, None] * normals[None, :]) ** 2   # (k, m)
    return probabilities_from_samples(gamma_ls[:, None] * power)


class ExpectedThroughputCQI:
    """Expected-throughput CQI selection for aged reports (interface of linkadapt.LinkAdapter)."""

    MODES = ("known", "estimated", "calibrated")

    def __init__(self, sim, mode: str, n_samples: int = 64, step_db: float = 0.5, seed: int = 0,
                 estimator: str = "lag1", source: str = "sinr", fresh: bool = False):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode}")
        if estimator not in ("lag1", "multilag") or source not in ("sinr", "signal"):
            raise ValueError(f"unknown estimator {estimator} or source {source}")
        self.sim, self.mode, self.step, self.source = sim, mode, step_db, source
        self.rng = np.random.default_rng((seed, 11))
        self.n_samples = n_samples
        self.estimator = AgingEstimator(sim.n_ues) if estimator == "lag1" else MultiLagEstimator(sim.n_ues)
        self.offset = np.zeros(sim.n_ues)                 # calibration Delta_u in dB
        self.gamma_ls = large_scale_all_transmit_sinr(sim)
        self.fresh = fresh
        self.age = 0 if fresh else sim.cfg.cqi_delay + 1
        if fresh and (mode != "known" or source != "sinr"):
            raise ValueError("fresh selection uses the known correlation c = 1 and the SINR report")
        self._last = None

    def correlation(self) -> np.ndarray:
        if self.mode == "known":
            return np.full(self.sim.n_ues, self.sim.rho ** (2 * self.age))
        return self.estimator.correlation(self.age)

    def select(self, targets: np.ndarray) -> np.ndarray:
        sim = self.sim
        if self.fresh:
            kappa = sim.all_transmit_sinr() / self.gamma_ls[:, None]
        elif self.source == "sinr":
            kappa = fading_correction(sim)
        else:
            kappa = sim.reported_signal_fading()
        self.estimator.update(kappa)
        tx = targets >= 0
        cqis = np.zeros(targets.shape, dtype=np.int64)
        self._last = np.zeros(targets.shape)
        if not tx.any():
            return cqis
        ue = targets[tx]
        sub = np.nonzero(tx)[1]
        normals = (self.rng.standard_normal(self.n_samples) + 1j * self.rng.standard_normal(self.n_samples)) / np.sqrt(2)
        scale = 10 ** (self.offset[ue] / 10)
        c = self.correlation()[ue]
        if self.source == "sinr":
            probs = success_probabilities(self.gamma_ls[ue] * scale, kappa[ue, sub], c, normals)
        else:
            probs = self._signal_model(ue, kappa[ue, sub], c, normals, scale)
        value = cqi.EFFICIENCY[None, :] * probs
        # If no CQI has a positive expected rate, the lowest one is sent as a probe, so that the calibration loop
        # keeps receiving feedback for every scheduled UE.
        chosen = np.argmax(value, axis=1) + 1
        cqis[tx] = chosen
        self._last[tx] = probs[np.arange(len(ue)), chosen - 1]
        return cqis

    def _signal_model(self, ue, g, c, normals, scale) -> np.ndarray:
        """Success probabilities with the serving-link fading from the conditional law and independent Rayleigh
        fading of every other cell (all-transmit assumption of the report), from shared samples."""
        sim = self.sim
        power = np.abs(np.sqrt(c * g)[:, None] + np.sqrt(1 - c)[:, None] * normals[None, :]) ** 2    # (k, m)
        signal = sim.rx_mean[sim.serving[ue], ue][:, None] * power
        others = sim.rx_mean[:, ue].T.copy()                                                          # (k, cells)
        others[np.arange(len(ue)), sim.serving[ue]] = 0.0
        fades = self.rng.exponential(1.0, (self.n_samples, sim.n_cells))                             # (m, cells)
        interference = others @ fades.T + sim.noise_mw                                                # (k, m)
        return probabilities_from_samples(signal / interference * scale[:, None])

    def update(self, targets: np.ndarray, cqis: np.ndarray, success: np.ndarray) -> None:
        if self.mode != "calibrated" or self._last is None:
            return
        for u, q, ok, p in zip(targets.ravel(), cqis.ravel(), success.ravel(), self._last.ravel()):
            if u >= 0 and q > 0:
                self.offset[u] += self.step * (float(ok) - p)
