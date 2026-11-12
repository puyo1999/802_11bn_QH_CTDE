"""
src/cellular/oracle.py
Exact per-subband oracle (paper3, Section V-A).

For a fixed target vector of the cluster on subband b, the transmitting set and every SINR are fixed, so the
best CQI of each link is the highest index whose threshold its SINR meets. Under full buffer, with the weights
fixed within the slot, the slot utility is a sum of per-subband terms, and the oracle enumerates the
(U + 1)^C target vectors of each subband separately. Under finite buffer the subbands share the backlog of a UE,
and the enumeration is exact only for one subband.
"""
from __future__ import annotations

import itertools

import numpy as np

from src.cellular import cqi


def subband_values(sim, rx: np.ndarray, b: int, combos: np.ndarray,
                   weights: np.ndarray, backlog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Utility and best CQI of every target vector in combos (K, C) on subband b."""
    bg = sim.background_interference(rx)[:, b]
    tx = combos >= 0
    ue = np.where(tx, combos, 0)
    rx_cluster = rx[sim.cluster][:, :, b]                                      # (C, n_ues)
    values = np.zeros(len(combos))
    best = np.zeros(combos.shape, dtype=np.int64)
    for i in range(len(sim.cluster)):
        u = ue[:, i]
        interference = bg[u].copy()
        for j in range(len(sim.cluster)):
            if j != i:
                interference += np.where(tx[:, j], rx_cluster[j, u], 0.0)
        q = cqi.best_cqi(rx_cluster[i, u] / interference)
        rate = np.minimum(cqi.efficiency(q) / sim.n_sub, backlog[u])
        best[:, i] = np.where(tx[:, i], q, 0)
        values += np.where(tx[:, i], weights[u] * rate, 0.0)
    return values, best


def candidate_options(sim, backlog: np.ndarray) -> list[list[int]]:
    n = sim.n_ue_cell
    return [[-1] + [c * n + u for u in range(n) if backlog[c * n + u] > 0] for c in sim.cluster]


def oracle_decisions(sim) -> tuple[np.ndarray, np.ndarray, float]:
    """Targets (C, n_sub), CQI indices (C, n_sub) and the slot utility of the exact oracle."""
    if not sim.cfg.full_buffer and sim.n_sub > 1:
        raise ValueError("the per-subband oracle is exact under finite buffer only for one subband")
    rx, weights, backlog = sim.rx_power(), sim.weights(), sim.backlog_now()
    combos = np.array(list(itertools.product(*candidate_options(sim, backlog))), dtype=np.int64)
    targets = np.full((len(sim.cluster), sim.n_sub), -1, dtype=np.int64)
    cqis = np.zeros_like(targets)
    total = 0.0
    for b in range(sim.n_sub):
        values, best = subband_values(sim, rx, b, combos, weights, backlog)
        k = int(np.argmax(values))
        targets[:, b], cqis[:, b] = combos[k], best[k]
        total += float(values[k])
    return targets, cqis, total
