"""
src/cellular/baselines.py
Reference schedulers of paper3 (Section V-B). By default, like the references of paper1, they select targets and
rates from the channel of the current slot (genie information), so that their gap to the oracle reflects
coordination only. Passing rx = linkadapt.estimated_rx(sim) runs the same rules on the channel a base station
can estimate from delayed reports (causal references for the comparison with learners).

    full_reuse     every cluster cell transmits on every subband to its best weighted-rate UE, CQI decodable
                   when all cells transmit
    cluster_tdma   one cluster cell per subband and slot, round robin over cells with backlog, CQI decodable
                   under the background interference alone
    greedy_muting  start from full reuse; repeatedly silence the cell whose silence raises the slot utility most
"""
from __future__ import annotations

import numpy as np

from src.cellular import cqi
from src.cellular.oracle import subband_values


def _cell_ues(sim, c: int, backlog: np.ndarray) -> np.ndarray:
    ues = np.arange(c * sim.n_ue_cell, (c + 1) * sim.n_ue_cell)
    return ues[backlog[ues] > 0]


def _pick(sim, ues: np.ndarray, sinr: np.ndarray, weights: np.ndarray, backlog: np.ndarray) -> tuple[int, int]:
    """Best weighted-rate UE among ues given their SINR; (-1, 0) if no UE has a decodable rate."""
    if len(ues) == 0:
        return -1, 0
    q = cqi.best_cqi(sinr)
    value = weights[ues] * np.minimum(cqi.efficiency(q) / sim.n_sub, backlog[ues])
    if value.max() <= 0:
        return -1, 0
    k = int(np.argmax(value))
    return int(ues[k]), int(q[k])


def full_reuse(sim, rx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rx = sim.rx_power() if rx is None else rx
    weights, backlog = sim.weights(), sim.backlog_now()
    sinr_all = sim.all_transmit_sinr(rx)
    targets = np.full((len(sim.cluster), sim.n_sub), -1, dtype=np.int64)
    cqis = np.zeros_like(targets)
    for b in range(sim.n_sub):
        for i, c in enumerate(sim.cluster):
            ues = _cell_ues(sim, c, backlog)
            targets[i, b], cqis[i, b] = _pick(sim, ues, sinr_all[ues, b], weights, backlog)
    return targets, cqis


def cluster_tdma(sim, turn: int, rx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rx = sim.rx_power() if rx is None else rx
    weights, backlog = sim.weights(), sim.backlog_now()
    bg = sim.background_interference(rx)
    targets = np.full((len(sim.cluster), sim.n_sub), -1, dtype=np.int64)
    cqis = np.zeros_like(targets)
    eligible = [i for i, c in enumerate(sim.cluster) if len(_cell_ues(sim, c, backlog))]
    for b in range(sim.n_sub):
        if not eligible:
            break
        i = eligible[(turn + b) % len(eligible)]
        ues = _cell_ues(sim, sim.cluster[i], backlog)
        targets[i, b], cqis[i, b] = _pick(sim, ues, rx[sim.cluster[i], ues, b] / bg[ues, b], weights, backlog)
    return targets, cqis


def greedy_muting(sim, rx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rx = sim.rx_power() if rx is None else rx
    weights, backlog = sim.weights(), sim.backlog_now()
    targets, _ = full_reuse(sim, rx)
    cqis = np.zeros_like(targets)
    for b in range(sim.n_sub):
        current = targets[:, b].copy()
        value, best = subband_values(sim, rx, b, current[None, :], weights, backlog)
        while True:
            active = np.flatnonzero(current >= 0)
            if len(active) == 0:
                break
            trials = np.repeat(current[None, :], len(active), axis=0)
            trials[np.arange(len(active)), active] = -1
            trial_values, trial_best = subband_values(sim, rx, b, trials, weights, backlog)
            k = int(np.argmax(trial_values))
            if trial_values[k] <= value[0]:
                break
            current, value, best = trials[k], trial_values[k:k + 1], trial_best[k:k + 1]
        targets[:, b], cqis[:, b] = current, best[0]
    return targets, cqis
