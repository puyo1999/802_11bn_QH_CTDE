"""Cellular coordinated-scheduling simulator of paper3: channel, CQI, wraparound, oracle and references."""
import itertools
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cellular import baselines, cqi, layout
from src.cellular.channel import (GaussMarkovFading, breakpoint_distance, doppler_rho, umi_los_pathloss_db,
                                  umi_los_probability, umi_nlos_pathloss_db)
from src.cellular.oracle import oracle_decisions
from src.cellular.sim import CellularConfig, CellularSim


def make(drop_seed=0, **kw):
    sim = CellularSim(replace(CellularConfig(), **kw), drop_seed=drop_seed)
    sim.reset(seed=0)
    return sim


def test_umi_pathloss_matches_hand_values():
    fc = 3.5
    assert breakpoint_distance(fc, 10, 1.5) == pytest.approx(210.0)
    d3d = np.hypot(100.0, 8.5)
    assert umi_los_pathloss_db(100.0, d3d, fc) == pytest.approx(32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc))
    nlos = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc)
    assert umi_nlos_pathloss_db(100.0, d3d, fc) == pytest.approx(nlos)
    far = np.hypot(300.0, 8.5)
    assert umi_los_pathloss_db(300.0, far, fc) == pytest.approx(
        32.4 + 40 * np.log10(far) + 20 * np.log10(fc) - 9.5 * np.log10(210.0 ** 2 + 8.5 ** 2))


def test_umi_los_probability():
    assert umi_los_probability(18.0) == 1.0
    assert umi_los_probability(50.0) == pytest.approx(18 / 50 + np.exp(-50 / 36) * (1 - 18 / 50))


def test_cqi_thresholds_and_selection():
    assert cqi.THRESHOLDS_DB[0] == pytest.approx(-7.2, abs=0.05)
    assert cqi.THRESHOLDS_DB[-1] == pytest.approx(27.9, abs=0.05)
    assert np.all(np.diff(cqi.THRESHOLDS) > 0)
    assert cqi.best_cqi(cqi.THRESHOLDS[4]) == 5
    assert cqi.best_cqi(cqi.THRESHOLDS[4] * 0.999) == 4
    assert cqi.best_cqi(1e-3) == 0 and cqi.efficiency(0) == 0.0


def test_olla_converges_to_target_error_rate():
    olla = cqi.OLLA(1, step_db=0.5, target=0.1)
    rng = np.random.default_rng(0)
    failures, n = 0, 20000
    for _ in range(n):
        true_db = 10 + rng.normal(0, 2)
        q = olla.select(0, 10.0)
        ok = q > 0 and 10 ** (true_db / 10) >= cqi.THRESHOLDS[q - 1]
        failures += not ok
        olla.update(0, ok)
    assert failures / n == pytest.approx(0.1, abs=0.02)


def test_gauss_markov_fading_has_unit_power_and_correlation():
    rng = np.random.default_rng(0)
    rho = doppler_rho(30, 3.5, 1e-3)
    fad = GaussMarkovFading((20000,), rho, rng)
    prev = fad.h.copy()
    fad.step()
    assert fad.power().mean() == pytest.approx(1.0, abs=0.03)
    assert np.real(np.vdot(prev, fad.h)) / len(prev) == pytest.approx(rho, abs=0.03)


def test_wraparound_distances_and_drops():
    isd = 200.0
    centers = layout.cell_centers(isd)
    d = layout.wrapped_distance_2d(centers, centers, isd)
    assert np.allclose(np.diag(d), 0)
    off = d[~np.eye(7, dtype=bool)]
    assert np.allclose(off, isd)  # under wraparound every cell is a neighbour of every other cell
    ues = layout.drop_ues(np.random.default_rng(0), 50, isd)
    for c in range(7):
        rel = ues[c] - centers[c]
        assert np.all(layout.in_hexagon(rel, isd)) and np.all(np.linalg.norm(rel, axis=1) >= 10)


def test_ues_attach_to_strongest_cell():
    sim = make(ues_per_cell=5)
    assert np.array_equal(np.argmax(sim.gain_db, axis=0), sim.serving)
    assert np.bincount(sim.serving).tolist() == [5] * 7


def brute_force_utility(sim):
    """Reference: evaluate every joint decision of all subbands with sim.sinr (no vectorization)."""
    n = sim.n_ue_cell
    options = [[-1] + [c * n + u for u in range(n)] for c in sim.cluster]
    rx, w = sim.rx_power(), sim.weights()
    total = 0.0
    for b in range(sim.n_sub):
        best = 0.0
        for combo in itertools.product(*options):
            t = np.full((len(sim.cluster), sim.n_sub), -1)
            t[:, b] = combo
            gamma = sim.sinr(t, rx)[:, b]
            value = sum(w[u] * cqi.efficiency(cqi.best_cqi(g)) / sim.n_sub for u, g in zip(combo, gamma) if u >= 0)
            best = max(best, value)
        total += best
    return total


@pytest.mark.parametrize("objective", ["throughput", "pf"])
def test_oracle_matches_brute_force_and_step(objective):
    sim = make(ues_per_cell=2, n_subbands=2, objective=objective)
    for _ in range(3):
        targets, cqis, value = oracle_decisions(sim)
        assert value == pytest.approx(brute_force_utility(sim))
        out = sim.step(targets, cqis)
        assert out["utility"] == pytest.approx(value)
        assert out["success"][targets >= 0].all()


def test_oracle_dominates_references_per_slot():
    sim = make(ues_per_cell=3, n_subbands=2)
    for turn in range(5):
        _, _, best = oracle_decisions(sim)
        for policy in (baselines.full_reuse(sim), baselines.cluster_tdma(sim, turn), baselines.greedy_muting(sim)):
            targets, cqis = policy
            # Utility of the reference with the same channel: replay on a copy of the state.
            snapshot = (sim.fading.h.copy(), sim.avg_rate.copy(), sim.backlog.copy(), sim.t, list(sim.reports),
                        sim.rng.bit_generator.state, sim.arrivals.copy())
            value = sim.step(targets, cqis)["utility"]
            (sim.fading.h, sim.avg_rate, sim.backlog, sim.t, reports, state, sim.arrivals) = snapshot
            sim.reports.clear(); sim.reports.extend(reports); sim.rng.bit_generator.state = state
            assert value <= best + 1e-9
        sim.step(*oracle_decisions(sim)[:2])


def test_cluster_tdma_uses_one_cell_per_subband():
    sim = make(cluster_size=7, ues_per_cell=2, n_subbands=4)
    targets, _ = baselines.cluster_tdma(sim, turn=3)
    assert ((targets >= 0).sum(axis=0) == 1).all()


def test_finite_buffer_conserves_bits():
    sim = make(full_buffer=False, load=0.05, n_subbands=1, ues_per_cell=3)
    arrived, delivered = sim.backlog_now().sum(), 0.0
    for _ in range(50):
        targets, cqis, _ = oracle_decisions(sim)
        delivered += sim.step(targets, cqis)["throughput"]
        arrived += sim.arrivals.sum()
    assert np.all(sim.backlog >= -1e-12)
    assert arrived - sim.arrivals.sum() == pytest.approx(delivered + sim.backlog.sum())
    assert sim.backlog[~sim.in_cluster].sum() == 0  # background cells carry no traffic


def test_link_adaptation_modes():
    from src.cellular.linkadapt import LinkAdapter, expected_sinr, genie_cqi
    # Without fading and without delay the intent prediction equals the true SINR.
    sim = make(ues_per_cell=2, n_subbands=2, cqi_delay=0)
    sim.fading.rho = 1.0
    sim.fading.h = np.ones_like(sim.fading.h)
    targets, _ = baselines.greedy_muting(sim)
    sim.step(targets, genie_cqi(sim, targets))                 # fills the report of the previous slot
    targets, _ = baselines.greedy_muting(sim)
    tx = targets >= 0
    assert np.allclose(expected_sinr(sim, targets)[tx], sim.sinr(targets)[tx])
    # The OLLA offsets of the local mode drive the error rate toward the target.
    sim = make(ues_per_cell=2, n_subbands=4, cqi_delay=4, speed_kmh=30)
    adapter = LinkAdapter(sim, "local")
    sent = fails = 0
    while not sim.done:
        targets, _ = baselines.full_reuse(sim)
        cqis = adapter.select(targets)
        out = sim.step(targets, cqis)
        adapter.update(targets, cqis, out["success"])
        used = (targets >= 0) & (cqis > 0)
        if sim.t > 100:
            sent += used.sum()
            fails += (used & ~out["success"]).sum()
    assert fails / sent == pytest.approx(0.1, abs=0.06)


def test_warmup_erases_the_initial_pf_averages():
    """The per-UE mean rates of the measurement window do not depend on how the PF averages start."""
    def window_rates(scale):
        sim = make(ues_per_cell=2, n_subbands=2, speed_kmh=30)
        sim.avg_rate = sim.avg_rate * scale
        delivered = np.zeros(sim.n_ues)
        while not sim.done:
            measuring = sim.measuring
            out = sim.step(*oracle_decisions(sim)[:2])
            if measuring:
                delivered += out["delivered"]
        return delivered[sim.in_cluster]
    base, perturbed = window_rates(1.0), window_rates(5.0)
    assert np.allclose(perturbed, base, rtol=0.05)


@pytest.mark.parametrize("policy", ["oracle", "cluster_tdma"])
def test_low_load_steady_state_delivers_the_offered_traffic(policy):
    # eta = 0.25 is far below the saturated capacity of cluster TDMA (about 0.7 e_15 in the E1 pilot).
    sim = make(n_subbands=1, ues_per_cell=2, objective="throughput", full_buffer=False, load=0.25 / 6,
               episode_len=900)
    arrived = delivered = 0.0
    while not sim.done:
        measuring, arrivals = sim.measuring, sim.arrivals.sum()
        if policy == "oracle":
            out = sim.step(*oracle_decisions(sim)[:2])
        else:
            out = sim.step(*baselines.cluster_tdma(sim, sim.t))
        if measuring:
            arrived, delivered = arrived + arrivals, delivered + out["throughput"]
    assert delivered / arrived == pytest.approx(1.0, abs=0.05)


def test_causal_references_use_the_estimated_channel():
    from src.cellular.linkadapt import estimated_rx
    sim = make(ues_per_cell=3, n_subbands=2, cqi_delay=0)
    sim.fading.rho = 1.0
    sim.fading.h = np.ones_like(sim.fading.h)
    sim.step(*baselines.full_reuse(sim))                       # one report of an unfaded channel
    est = estimated_rx(sim)
    assert np.allclose(est, sim.rx_power())                   # no fading: the estimate is exact
    for genie, causal in ((baselines.full_reuse(sim), baselines.full_reuse(sim, est)),
                          (baselines.greedy_muting(sim), baselines.greedy_muting(sim, est)),
                          (baselines.cluster_tdma(sim, 1), baselines.cluster_tdma(sim, 1, est))):
        assert np.array_equal(genie[0], causal[0]) and np.array_equal(genie[1], causal[1])


def test_oracle_refuses_finite_buffer_with_several_subbands():
    with pytest.raises(ValueError):
        oracle_decisions(make(full_buffer=False, load=0.1, n_subbands=2))
