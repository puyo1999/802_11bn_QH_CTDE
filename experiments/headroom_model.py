"""
experiments/headroom_model.py
Parameter-free prediction of the coordination headroom H = 1 - Co-TDMA / oracle from the link budget.

For every topology, the service vector of each joint target choice u (STA or silence per AP) is computed
with the same link tables and highest decodable MCS as the exact oracle, assuming saturated queues. Two
fluid models then predict the delivered packets per slot at arrival rate lambda per STA:

  simple : Theta = min(Lambda, C), with Lambda = lambda |S| and C the best single-slot delivery
           (C_T over single-AP choices, C_SR over all choices).
  LP     : Theta = max sum_k x_k  s.t.  x = sum_u p_u r(u), sum_u p_u = 1, p >= 0, x_k <= lambda,
           i.e., the best time sharing of the service vectors under per-STA arrival rates
           (TDMA: single-AP choices only; SR: all choices).

Both models average the normal and the degraded interference regime with the weights of their shares of the
episode. The prediction is H_hat = 1 - Theta_TDMA / Theta_SR, compared with the measured H of stage 1.

    python experiments/headroom_model.py
"""
from __future__ import annotations

import itertools
import os
import sys
from argparse import Namespace

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_random_topology import load_rows, per_topology  # noqa: E402
from oracle_gap import make_oracle_env  # noqa: E402
from src.envs.topology_registry import get_registry  # noqa: E402
from src.training.oracle_policy import link_tables  # noqa: E402

TOPOLOGY_FILE = "configs/topologies_random.yaml"


def service_vectors(env, harsh: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows: joint choices u; columns: STAs (flattened). Entry: packets delivered per slot when saturated."""
    env.reset(seed=0)
    signal, interference, blocks = link_tables(env, harsh)
    thresholds = np.asarray(env.mcs_table.sinr_thresholds)
    capacities = np.asarray(env.mcs_table.capacities, dtype=float)
    n_real = [len(stas) for stas in env.topo.sta_positions]
    offset = np.concatenate([[0], np.cumsum(n_real)])
    combos = list(itertools.product(*[[-1] + list(range(n)) for n in n_real]))
    rates = np.zeros((len(combos), offset[-1]))
    single = np.zeros(len(combos), dtype=bool)
    for row, u in enumerate(combos):
        tx = [i for i, s in enumerate(u) if s >= 0]
        single[row] = len(tx) == 1
        for i in tx:
            k = u[i]
            others = [j for j in tx if j != i]
            if any(blocks[j, i, k] for j in others):
                continue
            noise = interference[others, i, k].sum() + env.interference.noise_mw
            sinr_db = 10.0 * np.log10(signal[i, k] / noise)
            m = int((sinr_db >= thresholds).sum()) - 1
            if m >= 0:
                rates[row, offset[i] + k] = capacities[m]
    return rates, single


def lp_throughput(rates: np.ndarray, lam: float) -> float:
    """max sum_k min-constrained delivery over time sharing of the rows of `rates`."""
    n_u, n_k = rates.shape
    # Variables: p (n_u), x (n_k). maximize sum x.
    c = np.concatenate([np.zeros(n_u), -np.ones(n_k)])
    a_ub = np.hstack([-rates.T, np.eye(n_k)])                     # x_k <= sum_u p_u r_uk
    a_eq = np.concatenate([np.ones(n_u), np.zeros(n_k)])[None, :]
    bounds = [(0, None)] * n_u + [(0, lam)] * n_k
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(n_k), A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    return -res.fun


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main() -> None:
    measured = per_topology(load_rows("stage1"))
    loads = sorted({k[0] for k in measured})
    topos = sorted({k[1] for k in measured})
    registry = get_registry(TOPOLOGY_FILE)
    args = Namespace(config="configs/qh_ctde.yaml", episode_len=200, fading="none", fading_rho=0.0)

    records = []
    for t in topos:
        env = make_oracle_env(args, registry.get(t), loads[0])
        harsh_share = (1.0 - env.harsh_start_step / env.max_episodes) if env.harsh_mode_enabled else 0.0
        regimes = [(1.0 - harsh_share, *service_vectors(env, False))]
        if harsh_share > 0:
            regimes.append((harsh_share, *service_vectors(env, True)))
        n_sta = regimes[0][1].shape[1]
        c_t = sum(w * r[s].sum(1).max() for w, r, s in regimes)
        c_sr = sum(w * r.sum(1).max() for w, r, s in regimes)
        for lam in loads:
            if (lam, t) not in measured:
                continue
            offered = lam * n_sta
            simple = 1 - sum(w * min(offered, r[s].sum(1).max()) for w, r, s in regimes) /                 sum(w * min(offered, r.sum(1).max()) for w, r, s in regimes)
            lp = 1 - sum(w * lp_throughput(r[s], lam) for w, r, s in regimes) /                 sum(w * lp_throughput(r, lam) for w, r, s in regimes)
            records.append((lam, t, offered / c_t, c_sr / c_t, 1 - measured[(lam, t)]["co_tdma"], simple, lp))

    print(f"{'lambda':>6} {'n':>4} {'R2 simple':>10} {'R2 LP':>7} {'MAE simple':>11} {'MAE LP':>7}")
    groups = [(lam, [r for r in records if r[0] == lam]) for lam in loads] + [("all", records)]
    for lam, rs in groups:
        h, s, l = (np.array([r[i] for r in rs]) for i in (4, 5, 6))
        print(f"{lam!s:>6} {len(rs):>4} {r2(h, s):>10.2f} {r2(h, l):>7.2f} "
              f"{100 * np.abs(h - s).mean():>10.1f}p {100 * np.abs(h - l).mean():>6.1f}p")
    high = [r for r in records if r[0] > 1]
    h, l = np.array([r[4] for r in high]), np.array([r[6] for r in high])
    print(f"lambda > 1: corr {np.corrcoef(h, l)[0, 1]:.2f}, LP - H mean {100 * (l - h).mean():+.1f}p, "
          f"LP > H in {100 * np.mean(l > h):.0f}%")
    g = np.array([r[3] for r in records])
    print(f"\nG = C_SR / C_T: median {np.median(g):.2f}, range {g.min():.2f}-{g.max():.2f}")
    worst = sorted(records, key=lambda r: -abs(r[4] - r[6]))[:5]
    print("largest LP errors (lambda, topo, rho, G, H, simple, LP):")
    for r in worst:
        print("  ", r[0], r[1], *(f"{v:.2f}" for v in r[2:]))


if __name__ == "__main__":
    main()
