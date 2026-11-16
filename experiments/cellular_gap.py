"""
experiments/cellular_gap.py
paper3 experiments without learners (docs/paper3_5g_multicell_plan.md, docs/paper3_design_review.md).

Every episode has sim.cfg.warmup_slots warm-up slots followed by the measurement window; all metrics are taken over
the measurement window (steady state). Per episode: throughput per slot, delivery ratio (delivered / arrived,
finite buffer), the PF utility U = sum over cluster UEs of log(mean rate), the 5th percentile of the per-UE mean
rate, the mean slot utility, and the transmission error rate.

  e0  calibration: CDF of the wideband all-transmit SINR of the serving cell (large-scale gains only, all
      7 cells transmitting) over drops, and the run time of the exact oracle per slot for C in {3, 7}.
  e1  headroom sweep: finite buffer, one subband, cluster C = 3, throughput objective; the oracle and the
      references run on identical arrival and fading sequences. The headroom H = 1 - TDMA / oracle is reported
      against eta = lambda C U and against rho_TDMA = offered load / saturated capacity of cluster TDMA, measured
      per drop at the highest load.
  la0 link-adaptation study: full buffer, C = 3, U = 4, B = 4, PF objective; the targets of full reuse, greedy
      muting and the oracle are kept, and only the CQI selection changes (genie / local: delayed report + OLLA /
      fresh: report of the current slot + OLLA / intent: announced transmissions + OLLA / intent+commit:
      targets decided one slot ahead). Results relative to the oracle with genie CQI.

    python experiments/cellular_gap.py e0
    python experiments/cellular_gap.py e1 --drops 20 --seeds 1
    python experiments/cellular_gap.py la0 --drops 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cellular import baselines
from src.cellular.linkadapt import LinkAdapter
from src.cellular.oracle import oracle_decisions
from src.cellular.sim import CellularConfig, CellularSim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cellular_results")
POLICIES = ["oracle", "full_reuse", "cluster_tdma", "greedy_muting"]
ETAS = [0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
RATE_FLOOR = 1e-4   # mean-rate floor (units of W T_slot per slot) inside the log of the PF utility


def decide(sim, policy: str, turn: int):
    if policy == "oracle":
        return oracle_decisions(sim)[:2]
    if policy == "cluster_tdma":
        return baselines.cluster_tdma(sim, turn)
    return getattr(baselines, policy)(sim)


class Meter:
    """Statistics of the measurement window of one episode."""

    def __init__(self, sim):
        self.sim = sim
        self.delivered = np.zeros(sim.n_ues)
        self.arrived = self.utility = 0.0
        self.slots = self.sent = self.failed = 0

    def record(self, arrivals: np.ndarray, targets: np.ndarray, cqis: np.ndarray, out: dict) -> None:
        self.delivered += out["delivered"]
        self.arrived += float(arrivals.sum())
        self.utility += out["utility"]
        used = (targets >= 0) & (cqis > 0)
        self.sent += int(used.sum())
        self.failed += int((used & ~out["success"]).sum())
        self.slots += 1

    def result(self) -> dict:
        rates = self.delivered[self.sim.in_cluster] / self.slots
        return {"throughput": float(rates.sum()),
                "delivery": float(self.delivered.sum() / self.arrived) if self.arrived > 0 else float("nan"),
                "log_utility": float(np.log(np.maximum(rates, RATE_FLOOR)).sum()),
                "p5_rate": float(np.percentile(rates, 5)), "slot_utility": self.utility / self.slots,
                "bler": self.failed / max(self.sent, 1), "n_ues": int(len(rates))}


def run_episode(cfg: CellularConfig, drop: int, seed: int, policy: str) -> dict:
    sim = CellularSim(cfg, drop_seed=drop)
    sim.reset(seed=seed)
    meter = Meter(sim)
    while not sim.done:
        measuring, arrivals = sim.measuring, sim.arrivals.copy()
        targets, cqis = decide(sim, policy, sim.t)
        out = sim.step(targets, cqis)
        if measuring:
            meter.record(arrivals, targets, cqis, out)
    return meter.result()


def run_jobs(fn, jobs: list, workers: int) -> list[dict]:
    start, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            print(f"[progress] {done}/{len(jobs)} jobs, {time.time() - start:.0f} s", flush=True)
    print(f"{len(jobs)} jobs in {time.time() - start:.0f} s")
    return rows


def write_csv(rows: list[dict], tag: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------------------------------- E0
def e0(args) -> None:
    cfg = CellularConfig(n_subbands=1)
    sinr_db = []
    for drop in range(args.drops):
        sim = CellularSim(cfg, drop_seed=drop)
        rx = sim.rx_mean
        signal = rx[sim.serving, np.arange(sim.n_ues)]
        sinr_db.append(10 * np.log10(signal / (rx.sum(axis=0) - signal + sim.noise_mw)))
    sinr_db = np.concatenate(sinr_db)
    pct = {p: float(np.percentile(sinr_db, p)) for p in (5, 10, 50, 90, 95)}
    runtimes = {}
    for c, u in ((3, 2), (3, 4), (3, 6), (7, 2), (7, 4)):
        sim = CellularSim(replace(CellularConfig(), cluster_size=c, ues_per_cell=u, n_subbands=4), drop_seed=0)
        sim.reset(seed=0)
        start = time.perf_counter()
        for _ in range(args.timing_slots):
            sim.step(*oracle_decisions(sim)[:2])
        runtimes[f"C={c},U={u}"] = 1e3 * (time.perf_counter() - start) / args.timing_slots
    summary = {"drops": args.drops, "n_ues": int(len(sinr_db)), "sinr_percentiles_db": pct,
               "oracle_ms_per_slot_4_subbands": runtimes}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e0_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.sort(sinr_db)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color="#2a78d6", linewidth=1.4)
    ax.set_xlabel("Wideband SINR of the serving cell (dB)", color=INK2)
    ax.set_ylabel("CDF", color=INK2)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"e0_sinr_cdf.{ext}"), dpi=300)


# ---------------------------------------------------------------------------------------------------- E1
def e1_job(job: tuple) -> list[dict]:
    drop, seed, eta, ues = job
    cfg = CellularConfig(n_subbands=1, cluster_size=3, ues_per_cell=ues, objective="throughput",
                         full_buffer=False, load=eta / (3 * ues))
    return [{"drop": drop, "seed": seed, "eta": eta, "policy": p, **run_episode(cfg, drop, seed, p)}
            for p in POLICIES]


def e1(args) -> None:
    jobs = [(d, s, eta, args.ues) for d in range(args.drops) for s in range(args.seeds) for eta in ETAS]
    rows = run_jobs(e1_job, jobs, args.workers)
    tag = f"e1_U{args.ues}_D{args.drops}_S{args.seeds}"
    write_csv(rows, tag)

    by = {}
    for r in rows:
        by.setdefault((r["drop"], r["seed"]), {}).setdefault(r["eta"], {})[r["policy"]] = r
    cap = {k: {p: v[max(ETAS)][p]["throughput"] for p in POLICIES} for k, v in by.items()}
    ratios = np.array([c["oracle"] / c["cluster_tdma"] for c in cap.values()])
    summary = {"capacity_ratio_oracle_to_tdma": {"median": float(np.median(ratios)), "min": float(ratios.min()),
                                                 "max": float(ratios.max())}, "by_eta": {}, "by_rho": {}}
    pts = []
    for key, per_eta in by.items():
        for eta, pol in per_eta.items():
            offered = pol["oracle"]["throughput"] / pol["oracle"]["delivery"]
            rho = offered / cap[key]["cluster_tdma"]
            h = 1 - pol["cluster_tdma"]["throughput"] / pol["oracle"]["throughput"]
            pts.append((eta, rho, h, *[pol[p]["throughput"] / pol["oracle"]["throughput"] for p in POLICIES[1:]],
                        pol["oracle"]["delivery"], pol["cluster_tdma"]["delivery"]))
    pts = np.array(pts)
    print(f"capacity oracle / cluster TDMA at eta = {max(ETAS)}: median {np.median(ratios):.2f} "
          f"[{ratios.min():.2f}, {ratios.max():.2f}]")
    print(f"{'eta':>6} {'H med':>7} {'reuse':>7} {'TDMA':>7} {'greedy':>7} {'deliv orc':>10} {'deliv TDMA':>11}")
    for eta in ETAS:
        sel = pts[pts[:, 0] == eta]
        s = {"H_median": float(np.median(sel[:, 2])), "full_reuse": float(sel[:, 3].mean()),
             "cluster_tdma": float(sel[:, 4].mean()), "greedy_muting": float(sel[:, 5].mean()),
             "delivery_oracle": float(sel[:, 6].mean()), "delivery_tdma": float(sel[:, 7].mean())}
        summary["by_eta"][str(eta)] = s
        print(f"{eta:>6g} {100*s['H_median']:>6.1f}% {100*s['full_reuse']:>6.1f}% {100*s['cluster_tdma']:>6.1f}% "
              f"{100*s['greedy_muting']:>6.1f}% {100*s['delivery_oracle']:>9.1f}% {100*s['delivery_tdma']:>10.1f}%")
    print(f"{'rho_TDMA':>12} {'n':>4} {'H median':>9} {'H max':>7}")
    for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 1.1), (1.1, 2), (2, 5), (5, 1e9)):
        sel = pts[(pts[:, 1] >= lo) & (pts[:, 1] < hi)]
        if len(sel):
            summary["by_rho"][f"{lo}-{hi:g}"] = {"n": int(len(sel)), "H_median": float(np.median(sel[:, 2])),
                                                "H_max": float(sel[:, 2].max())}
            print(f"{lo:>5g}-{hi:<6g} {len(sel):>4} {100*np.median(sel[:, 2]):>8.1f}% {100*sel[:, 2].max():>6.1f}%")
    with open(os.path.join(OUT, f"{tag}_summary.json"), "w") as handle:
        json.dump({"config": vars(args), "base": asdict(CellularConfig()), **summary}, handle, indent=2)


# ---------------------------------------------------------------------------------------------------- LA0
LA_SELECTORS = ["full_reuse", "greedy_muting", "oracle"]
LA_VARIANTS = ["genie", "local", "fresh", "intent", "intent+commit"]
LA_DELAYS = [0, 4, 8]
LA_SPEEDS = [3.0, 30.0, 60.0]


def la_episode(cfg: CellularConfig, drop: int, seed: int, selector: str, variant: str) -> dict:
    sim = CellularSim(cfg, drop_seed=drop)
    sim.reset(seed=seed)
    adapter = LinkAdapter(sim, variant.split("+")[0])
    commit = variant.endswith("+commit")
    meter, pending = Meter(sim), None
    while not sim.done:
        measuring, arrivals = sim.measuring, sim.arrivals.copy()
        fresh_targets = decide(sim, selector, sim.t)[0]
        targets = fresh_targets if (not commit or pending is None) else pending
        pending = fresh_targets                              # decided now, applied in the next slot
        cqis = adapter.select(targets)
        out = sim.step(targets, cqis)
        adapter.update(targets, cqis, out["success"])
        if measuring:
            meter.record(arrivals, targets, cqis, out)
    return meter.result()


LA_JOBS = os.path.join(OUT, "la0_jobs")


def la0_job(job: tuple) -> list[dict]:
    """One drop and condition; the rows are saved to la0_jobs/ so that an interrupted run resumes there."""
    drop, seed, delay, speed = job
    path = os.path.join(LA_JOBS, f"d{drop}_s{seed}_delay{delay}_v{speed:g}.json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    cfg = CellularConfig(cluster_size=3, ues_per_cell=4, n_subbands=4, objective="pf",
                         cqi_delay=delay, speed_kmh=speed)
    rows = [{"drop": drop, "seed": seed, "delay": delay, "speed": speed, "selector": selector, "variant": variant,
             **la_episode(cfg, drop, seed, selector, variant)}
            for selector in LA_SELECTORS for variant in LA_VARIANTS]
    os.makedirs(LA_JOBS, exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(rows, handle)
    os.replace(path + ".tmp", path)
    return rows


def la0(args) -> None:
    jobs = [(d, s, delay, speed) for d in range(args.drops) for s in range(args.seeds)
            for delay in LA_DELAYS for speed in LA_SPEEDS]
    rows = run_jobs(la0_job, jobs, args.workers)
    tag = f"la0_D{args.drops}_S{args.seeds}"
    write_csv(rows, tag)

    ref = {(r["drop"], r["seed"], r["delay"], r["speed"]): r for r in rows
           if r["selector"] == "oracle" and r["variant"] == "genie"}
    rel = {}
    for r in rows:
        base = ref[(r["drop"], r["seed"], r["delay"], r["speed"])]
        key = (r["delay"], r["speed"], r["selector"], r["variant"])
        rel.setdefault(key, []).append((np.exp((r["log_utility"] - base["log_utility"]) / r["n_ues"]),
                                        r["throughput"] / base["throughput"], r["p5_rate"] / base["p5_rate"],
                                        r["bler"]))
    summary = {}
    for selector in LA_SELECTORS:
        print(f"\n== targets: {selector}: geometric-mean rate / oracle with genie CQI (%), [BLER %]")
        print(f"{'d':>3} {'km/h':>5} " + " ".join(f"{v:>15}" for v in LA_VARIANTS)
              + f" {'OLLA cost':>9} {'aging':>6} {'intent':>7}")
        for delay in LA_DELAYS:
            for speed in LA_SPEEDS:
                m = {v: np.mean(rel[(delay, speed, selector, v)], axis=0) for v in LA_VARIANTS}
                g = {v: 100 * m[v][0] for v in LA_VARIANTS}
                summary[f"{selector}|d={delay}|v={speed:g}"] = {
                    v: {"geo_rate": float(m[v][0]), "throughput": float(m[v][1]), "p5": float(m[v][2]),
                        "bler": float(m[v][3])} for v in LA_VARIANTS}
                print(f"{delay:>3} {speed:>5g} " + " ".join(f"{g[v]:>8.1f} [{100*m[v][3]:>4.1f}]" for v in LA_VARIANTS)
                      + f" {g['genie'] - g['fresh']:>8.1f}p {g['fresh'] - g['local']:>5.1f}p "
                        f"{g['intent'] - g['local']:>6.1f}p")
    with open(os.path.join(OUT, f"{tag}_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", choices=["e0", "e1", "la0"])
    parser.add_argument("--drops", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--ues", type=int, default=4)
    parser.add_argument("--timing-slots", type=int, default=20)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    {"e0": e0, "e1": e1, "la0": la0}[args.exp](args)


if __name__ == "__main__":
    main()
