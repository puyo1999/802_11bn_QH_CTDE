"""
experiments/cellular_gap.py
paper3 experiments without learners (docs/paper3_5g_multicell_plan.md, Section 7).

  e0  calibration: CDF of the wideband all-transmit SINR of the serving cell (large-scale gains only, all
      7 cells transmitting) over drops, and the run time of the exact oracle per slot for C in {3, 7}.
  e1  headroom sweep: finite buffer, one subband, cluster C = 3, throughput objective; for every drop and
      arrival rate the oracle and the references run on identical arrival and fading sequences, and the
      headroom H = 1 - TDMA / oracle is reported against the offered load eta = lambda C U.
  la0 link-adaptation pre-validation (docs/paper3_dynamic_quantization.md, Section 7): full buffer, C = 3,
      U = 4, B = 4, PF objective; the targets of full reuse, greedy muting and the oracle are kept, and only
      the CQI selection changes (genie / local: delayed report + OLLA / intent: announced transmissions +
      OLLA). "intent+commit" decides the targets one slot ahead, as the announcement requires. Utility is
      reported relative to the oracle with genie CQI, per CQI delay d and UE speed.

    python experiments/cellular_gap.py e0
    python experiments/cellular_gap.py e1 --drops 20 --seeds 1          # pilot
    python experiments/cellular_gap.py e1 --drops 200 --seeds 3         # full run
    python experiments/cellular_gap.py la0 --drops 20                    # link-adaptation pre-validation
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
COLORS = {"full_reuse": "#2a78d6", "cluster_tdma": "#eb6834", "greedy_muting": "#1baf7a"}


def decide(sim, policy: str, turn: int):
    if policy == "oracle":
        return oracle_decisions(sim)[:2]
    if policy == "cluster_tdma":
        return baselines.cluster_tdma(sim, turn)
    return getattr(baselines, policy)(sim)


def run_episode(cfg: CellularConfig, drop: int, seed: int, policy: str) -> dict:
    sim = CellularSim(cfg, drop_seed=drop)
    sim.reset(seed=seed)
    offered = sim.backlog_now().sum()
    thp = util = 0.0
    while not sim.done:
        out = sim.step(*decide(sim, policy, sim.t))
        thp, util = thp + out["throughput"], util + out["utility"]
        offered += sim.arrivals.sum() if not sim.done else 0.0
    return {"throughput": thp, "utility": util, "offered": float(offered)}


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
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for rs in pool.map(e1_job, jobs, chunksize=4) for r in rs]
    print(f"{len(jobs)} jobs in {time.time() - start:.0f} s")
    os.makedirs(OUT, exist_ok=True)
    tag = f"e1_U{args.ues}_D{args.drops}_S{args.seeds}"
    with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ratio = {}   # (eta, policy) -> per (drop, seed) throughput / oracle
    by = {}
    for r in rows:
        by.setdefault((r["drop"], r["seed"], r["eta"]), {})[r["policy"]] = r
    for (drop, seed, eta), pol in by.items():
        for p in POLICIES[1:]:
            ratio.setdefault((eta, p), []).append(pol[p]["throughput"] / pol["oracle"]["throughput"])
        ratio.setdefault((eta, "better"), []).append(max(pol[p]["throughput"] for p in POLICIES[1:])
                                                   / pol["oracle"]["throughput"])
        ratio.setdefault((eta, "delivered"), []).append(pol["oracle"]["throughput"] / pol["oracle"]["offered"])
    summary = {}
    print(f"{'eta':>6} {'H median':>9} {'H p10-p90':>14} {'reuse':>7} {'greedy':>7} {'better':>7} {'oracle/offered':>15}")
    for eta in ETAS:
        h = 1 - np.array(ratio[(eta, "cluster_tdma")])
        summary[str(eta)] = {"H_median": float(np.median(h)), "H_p10": float(np.percentile(h, 10)),
                             "H_p90": float(np.percentile(h, 90)),
                             **{f"{p}_mean": float(np.mean(ratio[(eta, p)])) for p in
                                ("full_reuse", "greedy_muting", "better", "delivered")}}
        s = summary[str(eta)]
        print(f"{eta:>6g} {100*s['H_median']:>8.1f}% {100*s['H_p10']:>6.1f}-{100*s['H_p90']:<6.1f}% "
              f"{100*s['full_reuse_mean']:>6.1f}% {100*s['greedy_muting_mean']:>6.1f}% {100*s['better_mean']:>6.1f}% "
              f"{100*s['delivered_mean']:>14.1f}%")
    with open(os.path.join(OUT, f"{tag}_summary.json"), "w") as handle:
        json.dump({"config": vars(args), "base": asdict(CellularConfig()), "etas": summary}, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for eta in ETAS:
        h = 100 * (1 - np.array(ratio[(eta, "cluster_tdma")]))
        ax.scatter(np.full(len(h), eta) * np.exp(np.random.default_rng(0).normal(0, 0.04, len(h))), h,
                   s=7, color="#eb6834", alpha=0.5, linewidths=0)
    ax.plot(ETAS, [100 * summary[str(e)]["H_median"] for e in ETAS], color=INK, linewidth=1.2, label="median")
    ax.axvline(1.0, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ETAS, [f"{e:g}" for e in ETAS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Offered load / cluster TDMA capacity η", color=INK2)
    ax.set_ylabel("Headroom H (%)", color=INK2)
    ax.set_title("(a) Headroom over cluster TDMA", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for p in POLICIES[1:]:
        ax.plot(ETAS, [100 * np.mean(ratio[(e, p)]) for e in ETAS], marker="o", markersize=3,
                color=COLORS[p], label=p.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xticks(ETAS, [f"{e:g}" for e in ETAS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("η", color=INK2)
    ax.set_ylabel("Throughput / oracle (%)", color=INK2)
    ax.set_title("(b) References relative to the oracle", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{tag}.{ext}"), dpi=300)


# ---------------------------------------------------------------------------------------------------- LA0
LA_SELECTORS = ["full_reuse", "greedy_muting", "oracle"]
LA_VARIANTS = ["genie", "local", "intent", "intent+commit"]
LA_DELAYS = [0, 4, 8]
LA_SPEEDS = [3.0, 30.0, 60.0]


def la_episode(cfg: CellularConfig, drop: int, seed: int, selector: str, variant: str) -> dict:
    sim = CellularSim(cfg, drop_seed=drop)
    sim.reset(seed=seed)
    adapter = LinkAdapter(sim, variant.split("+")[0])
    commit = variant.endswith("+commit")
    pending = None
    util = thp = 0.0
    sent = failed = 0
    while not sim.done:
        fresh = decide(sim, selector, sim.t)[0]
        targets = fresh if (not commit or pending is None) else pending
        pending = fresh                                  # decided now, applied in the next slot
        cqis = adapter.select(targets)
        out = sim.step(targets, cqis)
        adapter.update(targets, cqis, out["success"])
        used = (targets >= 0) & (cqis > 0)
        sent, failed = sent + int(used.sum()), failed + int((used & ~out["success"]).sum())
        util, thp = util + out["utility"], thp + out["throughput"]
    return {"utility": util, "throughput": thp, "bler": failed / max(sent, 1)}


def la0_job(job: tuple) -> list[dict]:
    drop, seed, delay, speed = job
    cfg = CellularConfig(cluster_size=3, ues_per_cell=4, n_subbands=4, objective="pf",
                         cqi_delay=delay, speed_kmh=speed)
    rows = []
    for selector in LA_SELECTORS:
        for variant in LA_VARIANTS:
            rows.append({"drop": drop, "seed": seed, "delay": delay, "speed": speed, "selector": selector,
                         "variant": variant, **la_episode(cfg, drop, seed, selector, variant)})
    return rows


def la0(args) -> None:
    jobs = [(d, s, delay, speed) for d in range(args.drops) for s in range(args.seeds)
            for delay in LA_DELAYS for speed in LA_SPEEDS]
    start = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(la0_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            print(f"[progress] {done}/{len(jobs)} jobs, {time.time() - start:.0f} s", flush=True)
    print(f"{len(jobs)} jobs in {time.time() - start:.0f} s")
    os.makedirs(OUT, exist_ok=True)
    tag = f"la0_D{args.drops}_S{args.seeds}"
    with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ref = {(r["drop"], r["seed"], r["delay"], r["speed"]): r["utility"] for r in rows
           if r["selector"] == "oracle" and r["variant"] == "genie"}
    stats = {}
    for r in rows:
        key = (r["delay"], r["speed"], r["selector"], r["variant"])
        stats.setdefault(key, {"ratio": [], "bler": []})
        stats[key]["ratio"].append(r["utility"] / ref[(r["drop"], r["seed"], r["delay"], r["speed"])])
        stats[key]["bler"].append(r["bler"])
    summary = {}
    for selector in LA_SELECTORS:
        print(f"\n== targets: {selector} (PF utility / oracle with genie CQI, %; BLER in brackets)")
        print(f"{'d':>3} {'km/h':>5} " + " ".join(f"{v:>16}" for v in LA_VARIANTS) + f" {'LA loss':>8} {'recovered':>9}")
        for delay in LA_DELAYS:
            for speed in LA_SPEEDS:
                m = {v: float(np.mean(stats[(delay, speed, selector, v)]["ratio"])) for v in LA_VARIANTS}
                b = {v: float(np.mean(stats[(delay, speed, selector, v)]["bler"])) for v in LA_VARIANTS}
                loss = m["genie"] - m["local"]
                rec = (m["intent"] - m["local"]) / loss if loss > 1e-9 else float("nan")
                summary[f"{selector}|d={delay}|v={speed:g}"] = {"ratio": m, "bler": b, "la_loss": loss,
                                                               "recovered_by_intent": rec}
                print(f"{delay:>3} {speed:>5g} " + " ".join(f"{100*m[v]:>8.1f} [{100*b[v]:>4.1f}]" for v in LA_VARIANTS)
                      + f" {100*loss:>7.1f}p {100*rec:>8.0f}%")
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
