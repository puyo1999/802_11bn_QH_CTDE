"""
paper3/figures/make_figures.py
Figures of paper3 from the committed results in experiments/cellular_results (E1: 200 drops, 3 seeds;
LA0 and LA1: 20 drops, 1 seed).

    fig_sinr_cdf     E0: CDF of the wideband all-transmit SINR of the serving cell (recomputed from the
                     simulator, 50 drops, large-scale gains only)
    fig_headroom     E1: headroom over cluster TDMA versus eta and the references relative to the oracle
    fig_delay        E1 with the max-weight objective: share of drops whose queues are stable, and the median and
                     95th-percentile packet delay on the drops where the oracle and cluster TDMA are stable
    fig_la0          LA0: PF utility relative to the oracle with genie CQI for the CQI selections, and the
                     link-adaptation loss with the share recovered by intent messages
    fig_la1          LA1: share of the link-adaptation loss that established remedies recover with causal
                     targets, and the split of the loss of OLLA and of expected-throughput selection into the
                     operating cost and the cost of aging

Every figure is written as PDF (for LaTeX) and PNG (for viewing).

    python paper3/figures/make_figures.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "experiments/cellular_results")
E1 = os.path.join(RESULTS, "e1_U4_D200_S3.csv")
E1_STATS = os.path.join(RESULTS, "e1_stats.json")   # experiments/cellular_results/e1_stats.py
LA0 = os.path.join(RESULTS, "la0_D20_S1.csv")
LA1 = os.path.join(RESULTS, "la1_D20_S1_summary.json")
sys.path.insert(0, ROOT)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def load(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def fig_sinr_cdf(drops: int = 50) -> None:
    from src.cellular.linkadapt import large_scale_all_transmit_sinr
    from src.cellular.sim import CellularConfig, CellularSim
    sinr = np.concatenate([10 * np.log10(large_scale_all_transmit_sinr(CellularSim(CellularConfig(), drop_seed=d)))
                           for d in range(drops)])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.sort(sinr)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=BLUE, linewidth=1.4)
    for p in (5, 50, 95):
        v = np.percentile(sinr, p)
        ax.axvline(v, color=INK2, linewidth=0.6, linestyle=":")
        ax.text(v + 0.5, p / 100 - 0.06 if p > 5 else 0.1, f"{p}%: {v:.1f} dB", fontsize=6.5, color=INK2)
    ax.set_xlabel("Wideband SINR of the serving cell (dB)", color=INK2, fontsize=8)
    ax.set_ylabel("CDF", color=INK2, fontsize=8)
    style(ax)
    save(fig, "fig_sinr_cdf")


def fig_headroom() -> None:
    """Steady-state E1: (a) headroom against the load relative to the saturated capacity of cluster TDMA,
    (b) references relative to the oracle and the delivery ratio of the oracle against eta."""
    by = defaultdict(dict)
    for r in load(E1):
        by[(r["drop"], r["seed"], float(r["eta"]))][r["policy"]] = {k: float(r[k]) for k in ("throughput", "delivery")}
    etas = sorted({k[2] for k in by})
    top = max(etas)
    cap = {(d, s): pol["cluster_tdma"]["throughput"] for (d, s, eta), pol in by.items() if eta == top}
    pts, ratio = [], defaultdict(list)
    for (d, s, eta), pol in by.items():
        offered = pol["oracle"]["throughput"] / pol["oracle"]["delivery"]
        pts.append((offered / cap[(d, s)], 100 * (1 - pol["cluster_tdma"]["throughput"] / pol["oracle"]["throughput"])))
        for p in ("full_reuse", "cluster_tdma", "greedy_muting"):
            ratio[(eta, p)].append(pol[p]["throughput"] / pol["oracle"]["throughput"])
        ratio[(eta, "delivery")].append(pol["oracle"]["delivery"])
    pts = np.array(pts)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=3, color=ORANGE, alpha=0.2, linewidths=0, label="drop, seed and load")
    edges = np.exp(np.linspace(np.log(pts[:, 0].min()), np.log(pts[:, 0].max()), 13))
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = pts[(pts[:, 0] >= lo) & (pts[:, 0] <= hi)]
        if len(sel) >= 10:
            centers.append(np.sqrt(lo * hi))
            medians.append(np.median(sel[:, 1]))
    ax.plot(centers, medians, color=INK, linewidth=1.2, label="binned median")
    ax.axvline(1.0, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Offered load / saturated cluster TDMA capacity", color=INK2, fontsize=8)
    ax.set_ylabel("Headroom H (%)", color=INK2, fontsize=8)
    ax.set_title("(a) Headroom over cluster TDMA", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[1]
    for p, color, label in (("full_reuse", BLUE, "Full reuse"), ("greedy_muting", GREEN, "Greedy muting"),
                            ("cluster_tdma", ORANGE, "Cluster TDMA")):
        ax.plot(etas, [100 * np.mean(ratio[(e, p)]) for e in etas], marker="o", markersize=3, color=color,
                label=label)
    ax.plot(etas, [100 * np.mean(ratio[(e, "delivery")]) for e in etas], color=INK2, linestyle="--",
            linewidth=1.0, label="Oracle delivery ratio")
    ax.set_xscale("log")
    ax.set_xticks(etas, [f"{e:g}" for e in etas])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Offered load η", color=INK2, fontsize=8)
    ax.set_ylabel("Throughput / oracle (%)", color=INK2, fontsize=8)
    ax.set_title("(b) References relative to the oracle", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    for ax in axes:
        style(ax)
    save(fig, "fig_headroom")


def la0_relative(selector: str = "oracle") -> dict:
    """{(delay, speed, variant): mean geometric-mean rate relative to the oracle with genie CQI, in %}."""
    rows = load(LA0)
    ref = {(r["drop"], r["seed"], r["delay"], r["speed"]): float(r["log_utility"]) for r in rows
           if r["selector"] == "oracle" and r["variant"] == "genie"}
    rel = defaultdict(list)
    for r in rows:
        if r["selector"] == selector:
            base = ref[(r["drop"], r["seed"], r["delay"], r["speed"])]
            rel[(int(r["delay"]), float(r["speed"]), r["variant"])].append(
                np.exp((float(r["log_utility"]) - base) / int(r["n_ues"])))
    return {k: 100 * np.mean(v) for k, v in rel.items()}


def fig_la0(selector: str = "oracle", delay: int = 4) -> None:
    """LA0: (a) geometric-mean rate per CQI selection versus speed, (b) decomposition of the link-adaptation loss
    into the operating cost of OLLA (genie - fresh) and the cost of aging (fresh - local), with the part
    recovered by intent-aware selection."""
    mean = la0_relative(selector)
    speeds = sorted({k[1] for k in mean})
    delays = sorted({k[0] for k in mean})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for variant, color, label in (("genie", INK, "Genie CQI"), ("fresh", GOLD, "Fresh report + OLLA"),
                                  ("intent", GREEN, "Intent + OLLA"), ("local", BLUE, "Delayed report + OLLA"),
                                  ("intent+commit", ORANGE, "Intent + OLLA, committed")):
        ax.plot(speeds, [mean[(delay, v, variant)] for v in speeds], marker="o", markersize=3, color=color,
                label=label)
    ax.set_xscale("log")
    ax.set_xticks(speeds, [f"{v:g}" for v in speeds])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("UE speed (km/h)", color=INK2, fontsize=8)
    ax.set_ylabel("Geometric-mean rate / oracle (%)", color=INK2, fontsize=8)
    ax.set_title(f"(a) CQI selection, d = {delay}", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6, frameon=False, loc="lower left")

    ax = axes[1]
    width = 0.8 / len(delays)
    for k, d in enumerate(delays):
        pos = np.arange(len(speeds)) + (k - (len(delays) - 1) / 2) * width
        olla = np.array([mean[(d, v, "genie")] - mean[(d, v, "fresh")] for v in speeds])
        aging = np.array([mean[(d, v, "fresh")] - mean[(d, v, "local")] for v in speeds])
        rec = np.array([mean[(d, v, "intent")] - mean[(d, v, "local")] for v in speeds])
        ax.bar(pos, olla, width * 0.9, color=GOLD, label="OLLA operating cost" if k == 0 else None)
        ax.bar(pos, aging, width * 0.9, bottom=olla, color=BLUE, alpha=0.4 + 0.25 * k, label=f"aging, d = {d}")
        ax.scatter(pos, olla + aging - rec, marker="_", s=60, color=GREEN, zorder=3,
                   label="after intent recovery" if k == 0 else None)
    ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
    ax.set_xlabel("UE speed (km/h)", color=INK2, fontsize=8)
    ax.set_ylabel("Loss in geometric-mean rate (points)", color=INK2, fontsize=8)
    ax.set_title("(b) Link-adaptation loss", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    for ax in axes:
        style(ax)
    save(fig, "fig_la0")


def fig_la1(selector: str = "causal_greedy", delay: int = 4) -> None:
    """LA1 with causal targets: (a) share of the link-adaptation loss (genie - OLLA at 10%) that each remedy
    recovers versus speed, (b) loss relative to genie CQI of OLLA and of expected-throughput selection with known
    correlation, each split into the operating cost (genie - fresh) and the cost of aging (fresh - aged)."""
    with open(LA1) as handle:
        summary = json.load(handle)
    speeds = [3.0, 30.0, 60.0]

    def geo(v, variant, d=delay):
        return 100 * summary[f"{selector}|d={d}|v={v:g}"][variant]["geo_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for variant, color, marker, label in (
            ("olla_tuned", INK, "o", "OLLA, tuned target"),
            ("et_estimated", BLUE, "s", "ET, lag-1 estimate"),
            ("et_multilag", GREEN, "^", "ET, multi-lag estimate"),
            ("et_known", GOLD, "D", "ET, known correlation"),
            ("et_signal", ORANGE, "v", "ET, signal-only report"),
            ("et_calibrated", "#8a5cd6", "x", "ET, calibrated")):
        share = [100 * summary[f"{selector}|d={delay}|v={v:g}"]["recovered_share_of_la_loss"][variant]
                 for v in speeds]
        floor = -25.0
        shown = [max(x, floor) for x in share]
        ax.plot(range(len(speeds)), shown, marker=marker, markersize=3.5, color=color, label=label, linewidth=1)
        for i, x in enumerate(share):
            if x < floor:    # off the axis: drawn at the floor and labeled with its value
                ax.annotate(f"{x:.0f}%", (i, floor), xytext=(6, 2), textcoords="offset points", fontsize=6,
                            color=color)
    ax.set_ylim(-28, 60)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
    ax.set_xlabel("UE speed (km/h)", color=INK2, fontsize=8)
    ax.set_ylabel("Recovered share of LA loss (%)", color=INK2, fontsize=8)
    ax.set_title(f"(a) Remedies, d = {delay}", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6, loc="lower right", framealpha=1, edgecolor="none")

    ax = axes[1]
    width = 0.38
    for k, (rule, fresh, aged, label) in enumerate((("OLLA", "fresh", "olla0.1", "OLLA 10%"),
                                                     ("ET", "et_fresh", "et_known", "ET, known corr."))):
        pos = np.arange(len(speeds)) + (k - 0.5) * width
        operating = np.array([geo(v, "genie") - geo(v, fresh) for v in speeds])
        aging = np.array([geo(v, fresh) - geo(v, aged) for v in speeds])
        ax.bar(pos, operating, width * 0.9, color=GOLD, alpha=0.55 + 0.45 * k,
               label=f"{label}: operating" )
        ax.bar(pos, aging, width * 0.9, bottom=operating, color=BLUE, alpha=0.45 + 0.45 * k,
               label=f"{label}: aging")
    ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
    ax.set_xlabel("UE speed (km/h)  [left bar OLLA, right bar ET]", color=INK2, fontsize=8)
    ax.set_ylabel("Loss in geometric-mean rate (points)", color=INK2, fontsize=8)
    ax.set_title(f"(b) Loss split, d = {delay}", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    for ax in axes:
        style(ax)
    save(fig, "fig_la1")


def fig_delay() -> None:
    """E1 with the max-weight objective: (a) share of drops on which every queue of the policy is stable within the
    episode, (b) median over drops of the median and the 95th-percentile packet delay, on the drops where both
    the oracle and cluster TDMA are stable; bars are bootstrap 95% CIs over drops."""
    with open(E1_STATS) as handle:
        stats = json.load(handle)["maxweight"]["by_eta"]
    etas = sorted(float(e) for e in stats)[:-1]          # the highest load only sets the saturated capacity
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for p, color, label in (("oracle", INK, "Oracle"), ("greedy_muting", GREEN, "Greedy muting"),
                            ("full_reuse", BLUE, "Full reuse"), ("cluster_tdma", ORANGE, "Cluster TDMA")):
        est = np.array([stats[str(e)][f"stable_{p}"] for e in etas]) * 100
        ax.errorbar(etas, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]], marker="o", markersize=3,
                    color=color, capsize=2, linewidth=1.1, label=label)
    ax.set_xticks(etas, [f"{e:g}" for e in etas], rotation=45)
    ax.set_xlabel("Offered load η", color=INK2, fontsize=8)
    ax.set_ylabel("Drops with stable queues (%)", color=INK2, fontsize=8)
    ax.set_title("(a) Stability under max-weight scheduling", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax = axes[1]
    sel = [e for e in etas if "delay_p95_oracle" in stats[str(e)]]
    for p, color, label in (("oracle", INK, "Oracle"), ("cluster_tdma", ORANGE, "Cluster TDMA")):
        for k, style_, name in (("delay_median", "-", "median"), ("delay_p95", "--", "95th pct.")):
            est = np.array([stats[str(e)][f"{k}_{p}"] for e in sel])
            ax.errorbar(sel, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]], marker="o", markersize=3,
                        color=color, linestyle=style_, capsize=2, linewidth=1.1, label=f"{label}, {name}")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 5000)
    ax.set_xticks(sel, [f"{e:g}" for e in sel], rotation=45)
    ax.set_xlabel("Offered load η", color=INK2, fontsize=8)
    ax.set_ylabel("Packet delay (slots of 1 ms)", color=INK2, fontsize=8)
    ax.set_title("(b) Delay where both are stable", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)
    for ax in axes:
        style(ax)
    save(fig, "fig_delay")


if __name__ == "__main__":
    fig_sinr_cdf()
    fig_headroom()
    fig_delay()
    fig_la0()
    fig_la1()
