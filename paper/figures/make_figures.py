"""Paper figures, built only from committed result CSVs.

    python paper/figures/make_figures.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
HEADROOM = os.path.join(ROOT, "experiments/oracle_gap_results_envv2_headroom/oracle_gap.csv")
FULL_JOBS = os.path.join(ROOT, "experiments/oracle_gap_results_envv2_full/jobs")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
# Reference categorical slots in fixed order, keyed by topology (never by rank).
TOPO_STYLE = {"T1": ("#2a78d6", "o"), "T3": ("#eb6834", "s"), "T4": ("#1baf7a", "^"),
              "T5": ("#eda100", "D"), "T6": ("#e87ba4", "v")}


def load(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def ratio_by_load(rows, policy, reference):
    """{topo: (loads, mean ratio, std ratio)} of policy / reference over seeds."""
    paired = defaultdict(dict)
    for r in rows:
        paired[(r["topology"], float(r["load"]), r["seed"])][r["policy"]] = float(r["throughput"])
    acc = defaultdict(lambda: defaultdict(list))
    for (topo, lam, _), vals in paired.items():
        acc[topo][lam].append(vals[policy] / vals[reference])
    out = {}
    for topo, by_load in acc.items():
        loads = sorted(by_load)
        out[topo] = (loads, [np.mean(by_load[l]) for l in loads], [np.std(by_load[l]) for l in loads])
    return out


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2)


def fig_headroom():
    rows = load(HEADROOM)
    panels = [("co_tdma", "(a) Co-TDMA / exact oracle"),
              ("oracle", "(b) All-transmit SINR heuristic / exact oracle")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (policy, title) in zip(axes, panels):
        data = ratio_by_load(rows, policy, "exact_oracle")
        for topo, (color, marker) in TOPO_STYLE.items():
            loads, mean, std = data[topo]
            ax.errorbar(loads, 100 * np.array(mean), yerr=100 * np.array(std), color=color, marker=marker,
                        markersize=5, linewidth=2, capsize=2, label=topo)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 3, 5, 8], ["0.25", "0.5", "1", "2", "3", "5", "8"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Arrival rate λ (packets / STA / slot)", color=INK2)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_ylim(0, 105)
        style(ax)
    axes[0].set_ylabel("Throughput (% of exact oracle)", color=INK2)
    axes[1].legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_headroom.{ext}"), dpi=300)


def full_table(load: float = 5.0, topos=("T1", "T3", "T4", "T5", "T6")):
    """LaTeX rows: % of exact oracle, mean ± std over seeds, sampled actions (5-seed sweep)."""
    rows = []
    for path in glob.glob(os.path.join(FULL_JOBS, f"*_{load}_*.json")):
        with open(path) as handle:
            rows += json.load(handle)["rows"]
    by = defaultdict(dict)
    for r in rows:
        by[(r["topology"], r["seed"])][r["policy"]] = r["throughput"]

    def cell(topo, policy):
        ratios = [100 * v[policy] / v["exact_oracle"] for (t, _), v in by.items() if t == topo and policy in v]
        return f"{np.mean(ratios):.1f} $\\pm$ {np.std(ratios):.1f}"

    entries = [("Co-TDMA", "co_tdma"), ("All-transmit SINR heuristic", "oracle")]
    entries += [(f"{lname}, MCS {mname}", f"{learner}+{mcs}/sampled")
                for learner, lname in [("mappo", "MAPPO (0-bit)"), ("central", "Central PPO")]
                for mcs, mname in [("learned", "learned"), ("arf", "ARF"), ("genie", "ideal")]]
    lines = [name + " & " + " & ".join(cell(t, key) for t in topos) + r" \\" for name, key in entries]
    with open(os.path.join(OUT, f"table_load{load:g}_rows.tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_headroom()
    full_table(5.0)
    full_table(8.0)
    print("figures written to", OUT)
