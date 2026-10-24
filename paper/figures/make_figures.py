"""Paper figures, built only from committed result CSVs.

    python paper/figures/make_figures.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
HEADROOM = os.path.join(ROOT, "experiments/oracle_gap_results_envv2_headroom/oracle_gap.csv")
PILOT = os.path.join(ROOT, "experiments/oracle_gap_results_envv2_mcs_pilot/oracle_gap.csv")

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


def pilot_table():
    """LaTeX rows for the MCS-separation pilot: % of exact oracle (argmax / sampled)."""
    rows = load(PILOT)
    val = {(r["topology"], r["policy"]): float(r["throughput"]) for r in rows}
    topos = ["T3", "T5", "T6"]
    lines = []
    for name, key in [("Co-TDMA", "co_tdma"), ("All-transmit SINR heuristic", "oracle")]:
        lines.append(name + " & " + " & ".join(f"{100 * val[(t, key)] / val[(t, 'exact_oracle')]:.0f}" for t in topos) + r" \\")
    for learner, lname in [("mappo", "MAPPO (0-bit)"), ("central", "Central PPO")]:
        for mcs, mname in [("learned", "learned"), ("arf", "ARF"), ("genie", "genie")]:
            cells = []
            for t in topos:
                ref = val[(t, "exact_oracle")]
                a, s = val[(t, f"{learner}+{mcs}/argmax")], val[(t, f"{learner}+{mcs}/sampled")]
                cells.append(f"{100 * a / ref:.0f} / {100 * s / ref:.0f}")
            lines.append(f"{lname}, MCS {mname} & " + " & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "table_pilot_rows.tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_headroom()
    pilot_table()
    print("figures written to", OUT)
