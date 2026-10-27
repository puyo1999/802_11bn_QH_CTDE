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
RANDOM_STAGE1 = os.path.join(ROOT, "experiments/oracle_gap_results_random_stage1/oracle_gap.csv")
RANDOM_TOPOS = os.path.join(ROOT, "configs/topologies_random.yaml")
CO_TDMA_CAPACITY = 23.0  # packets per slot: one link at the highest MCS

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


def fading_table(seeds=(0, 1, 2)):
    """LaTeX rows: % of exact oracle without / with Rayleigh fading (mean over common seeds)."""
    def collect(directory):
        by = defaultdict(dict)
        for path in glob.glob(os.path.join(ROOT, "experiments", directory, "jobs", "*.json")):
            with open(path) as handle:
                for r in json.load(handle)["rows"]:
                    if r["seed"] in seeds:
                        by[(r["load"], r["topology"], r["seed"])][r["policy"]] = r["throughput"]
        return by

    plain, faded = collect("oracle_gap_results_envv2_full"), collect("oracle_gap_results_envv2_fading")
    cases = [(5.0, "T3"), (5.0, "T6"), (8.0, "T3"), (8.0, "T6")]

    def mean_pct(by, load, topo, policy):
        return np.mean([100 * by[(load, topo, s)][policy] / by[(load, topo, s)]["exact_oracle"] for s in seeds])

    entries = [("Co-TDMA", "co_tdma"), ("All-transmit SINR heuristic", "oracle"),
               ("MAPPO (0-bit), MCS learned", "mappo+learned/sampled"),
               ("MAPPO (0-bit), MCS ideal", "mappo+genie/sampled"),
               ("Central PPO, MCS learned", "central+learned/sampled"),
               ("Central PPO, MCS ideal", "central+genie/sampled")]
    lines = [name + " & " + " & ".join(f"{mean_pct(plain, l, t, key):.1f} / {mean_pct(faded, l, t, key):.1f}"
                                       for l, t in cases) + r" \\" for name, key in entries]
    with open(os.path.join(OUT, "table_fading_rows.tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def fig_learning_curves(window=25):
    """Training throughput per episode (moving average, mean +- std over seeds) relative to the
    mean exact-oracle throughput on the evaluation seeds of the same configuration."""
    jobs = defaultdict(list)
    oracle, cotdma = defaultdict(list), defaultdict(list)
    for path in glob.glob(os.path.join(FULL_JOBS, "*.json")):
        with open(path) as handle:
            j = json.load(handle)
        kind, load, topo = j["job"][:3]
        if kind == "learner":
            jobs[(load, topo, j["job"][4], j["job"][5])].append(np.asarray(j["curve"], dtype=float))
        else:
            for r in j["rows"]:
                if r["policy"] == "exact_oracle":
                    oracle[(load, topo)].append(r["throughput"])
                elif r["policy"] == "co_tdma":
                    cotdma[(load, topo)].append(r["throughput"])
    series = [("mappo", "learned", "MAPPO, MCS learned", "#2a78d6", "--"),
              ("mappo", "genie", "MAPPO, MCS ideal", "#2a78d6", "-"),
              ("central", "learned", "Central PPO, MCS learned", "#eb6834", "--"),
              ("central", "genie", "Central PPO, MCS ideal", "#eb6834", "-")]
    panels = [(5.0, "T3"), (5.0, "T6"), (8.0, "T3"), (8.0, "T6")]
    kernel = np.ones(window) / window
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for ax, (load, topo) in zip(axes.flat, panels):
        ref = np.mean(oracle[(load, topo)])
        for key, mcs, label, color, ls in series:
            curves = np.stack([np.convolve(c, kernel, mode="valid") for c in jobs[(load, topo, key, mcs)]])
            pct = 100 * curves / ref
            x = np.arange(window, window + pct.shape[1])
            ax.plot(x, pct.mean(0), color=color, linestyle=ls, linewidth=1.6, label=label)
            ax.fill_between(x, pct.mean(0) - pct.std(0), pct.mean(0) + pct.std(0), color=color, alpha=0.12,
                            linewidth=0)
        ax.axhline(100 * np.mean(cotdma[(load, topo)]) / ref, color=INK2, linewidth=1.0, linestyle=":",
                   label="Co-TDMA")
        ax.set_title(f"{topo}, λ = {load:g}", loc="left", fontsize=9, color=INK)
        ax.set_ylim(0, 105)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Training episode", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (% of oracle)", color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_learning_curves.{ext}"), dpi=300)


def fig_topologies():
    """The six reference topologies: APs, associated STAs, and overlap STAs."""
    import sys
    sys.path.insert(0, ROOT)
    from src.envs.topology_registry import get_registry

    registry = get_registry(os.path.join(ROOT, "configs/topologies.yaml"))
    accent = "#eb6834"  # single highlight for overlap STAs; shape also differs (diamond)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, topo_id in zip(axes.flat, [f"T{i}" for i in range(1, 7)]):
        topo = registry.get(topo_id)
        for i, (ap, stas) in enumerate(zip(topo.ap_positions, topo.sta_positions)):
            for k, sta in enumerate(stas):
                overlap = topo.sta_overlap[i][k]
                ax.plot([ap[0], sta[0]], [ap[1], sta[1]], color=INK2, linewidth=0.8, zorder=1)
                for j in overlap:
                    peer = topo.ap_positions[j]
                    ax.plot([peer[0], sta[0]], [peer[1], sta[1]], color=accent, linewidth=0.8,
                            linestyle="--", zorder=1)
                ax.scatter(*sta, s=34 if overlap else 22, marker="D" if overlap else "o",
                           color=accent if overlap else "#ffffff", edgecolor=accent if overlap else INK,
                           linewidth=1.0, zorder=3)
            ax.scatter(*ap, s=70, marker="^", color=INK, zorder=4)
            ax.annotate(f"AP{i + 1}", ap, textcoords="offset points", xytext=(-6, 4), ha="right",
                        fontsize=7, color=INK)
        ax.set_title(topo_id, loc="left", fontsize=9, color=INK, pad=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.22)
        ax.tick_params(labelsize=7, colors=INK2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("x (m)", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", fontsize=8, color=INK2)
    handles = [
        plt.Line2D([], [], marker="^", color=INK, linestyle="", markersize=7, label="AP"),
        plt.Line2D([], [], marker="o", markerfacecolor="#ffffff", markeredgecolor=INK, color=INK2,
                   markersize=5, label="STA and serving link"),
        plt.Line2D([], [], marker="D", color=accent, linestyle="--", markersize=5,
                   label="Overlap STA and overlapping AP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_topologies.{ext}"), dpi=300)


def fig_random_headroom():
    """Headroom H = 1 - Co-TDMA / oracle versus normalized offered load rho = lambda * n_STA / 23."""
    import yaml
    from src.envs.topology_registry import get_registry

    design = {t: spec["design"] for t, spec in yaml.safe_load(open(RANDOM_TOPOS))["topologies"].items()}
    data = ratio_by_load(load(RANDOM_STAGE1), "co_tdma", "exact_oracle")
    n_colors = {3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a", 6: "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for n, color in n_colors.items():
        xs, ys = [], []
        for topo, (loads, mean, _) in data.items():
            if design[topo]["n_aps"] != n:
                continue
            n_sta = n * design[topo]["stas_per_ap"]
            xs += [lam * n_sta / CO_TDMA_CAPACITY for lam in loads]
            ys += [100 * (1 - m) for m in mean]
        ax.scatter(xs, ys, s=7, color=color, alpha=0.7, linewidths=0, label=f"N = {n}")
    fixed = ratio_by_load(load(HEADROOM), "co_tdma", "exact_oracle")
    registry = get_registry()
    xs, ys = [], []
    for topo in TOPO_STYLE:
        n_sta = sum(len(s) for s in registry.get(topo).sta_positions)
        loads, mean, _ = fixed[topo]
        xs += [lam * n_sta / CO_TDMA_CAPACITY for lam in loads]
        ys += [100 * (1 - m) for m in mean]
    ax.scatter(xs, ys, s=16, facecolors="none", edgecolors=INK, linewidths=0.8, label="T1, T3–T6")
    ax.axvline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Offered load / Co-TDMA capacity ρ", color=INK2)
    ax.set_ylabel("Headroom H (%)", color=INK2)
    ax.set_title("(a) Headroom versus offered load", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
    ax = axes[1]
    for k, lam in enumerate((5.0, 8.0)):
        groups = [[100 * (1 - data[t][1][data[t][0].index(lam)]) for t in data if design[t]["n_aps"] == n]
                  for n in n_colors]
        pos = np.arange(len(n_colors)) + (k - 0.5) * 0.36
        box = ax.boxplot(groups, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                         medianprops={"color": INK, "linewidth": 1})
        for patch in box["boxes"]:
            patch.set(facecolor=("#2a78d6", "#eb6834")[k], alpha=0.6, linewidth=0.6)
        ax.plot([], [], color=("#2a78d6", "#eb6834")[k], linewidth=6, alpha=0.6, label=f"λ = {lam:g}")
    ax.set_xticks(range(len(n_colors)), [str(n) for n in n_colors])
    ax.set_xlabel("Number of APs N", color=INK2)
    ax.set_title("(b) Headroom by network size", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_random_headroom.{ext}"), dpi=300)


if __name__ == "__main__":
    fig_topologies()
    fig_headroom()
    fig_learning_curves()
    fig_random_headroom()
    full_table(5.0)
    full_table(8.0)
    fading_table()
    print("figures written to", OUT)
