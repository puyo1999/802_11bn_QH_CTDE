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
RANDOM_STAGE2_JOBS = os.path.join(ROOT, "experiments/oracle_gap_results_random_stage2/jobs")
T_CRIT_19 = 2.093  # two-sided 95% Student t, 19 degrees of freedom
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


TOPOLOGY_PANELS = [("configs/topologies.yaml", t) for t in ("T1", "T3", "T4", "T5", "T6")] + \
    [("configs/topologies_random.yaml", "R052")]
PANEL_SPAN = 66.0  # m, identical window for every panel so that distances are comparable


def all_transmit_sinr_db(ap_positions, xx, yy, tx_dbm=15.0, noise_dbm=-95.0):
    """Best SINR over APs at each grid point when every AP transmits (path loss of eq. (1))."""
    dist = np.stack([np.hypot(xx - x, yy - y) for x, y in ap_positions])
    power = 10 ** ((tx_dbm - 46.67 - 30 * np.log10(np.maximum(dist, 1.0))) / 10)
    noise = 10 ** (noise_dbm / 10)
    sinr = power / (power.sum(axis=0, keepdims=True) - power + noise)
    return 10 * np.log10(sinr.max(axis=0))


def fig_topologies():
    """T1, T3-T6 and one random topology on a common scale, over the all-transmit SINR map."""
    import sys
    sys.path.insert(0, ROOT)
    from src.envs.topology_registry import get_registry

    fixed_h = ratio_by_load(load(HEADROOM), "co_tdma", "exact_oracle")
    random_h = ratio_by_load(load(RANDOM_STAGE1), "co_tdma", "exact_oracle")
    accent = "#eb6834"  # single highlight for overlap STAs; shape also differs (diamond)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.5))
    for ax, (spec, topo_id) in zip(axes.flat, TOPOLOGY_PANELS):
        topo = get_registry(os.path.join(ROOT, spec)).get(topo_id)
        aps = np.asarray(topo.ap_positions, dtype=float)
        points = np.vstack([aps] + [np.asarray(s, dtype=float) for s in topo.sta_positions])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        x0, y0 = center - PANEL_SPAN / 2
        xx, yy = np.meshgrid(np.linspace(x0, x0 + PANEL_SPAN, 300), np.linspace(y0, y0 + PANEL_SPAN, 300))
        sinr = all_transmit_sinr_db(aps, xx, yy)
        image = ax.pcolormesh(xx, yy, sinr, cmap="Blues", vmin=-10, vmax=40, shading="auto",
                              rasterized=True, zorder=0)
        ax.contour(xx, yy, sinr, levels=[2.0], colors=INK2, linewidths=0.6, linestyles="--", zorder=1)
        # Sample points of everything drawn, so that each AP label goes where it overlaps least.
        drawn = [points]
        for i, stas in enumerate(topo.sta_positions):
            for k, sta in enumerate(stas):
                for j in [i] + list(topo.sta_overlap[i][k]):
                    drawn.append(np.linspace(aps[j], np.asarray(sta, dtype=float), 12))
        drawn = np.vstack(drawn)
        for i, (ap, stas) in enumerate(zip(aps, topo.sta_positions)):
            for k, sta in enumerate(stas):
                overlap = topo.sta_overlap[i][k]
                ax.plot([ap[0], sta[0]], [ap[1], sta[1]], color=INK, linewidth=0.8, zorder=2)
                for j in overlap:
                    peer = aps[j]
                    ax.plot([peer[0], sta[0]], [peer[1], sta[1]], color=accent, linewidth=0.9,
                            linestyle=(0, (3, 2)), zorder=2)
                ax.scatter(*sta, s=30 if overlap else 20, marker="D" if overlap else "o",
                           color=accent if overlap else "#ffffff", edgecolor=accent if overlap else INK,
                           linewidth=0.9, zorder=3)
            ax.scatter(*ap, s=60, marker="^", color=INK, edgecolor="#ffffff", linewidth=0.6, zorder=4)
            angles = np.deg2rad(np.arange(0, 360, 45) + 90)
            spots = ap + 5.0 * np.column_stack([np.cos(angles), np.sin(angles)])
            clearance = [np.min(np.linalg.norm(drawn - spot, axis=1)) for spot in spots]
            ax.text(*spots[int(np.argmax(clearance))], f"AP{i + 1}", ha="center", va="center",
                    fontsize=6.5, color=INK, zorder=5)
        loads, mean, _ = (fixed_h if topo_id.startswith("T") else random_h)[topo_id]
        h8 = 100 * (1 - mean[loads.index(8.0)])
        n_sta = sum(len(s) for s in topo.sta_positions)
        name = topo_id if topo_id.startswith("T") else f"Random ({topo_id})"
        ax.set_title(f"{name}: N = {topo.n_aps}, {n_sta} STAs, H = {h8:.0f}%", loc="left", fontsize=7.5,
                     color=INK, pad=4)
        ax.set_xlim(x0, x0 + PANEL_SPAN)
        ax.set_ylim(y0, y0 + PANEL_SPAN)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
    # 20 m scale bar in the first panel
    ax = axes[0, 0]
    x1, y1 = ax.get_xlim()[0] + 4, ax.get_ylim()[0] + 4
    ax.plot([x1, x1 + 20], [y1, y1], color=INK, linewidth=1.5, zorder=5)
    ax.text(x1 + 10, y1 + 1.5, "20 m", ha="center", va="bottom", fontsize=6.5, color=INK, zorder=5)
    handles = [
        plt.Line2D([], [], marker="^", color=INK, linestyle="", markersize=6, label="AP"),
        plt.Line2D([], [], marker="o", markerfacecolor="#ffffff", markeredgecolor=INK, color=INK,
                   markersize=4.5, label="STA and serving link"),
        plt.Line2D([], [], marker="D", color=accent, linestyle=(0, (3, 2)), markersize=4.5,
                   label="Overlap STA and overlapping AP"),
        plt.Line2D([], [], color=INK2, linestyle="--", linewidth=0.8, label="All-transmit SINR = 2 dB"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.02, 0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.4, w_pad=0.4)
    cax = fig.add_axes([0.62, 0.07, 0.34, 0.016])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label("Best SINR when all APs transmit (dB)", fontsize=7, color=INK2)
    bar.ax.tick_params(labelsize=6.5, colors=INK2)
    bar.outline.set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_topologies.{ext}"), dpi=300)


def fig_random_headroom():
    """Headroom H = 1 - Co-TDMA / oracle versus normalized offered load eta = lambda * n_STA / 23."""
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
    ax.set_xlabel("Offered load / Co-TDMA capacity η", color=INK2)
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


def job_ratios(job_dir, load_filter=None):
    """{(load, topo): {policy: mean over seeds of throughput / exact oracle}} from per-job JSONs."""
    by = defaultdict(dict)
    for path in glob.glob(os.path.join(job_dir, "*.json")):
        with open(path) as handle:
            for r in json.load(handle)["rows"]:
                by[(r["load"], r["topology"], r["seed"])][r["policy"]] = r["throughput"]
    acc = defaultdict(lambda: defaultdict(list))
    for (lam, topo, _), vals in by.items():
        if "exact_oracle" not in vals or (load_filter and lam not in load_filter):
            continue
        for policy, thp in vals.items():
            acc[(lam, topo)][policy].append(thp / vals["exact_oracle"])
    return {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in acc.items()}


def fig_random_gap():
    """(a) Advantage of Central PPO with ideal MCS over Co-TDMA versus headroom H; (b) gap components."""
    rand = job_ratios(RANDOM_STAGE2_JOBS)
    fixed = job_ratios(FULL_JOBS)
    learner = "central+genie/sampled"
    load_style = {5.0: ("#2a78d6", "o"), 8.0: ("#eb6834", "s")}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    xs_all, ys_all = [], []
    for lam, (color, marker) in load_style.items():
        pts = [(100 * (1 - v["co_tdma"]), 100 * (v[learner] - v["co_tdma"])) for (l, t), v in rand.items() if l == lam]
        xs, ys = zip(*pts)
        xs_all += xs
        ys_all += ys
        ax.scatter(xs, ys, s=18, color=color, marker=marker, alpha=0.85, linewidths=0, label=f"Random, λ = {lam:g}")
        fx = [(100 * (1 - v["co_tdma"]), 100 * (v[learner] - v["co_tdma"])) for (l, t), v in fixed.items()
              if l == lam and learner in v]
        ax.scatter(*zip(*fx), s=26, facecolors="none", edgecolors=color, marker=marker, linewidths=1.0,
                   label=f"T1, T3–T6, λ = {lam:g}")
    grid = np.linspace(0, 80, 2)
    ax.plot(grid, grid, color=INK2, linewidth=0.8, linestyle="--", label="Exact oracle")
    ax.axhline(0, color=INK, linewidth=0.8)
    slope, intercept = np.polyfit(xs_all, ys_all, 1)
    ax.plot(grid, slope * grid + intercept, color=INK, linewidth=0.8, linestyle=":",
            label=f"Fit: {slope:.2f}H − {abs(intercept):.1f}")
    ax.set_xlabel("Headroom H (%)", color=INK2)
    ax.set_ylabel("Advantage over Co-TDMA (pt)", color=INK2)
    ax.set_title("(a) Central PPO, ideal MCS, versus Co-TDMA", loc="left", fontsize=9, color=INK)
    ax.set_xlim(0, 80)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    style(ax)
    ax = axes[1]
    comps = [("Δ$_{LA}$ MAPPO", "mappo+genie/sampled", "mappo+learned/sampled"),
             ("Δ$_{LA}$ Central", "central+genie/sampled", "central+learned/sampled"),
             ("Δ$_{I}$ learned", "central+learned/sampled", "mappo+learned/sampled"),
             ("Δ$_{I}$ ideal", "central+genie/sampled", "mappo+genie/sampled")]
    for k, (lam, (color, _)) in enumerate(load_style.items()):
        data = [[100 * (v[a] - v[b]) for (l, t), v in rand.items() if l == lam] for _, a, b in comps]
        pos = np.arange(len(comps)) + (k - 0.5) * 0.36
        box = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True,
                         medianprops={"color": INK, "linewidth": 1},
                         flierprops={"markersize": 2.5, "markeredgecolor": INK2})
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=0.6, linewidth=0.6)
        ax.plot([], [], color=color, linewidth=6, alpha=0.6, label=f"λ = {lam:g}")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(comps)), [c[0] for c in comps], fontsize=7.5)
    ax.set_ylabel("Gap (percentage points)", color=INK2)
    ax.set_title("(b) Gap components, 20 topologies", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_random_gap.{ext}"), dpi=300)


def random_table():
    """LaTeX rows: gap components over the 20 stage-2 topologies, mean [95% CI] in percentage points."""
    rand = job_ratios(RANDOM_STAGE2_JOBS)
    entries = [("Headroom $H$", lambda v: 1 - v["co_tdma"]),
               ("MAPPO, MCS learned", lambda v: v["mappo+learned/sampled"]),
               ("MAPPO, MCS ideal", lambda v: v["mappo+genie/sampled"]),
               ("Central PPO, MCS learned", lambda v: v["central+learned/sampled"]),
               ("Central PPO, MCS ideal", lambda v: v["central+genie/sampled"]),
               (r"$\Delta_{\text{LA}}$, MAPPO", lambda v: v["mappo+genie/sampled"] - v["mappo+learned/sampled"]),
               (r"$\Delta_{\text{LA}}$, Central PPO", lambda v: v["central+genie/sampled"] - v["central+learned/sampled"]),
               (r"$\Delta_{\text{I}}$, MCS learned", lambda v: v["central+learned/sampled"] - v["mappo+learned/sampled"]),
               (r"$\Delta_{\text{I}}$, MCS ideal", lambda v: v["central+genie/sampled"] - v["mappo+genie/sampled"]),
               (r"$\Delta_{\text{opt}}$", lambda v: 1 - v["central+genie/sampled"]),
               ("Central PPO, MCS ideal $-$ Co-TDMA", lambda v: v["central+genie/sampled"] - v["co_tdma"])]
    lines = []
    for name, fn in entries:
        cells = []
        for lam in (5.0, 8.0):
            x = 100 * np.array([fn(v) for (l, t), v in rand.items() if l == lam])
            half = T_CRIT_19 * x.std(ddof=1) / np.sqrt(len(x))
            cells.append(f"{x.mean():.1f} [{x.mean() - half:.1f}, {x.mean() + half:.1f}]")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "table_random_rows.tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_topologies()
    fig_headroom()
    fig_learning_curves()
    fig_random_headroom()
    fig_random_gap()
    random_table()
    full_table(5.0)
    full_table(8.0)
    fading_table()
    print("figures written to", OUT)
