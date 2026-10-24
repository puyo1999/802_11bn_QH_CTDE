"""Paper figures for the QH-CTDE ablation (IEEE Access, single-column width).

Fig. 1  VQ codebook initialization: reset-based vs k-means (codes used, MSE, PCA view).
Fig. 2  Training reward curves of C1-C5 per topology (mean +/- std over seeds).
Fig. 3  Codec rate-distortion: held-out MSE and codewords used versus bits per AP.
Fig. 4  Harsh-regime MCS trace (input: eval_h5_hybrid.py --trace).

Usage:
    python experiments/plot_paper_figures.py fig1 --out results_csv/main_full/figures
    python experiments/plot_paper_figures.py fig2 --results results_csv/main_full \
        --out results_csv/main_full/figures [--extra "C4 (reset init)=path/to/dir"]

Only numpy / csv / matplotlib are required.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Categorical slots 1-5 of the validated reference palette, fixed per condition.
# Line style + marker give a second encoding for grayscale print.
STYLE = {
    "C1": dict(color="#2a78d6", ls="-", marker="o", label="C1: local actor, local critic"),
    "C2": dict(color="#eb6834", ls="--", marker="s", label="C2: local actor, central critic"),
    "C3": dict(color="#1baf7a", ls="-.", marker="^", label="C3: joint actor (raw)"),
    "C4": dict(color="#eda100", ls="-", marker="D", label="C4: joint actor + VQ (proposed)"),
    "C5": dict(color="#e87ba4", ls=":", marker="v", label="C5: dummy joint capacity"),
}
EXTRA_STYLE = dict(color="#6f6e69", ls=(0, (1, 1)), marker="x")

SINGLE_COL = 3.5    # inch
DOUBLE_COL = 7.16


def apply_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e3e2dc", "grid.linewidth": 0.5,
        "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {os.path.join(out_dir, name)}.pdf/.png")


# --------------------------------------------------------------------------- Fig. 1
def fig1(args):
    from src.agents.codebook_init import collect_observations, kmeans_centers, quantization_report
    from src.config import ExperimentConfig, load_config
    from src.envs.env_factory import make_env
    from src.envs.topology_registry import get_registry

    exp = ExperimentConfig.from_dict(load_config(args.config))
    env = make_env(exp.env, topology=get_registry("configs/topologies.yaml").get(args.topology),
                   topology_file="configs/topologies.yaml")
    k = 1 << exp.agent.vq_bits

    # Same protocol as tests/test_codebook_init.py.
    reset_rows = np.concatenate([env.reset(seed=43 + i)[0] for i in range(8)], axis=0)
    reset_cb = np.unique(reset_rows.round(6), axis=0)
    reset_cb = np.concatenate([reset_cb, np.tile(reset_cb[:1], (k - len(reset_cb), 1))])
    kmeans_cb = kmeans_centers(collect_observations(env, 4, base_seed=100), k, seed=0)
    heldout = collect_observations(env, 4, base_seed=900)
    rep = {"Reset-based": quantization_report(heldout, reset_cb),
           "k-means": quantization_report(heldout, kmeans_cb)}
    for name, r in rep.items():
        print(name, r)

    names = list(rep)
    colors = ["#a3a29b", STYLE["C4"]["color"]]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.1),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.6]))

    ax = axes[0]
    vals = [rep[n]["codes_used"] for n in names]
    ax.bar(names, vals, color=colors, width=0.55, edgecolor="white", linewidth=1)
    ax.axhline(k, color="#52514e", lw=0.6, ls="--")
    ax.text(-0.45, k, f"$K={k}$", va="bottom", ha="left", fontsize=7, color="#52514e")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Codewords used (held-out)")
    ax.set_ylim(0, k * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a)", loc="left")

    ax = axes[1]
    vals = [rep[n]["mse"] for n in names]
    ax.bar(names, vals, color=colors, width=0.55, edgecolor="white", linewidth=1)
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Quantization MSE")
    ax.set_ylim(min(vals) / 3, max(vals) * 4)
    ax.grid(axis="x", visible=False)
    ax.set_title("(b)", loc="left")

    ax = axes[2]
    mu = heldout.mean(0)
    _, _, vt = np.linalg.svd(heldout - mu, full_matrices=False)
    proj = lambda x: (x - mu) @ vt[:2].T
    p = proj(heldout)
    ax.scatter(p[:, 0], p[:, 1], s=3, color="#c3c2b7", alpha=0.5, lw=0, label="Observations",
               rasterized=True)
    pr, pk = proj(np.unique(reset_cb, axis=0)), proj(kmeans_cb)
    ax.scatter(pk[:, 0], pk[:, 1], s=22, marker="D", color=STYLE["C4"]["color"],
               edgecolor="white", linewidth=0.6, label=f"k-means codewords ({len(pk)})", zorder=3)
    ax.scatter(pr[:, 0], pr[:, 1], s=40, marker="x", color="#0b0b0b", linewidth=1.0,
               label=f"Reset-based codewords ({len(pr)} distinct)", zorder=4)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best", frameon=False, handletextpad=0.3, markerscale=1)
    ax.set_title("(c)", loc="left")

    fig.tight_layout()
    save(fig, args.out, "fig1_codebook_init")


# --------------------------------------------------------------------------- Fig. 2
def read_rewards(results_dir):
    """{(topology, condition): {seed: np.array(reward per episode)}}"""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results_dir, "*_training_rewards.csv"))):
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        if not rows:
            continue
        key = (rows[0]["topology"], rows[0]["condition_id"])
        out[key][int(rows[0]["seed"])] = np.array([float(r["reward"]) for r in rows])
    return out


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    head = c[1:w] / np.arange(1, w)               # growing window at the start
    return np.concatenate([head, (c[w:] - c[:-w]) / w])


def band(runs, w):
    n = min(len(r) for r in runs)
    m = np.stack([smooth(r[:n], w) for r in runs])
    return np.arange(1, n + 1), m.mean(0), m.std(0), len(runs)


def fig2(args):
    data = read_rewards(args.results)
    extras = []
    for spec in args.extra or []:
        label, path = spec.split("=", 1)
        extras.append((label, read_rewards(path)))
    topologies = sorted({t for t, _ in data})
    if not topologies:
        sys.exit(f"no *_training_rewards.csv in {args.results}")

    fig, axes = plt.subplots(1, len(topologies), figsize=(SINGLE_COL * len(topologies), 2.4),
                             squeeze=False, sharey=False)
    for ax, topo in zip(axes[0], topologies):
        for cond, st in STYLE.items():
            if cond not in args.conditions:
                continue
            runs = list(data.get((topo, cond), {}).values())
            if not runs:
                continue
            x, m, s, n = band(runs, args.window)
            ax.plot(x, m, color=st["color"], ls=st["ls"], marker=st["marker"],
                    markevery=max(len(x) // 6, 1), markersize=3.5,
                    label=st["label"])
            ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.15, lw=0)
        for label, ex in extras:
            for (t, cond), seeds in ex.items():
                if t != topo:
                    continue
                x, m, s, n = band(list(seeds.values()), args.window)
                ax.plot(x, m, label=label, markevery=max(len(x) // 6, 1),
                        markersize=3.5, **EXTRA_STYLE)
        ax.set_xlabel("Training episode")
        ax.set_ylabel(f"Episode reward ({args.window}-episode moving avg.)")
        ax.set_title(topo, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.05 * ((len(labels) + 1) // 2)))
    fig.tight_layout()
    save(fig, args.out, args.name)


# --------------------------------------------------------------------------- Fig. 3
def fig3(args):
    """Codec rate-distortion: held-out MSE and codewords used versus bit width b (k-means init)."""
    from src.agents.codebook_init import collect_observations, kmeans_centers, quantization_report
    from src.config import ExperimentConfig, load_config
    from src.envs.env_factory import make_env
    from src.envs.topology_registry import get_registry

    exp = ExperimentConfig.from_dict(load_config(args.config))
    bits = list(range(args.bmin, args.bmax + 1))
    rows = []
    for topo in args.topologies:
        env = make_env(exp.env, topology=get_registry("configs/topologies.yaml").get(topo),
                       topology_file="configs/topologies.yaml")
        # Same protocol as fig1: four random-policy episodes for the codebook, four disjoint ones held out.
        train = collect_observations(env, 4, base_seed=100)
        heldout = collect_observations(env, 4, base_seed=900)
        n_distinct = len(np.unique(train.round(6), axis=0))
        for b in bits:
            r = quantization_report(heldout, kmeans_centers(train, 1 << b, seed=0))
            rows.append(dict(topology=topo, bits=b, codewords=1 << b, centers=r["centers"],
                             codes_used=r["codes_used"], mse=r["mse"], max_abs_err=r["max_abs_err"],
                             distinct_train=n_distinct, payload_reduction=1 - b / (32 * env.obs_dim)))
            print(rows[-1], flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fig3_codec_rate_distortion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    topo_style = {args.topologies[0]: dict(color=STYLE["C4"]["color"], marker="D", ls="-"),
                  args.topologies[-1]: dict(color="#52514e", marker="o", ls="--")}
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL, 1.9))
    for topo in args.topologies:
        rs = [r for r in rows if r["topology"] == topo]
        st = topo_style[topo]
        axes[0].plot(bits, [r["mse"] for r in rs], ms=3.5, lw=1.1, label=topo, **st)
        axes[1].plot(bits, [r["codes_used"] for r in rs], ms=3.5, lw=1.1, label=topo, **st)
    axes[1].plot(bits, [1 << b for b in bits], color="#a3a29b", lw=0.8, ls=":", label="$2^b$")
    for ax, title in zip(axes, ("(a)", "(b)")):
        ax.axvline(exp.agent.vq_bits, color="#52514e", lw=0.6, ls="--")
        ax.set_xticks(bits)
        ax.set_xlabel("Bits per AP, $b$")
        ax.set_yscale("log", base=10 if ax is axes[0] else 2)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Quantization MSE")
    axes[1].set_ylabel("Codewords used")
    axes[1].legend(loc="upper left", frameon=False, handlelength=1.8)
    fig.tight_layout()
    save(fig, args.out, "fig3_codec_rate_distortion")


# --------------------------------------------------------------------------- Fig. 4
def fig4(args):
    """Harsh-regime MCS trace: selected MCS of the learned policy and of ARF versus the decodable bound."""
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(args.trace, encoding="utf-8")):
        data[(r["topology"], r["condition"], r["seed"], r["ap"])][r["mode"]].append(r)
    panels = [tuple(p.split(":")) for p in args.panels]       # TOPO:COND:SEED:AP
    fig, axes = plt.subplots(len(panels), 1, figsize=(SINGLE_COL, 1.55 * len(panels) + 0.35),
                             sharex=True, squeeze=False)
    c4 = STYLE["C4"]["color"]
    for ax, (topo, cond, seed, ap), title in zip(axes[:, 0], panels, "abcdef"):
        runs = data[(topo, cond, seed, ap)]
        lr, ar = runs["learned"], runs["arf"]
        step = np.array([int(r["step"]) for r in lr])
        harsh_start = int(step[np.argmax([r["harsh"] == "1" for r in lr])])

        def series(rs, key):
            return np.array([int(r[key]) if r["attempted"] == "1" and r[key] != "" else np.nan
                             for r in rs], dtype=float)

        bound = series(lr, "bound")
        ax.step(step, bound, where="mid", color="#0b0b0b", lw=0.9, ls="--",
                label="Highest decodable MCS")
        # ARF state is plotted in every step (the MCS it would use), learned MCS only when transmitted.
        ax.plot(step, [int(r["mcs"]) for r in ar], color="#7a7974", lw=1.0, label="ARF MCS")
        m = series(lr, "mcs")
        ok = np.array([r["success"] == "1" for r in lr])
        ax.scatter(step[ok], m[ok], s=7, marker="D", color=c4, lw=0, label="Learned MCS, decoded",
                   zorder=3)
        ax.scatter(step[~ok], m[~ok], s=9, marker="x", color=c4, lw=0.7, label="Learned MCS, failed",
                   zorder=3)
        ax.axvline(harsh_start, color="#52514e", lw=0.6)
        ax.text(harsh_start + 2, 10.5, "harsh $\\rightarrow$", fontsize=6.5, color="#52514e",
                va="center", bbox=dict(fc="white", ec="none", pad=0.5))
        ax.set_ylim(-1.8, 16)
        ax.set_yticks([-1, 0, 5, 10, 15])
        ax.set_yticklabels(["none", "0", "5", "10", "15"])
        ax.set_ylabel("MCS index")
        ax.set_title(f"({title}) {topo}, {cond}, AP {int(ap) + 1}", loc="left")
    axes[-1, 0].set_xlabel("Evaluation step")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1),
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    save(fig, args.out, "fig4_harsh_mcs_trace")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fig1")
    a.add_argument("--config", default="configs/qh_ctde.yaml")
    a.add_argument("--topology", default="T3")
    a.add_argument("--out", default="results_csv/main_full/figures")
    b = sub.add_parser("fig2")
    b.add_argument("--results", default="results_csv/main_full")
    b.add_argument("--extra", action="append", help='"label=dir" extra run set, e.g. old-init C4')
    b.add_argument("--window", type=int, default=20)
    b.add_argument("--conditions", nargs="+", default=list(STYLE), choices=list(STYLE))
    b.add_argument("--out", default="results_csv/main_full/figures")
    b.add_argument("--name", default="fig2_training_curves")
    c = sub.add_parser("fig3")
    c.add_argument("--config", default="configs/qh_ctde.yaml")
    c.add_argument("--topologies", nargs="+", default=["T3", "T6"])
    c.add_argument("--bmin", type=int, default=1)
    c.add_argument("--bmax", type=int, default=8)
    c.add_argument("--out", default="results_csv/main_full/figures")
    d = sub.add_parser("fig4")
    d.add_argument("--trace", default="results_csv/harsh_tuning/h5_trace.csv")
    d.add_argument("--panels", nargs="+", default=["T1:C4:1:0", "T6:C4:1:2"],
                   help="TOPO:COND:SEED:AP (AP is 0-based)")
    d.add_argument("--out", default="results_csv/main_full/figures")
    args = p.parse_args()
    apply_rc()
    {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}[args.cmd](args)


if __name__ == "__main__":
    main()
