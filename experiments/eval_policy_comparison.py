"""Cross-topology policy comparison on the QH simulator (replaces the legacy CW-env Fig. 9).

Every policy is run on the same evaluation episodes as train_ablation.evaluate():
env seed offset = seed + 10000, eval_steps from the config, harsh regime from
env.harsh_start_ratio. Throughput is the value returned by env.step() -- no
post-hoc scaling, offsets, or synthetic noise is applied anywhere.

Policies
  baselines : random, greedy, round_robin, oracle, exact_oracle, co_tdma  (src/training/baseline_policies.py)
  learned   : C1-C4 checkpoints from train_ablation.py ({topo}_{cond}_seed{s}_model.pt)

Outputs (in --output):
  policy_comparison_steps.csv     one row per (topology, policy, seed, step)
  policy_comparison_summary.csv   one row per (topology, policy, seed)
  missing_checkpoints.txt         learned runs that were skipped
  fig_policy_comparison.{pdf,png} cumulative throughput, mean +/- std over seeds

Usage:
  python experiments/eval_policy_comparison.py --topologies T1 T2 T3 T4 T5 T6 \
      --conditions C2 C4 --checkpoints results_csv/main_full results_csv/topo_c2c4
  python experiments/eval_policy_comparison.py --plot-only --output results_csv/policy_comparison
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.topology_registry import load_topologies
from src.training.baseline_policies import POLICIES, BaselinePolicy
from src.training.centralized_execution import CentralizedExecutor
from experiments.train_ablation import CONDITIONS, build_trainer, new_env

EVAL_SEED_OFFSET = 10_000  # identical to train_ablation.evaluate()

LABELS = {
    "random": "Random",
    "greedy": "Greedy (max queue, ARF)",
    "round_robin": "Round-robin (ARF)",
    "oracle": "All-transmit SINR heuristic",
    "exact_oracle": "Exact per-slot oracle (upper bound)",
    "co_tdma": "Co-TDMA (round-robin, one AP/slot)",
    "C1": "C1: local actor, local critic",
    "C2": "C2: CTDE (local actor, central critic)",
    "C3": "C3: joint actor (raw)",
    "C4": "C4: QH-CTDE (joint actor + VQ)",
}
# Fixed per entity (never by rank); slots from the validated reference palette.
# Line style + marker give a second encoding for grayscale print.
STYLE = {
    "random": dict(color="#a3a29b", ls="--", marker="x"),
    "greedy": dict(color="#e34948", ls=":", marker="v"),
    "round_robin": dict(color="#4a3aa7", ls="-.", marker="s"),
    "oracle": dict(color="#52514e", ls=(0, (1, 1)), marker="*"),
    "exact_oracle": dict(color="#0b0b0b", ls="--", marker="P"),
    "co_tdma": dict(color="#008300", ls=(0, (3, 1, 1, 1)), marker="h"),
    "C1": dict(color="#2a78d6", ls="-", marker="o"),
    "C2": dict(color="#eb6834", ls="--", marker="s"),
    "C3": dict(color="#1baf7a", ls="-.", marker="^"),
    "C4": dict(color="#eda100", ls="-", marker="D"),
}


def find_checkpoint(dirs, topo_id, cond, seed):
    for d in dirs:
        path = os.path.join(d, f"{topo_id}_{cond}_seed{seed}_model.pt")
        if os.path.exists(path):
            return path
    return None


def run_episode(env, eval_seed, steps, act_fn, observe_fn=None):
    """Run one evaluation episode; act_fn(obs, env) -> (action, backhaul_bits).

    observe_fn(info) receives the step outcome (used by ARF rate adaptation).
    """
    obs, _ = env.reset(seed=eval_seed)
    rows = []
    for t in range(steps):
        harsh = bool(env.harsh_mode_enabled and env._step >= env.harsh_start_step)
        action, bits = act_fn(obs, env)
        obs, _, done, _, info = env.step(np.asarray(action, dtype=np.int32).reshape(-1))
        if observe_fn is not None:
            observe_fn(info)
        rows.append(dict(step=t, throughput=float(info.get("system_throughput", 0.0)),
                         blocked=int(info.get("overlap_blocked_step", 0)),
                         backhaul_bits=int(bits), harsh=int(harsh)))
        if done:
            break
    return rows


def evaluate(args):
    cfg = load_config(args.config)
    exp = ExperimentConfig.from_dict(cfg)
    steps = args.steps or exp.train.eval_steps
    if exp.communication_delay.enabled:
        sys.exit("communication delay is enabled in the config; this script evaluates the default "
                 "(delay-free) configuration only")
    topos = load_topologies(args.topologies, cfg.get("topology_file", "configs/topologies.yaml"))
    os.makedirs(args.output, exist_ok=True)

    step_rows, summary_rows, missing = [], [], []
    for topo in topos:
        topo_id = topo.metadata.get("id", topo.name)
        counts = [len(s) for s in topo.sta_positions]
        for seed in args.seeds:
            eval_seed = seed + EVAL_SEED_OFFSET
            runs = [(p, None) for p in args.policies] + [(c, "learned") for c in args.conditions]
            for name, kind in runs:
                env = new_env(topo, cfg, steps, seed_offset=eval_seed)
                observe = None
                if kind is None:
                    policy = BaselinePolicy(name, seed=eval_seed)
                    act = lambda obs, e, p=policy: (p.act(e), 0)
                    observe = policy.observe
                else:
                    ckpt = find_checkpoint(args.checkpoints, topo_id, name, seed)
                    if ckpt is None:
                        missing.append(f"{topo_id} {name} seed{seed}")
                        env.close()
                        continue
                    model = build_trainer(env, cfg, name).model
                    state = torch.load(ckpt, map_location="cpu", weights_only=True)
                    model.load_state_dict(state.get("model_state_dict", state), strict=True)
                    model.eval()
                    ex = CentralizedExecutor(model, counts, threshold=exp.agent.gamma_margin,
                                             device="cpu",
                                             execution_mode=CONDITIONS[name]["execution_mode"])

                    def act(obs, e, ex=ex):
                        with torch.no_grad():
                            d = ex.act(obs, deterministic=True)
                        return d.actions, d.transmitted_bits
                try:
                    rows = run_episode(env, eval_seed, steps, act, observe)
                finally:
                    env.close()
                for r in rows:
                    step_rows.append(dict(topology=topo_id, policy=name, seed=seed, **r))
                summary_rows.append(dict(
                    topology=topo_id, policy=name, seed=seed, evaluation_seed=eval_seed,
                    steps=len(rows),
                    throughput_packets=sum(r["throughput"] for r in rows),
                    throughput_normal=sum(r["throughput"] for r in rows if not r["harsh"]),
                    throughput_harsh=sum(r["throughput"] for r in rows if r["harsh"]),
                    blocked_per_step=sum(r["blocked"] for r in rows) / max(len(rows), 1),
                    backhaul_bits_per_step=sum(r["backhaul_bits"] for r in rows) / max(len(rows), 1),
                ))
                print(f"{topo_id} seed{seed} {name:12s} thp={summary_rows[-1]['throughput_packets']:.0f}",
                      flush=True)

    write(os.path.join(args.output, "policy_comparison_steps.csv"), step_rows)
    write(os.path.join(args.output, "policy_comparison_summary.csv"), summary_rows)
    with open(os.path.join(args.output, "missing_checkpoints.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(missing))
    if missing:
        print(f"[WARN] {len(missing)} learned runs skipped (no checkpoint): see missing_checkpoints.txt")


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.plot_paper_figures import apply_rc, save, DOUBLE_COL

    path = os.path.join(args.output, "policy_comparison_steps.csv")
    data = defaultdict(lambda: defaultdict(dict))   # topo -> policy -> seed -> (thp array, harsh array)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = data[r["topology"]][r["policy"]].setdefault(int(r["seed"]), ([], []))
        d[0].append(float(r["throughput"]))
        d[1].append(int(r["harsh"]))
    topos = sorted(data)
    order = [p for p in list(POLICIES) + list(CONDITIONS) if any(p in data[t] for t in topos)]

    apply_rc()
    ncol = 3 if len(topos) > 2 else len(topos)
    nrow = int(np.ceil(len(topos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE_COL, 2.0 * nrow + 0.6), squeeze=False)
    for ax, topo in zip(axes.flat, topos):
        harsh_start = None
        for p in order:
            seeds = data[topo].get(p)
            if not seeds:
                continue
            n = min(len(v[0]) for v in seeds.values())
            cum = np.stack([np.cumsum(v[0][:n]) for v in seeds.values()])
            m, s = cum.mean(0), cum.std(0)
            x = np.arange(1, n + 1)
            st = STYLE[p]
            ax.plot(x, m, color=st["color"], ls=st["ls"], marker=st["marker"], markersize=3,
                    markevery=max(n // 5, 1), lw=1.1, label=LABELS[p])
            ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.12, lw=0)
            h = np.asarray(next(iter(seeds.values()))[1][:n])
            if harsh_start is None and h.any():
                harsh_start = int(np.argmax(h)) + 1
        if harsh_start:
            ax.axvline(harsh_start, color="#52514e", lw=0.6, ls="--")
            ax.text(harsh_start, ax.get_ylim()[1], " harsh", va="top", ha="left", fontsize=6,
                    color="#52514e")
        ax.set_title(topo, loc="left")
        ax.set_xlabel("Evaluation step")
        ax.set_ylabel("Cumulative packets")
    for ax in list(axes.flat)[len(topos):]:
        ax.set_visible(False)
    # Collect legend entries from every panel: learned policies may be absent in some topologies.
    by_label = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(l, h)
    labels = [LABELS[p] for p in order if LABELS[p] in by_label]
    handles = [by_label[l] for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06 - 0.03 * ((len(labels) + 2) // 3)))
    fig.tight_layout()
    save(fig, args.output, "fig_policy_comparison")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/qh_ctde.yaml")
    p.add_argument("--topologies", nargs="+", default=["T1", "T2", "T3", "T4", "T5", "T6"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--policies", nargs="*", default=list(POLICIES), choices=list(POLICIES))
    p.add_argument("--conditions", nargs="*", default=["C2", "C4"], choices=list(CONDITIONS))
    p.add_argument("--checkpoints", nargs="+",
                   default=["results_csv/main_full", "results_csv/topo_c2c4"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output", default="results_csv/policy_comparison")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if not args.plot_only:
        evaluate(args)
    plot(args)


if __name__ == "__main__":
    main()
