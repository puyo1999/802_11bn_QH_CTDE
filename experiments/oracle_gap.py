"""
experiments/oracle_gap.py
Oracle gap on the 6 fixed topologies: Random / Local Greedy / Co-TDMA / 0-bit MAPPO
against a genie per-slot exact oracle, with the mute action enabled.

The question this answers: how much throughput does a learner with *no* inter-AP
messages leave on the table compared with perfect coordination? If the gap is large,
it is worth measuring how many bits per TXOP close it.

    python experiments/oracle_gap.py                 # full run (2 loads x 6 topologies x 3 seeds)
    python experiments/oracle_gap.py --quick         # smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.mapc_baselines import CoTDMA, GenieOracle, LocalGreedy, RandomScheduler
from src.agents.mappo import MAPPO, train_mappo
from src.envs.qh_obss_env_mod1 import OBSSEnv
from src.envs.topology import load_fixed_topologies

POLICIES = ["Random", "Local Greedy (0-bit)", "Co-TDMA (RR)", "MAPPO (0-bit)", "Genie Oracle"]


def make_env(topo, args, load: float) -> OBSSEnv:
    return OBSSEnv(topology=topo, mode="qh", max_episodes=args.episode_len, k_stas=args.k_stas,
                   poisson_lambda=load, allow_mute=True, harsh_start_step=args.episode_len // 2)


def run_episode(env: OBSSEnv, act, seed: int) -> tuple[float, float]:
    """Returns (system throughput, arrival bound = initial backlog + all arrivals)."""
    obs, _ = env.reset(seed=seed)
    bound = float(env.queues().sum())
    total, done = 0.0, False
    while not done:
        bound += float(env.peek_arrivals().sum())
        obs, _, done, _, info = env.step(act(obs))
        total += info["system_throughput"]
    return total, bound


def run_job(args, job: tuple[float, str, int]) -> list[dict]:
    load, name, seed = job
    torch.set_num_threads(1)
    topo = load_fixed_topologies()[name]
    env = make_env(topo, args, load)

    learner = MAPPO(env.observation_space.shape[-1], env.n_aps, env.k_stas, env.valid_sta_counts, seed=seed)
    start = time.time()
    curve = train_mappo(make_env(topo, args, load), learner, args.mappo_episodes,
                        reward_scale=1.0 / (10.0 * env.n_aps), seed=seed)
    train_s = time.time() - start

    policies = {
        "Random": RandomScheduler(env, seed).act,
        "Local Greedy (0-bit)": LocalGreedy(env).act,
        "Co-TDMA (RR)": CoTDMA(env).act,
        "MAPPO (0-bit)": lambda o: learner.act(o, deterministic=True)[0],
        "Genie Oracle": GenieOracle(env).act,
    }
    rows = []
    for policy, act in policies.items():
        thp, bound = run_episode(env, act, 10_000 + seed)
        rows.append({"load": load, "topology": topo.name, "seed": seed, "policy": policy,
                     "throughput": thp, "arrival_bound": bound})
    print(f"[load={load} {topo.name} seed={seed}] "
          + " | ".join(f"{r['policy']}: {r['throughput']:.0f}" for r in rows)
          + f" | bound {bound:.0f} | MAPPO train {train_s:.0f}s, last-10 thp {np.mean(curve[-10:]):.0f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topos", nargs="+", default=[f"T{i}" for i in range(1, 7)])
    parser.add_argument("--loads", nargs="+", type=float, default=[0.1, 1.0],
                        help="Poisson arrivals per STA per slot (0.1 = legacy light load)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--k-stas", type=int, default=4)
    parser.add_argument("--mappo-episodes", type=int, default=300)
    parser.add_argument("--out", default="experiments/oracle_gap_results")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.topos, args.loads, args.seeds, args.episode_len, args.mappo_episodes = ["T3", "T6"], [1.0], 1, 50, 5

    os.makedirs(args.out, exist_ok=True)
    jobs = [(load, name, seed) for load in args.loads for name in args.topos for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, [args] * len(jobs), jobs))
    rows = [row for job_rows in results for row in job_rows]

    with open(os.path.join(args.out, "oracle_gap.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, args)
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print_table(summary)
    plot(summary, os.path.join(args.out, "oracle_gap.png"))


def summarize(rows: list[dict], args) -> dict:
    summary = {"config": {k: v for k, v in vars(args).items() if k != "out"}, "results": {}}
    for load in sorted({r["load"] for r in rows}):
        per_load = {}
        for topo in dict.fromkeys(r["topology"] for r in rows):
            sel = [r for r in rows if r["load"] == load and r["topology"] == topo]
            oracle = np.array([r["throughput"] for r in sel if r["policy"] == "Genie Oracle"])
            entry = {"arrival_bound": float(np.mean([r["arrival_bound"] for r in sel]))}
            for policy in POLICIES:
                thp = np.array([r["throughput"] for r in sel if r["policy"] == policy])
                ratio = thp / np.maximum(oracle, 1e-9)
                entry[policy] = {"mean": float(thp.mean()), "std": float(thp.std()),
                                 "ratio_to_oracle": float(ratio.mean()), "ratio_std": float(ratio.std())}
            per_load[topo] = entry
        summary["results"][str(load)] = per_load
    return summary


def print_table(summary: dict) -> None:
    for load, per_load in summary["results"].items():
        print(f"\n== load λ={load} (throughput, % of Genie Oracle) ==")
        print(f"{'topology':10s} " + " ".join(f"{p[:14]:>16s}" for p in POLICIES) + f" {'arrival bnd':>12s}")
        for topo, entry in per_load.items():
            cells = [f"{entry[p]['mean']:7.0f} ({100 * entry[p]['ratio_to_oracle']:4.0f}%)" for p in POLICIES]
            print(f"{topo:10s} " + " ".join(f"{c:>16s}" for c in cells) + f" {entry['arrival_bound']:12.0f}")


def plot(summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference categorical slots in fixed order; the oracle is the 100% reference line.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    learners = POLICIES[:-1]
    loads = list(summary["results"])
    fig, axes = plt.subplots(len(loads), 1, figsize=(10, 3.4 * len(loads)), squeeze=False)
    for ax, load in zip(axes[:, 0], loads):
        per_load = summary["results"][load]
        topos = list(per_load)
        x = np.arange(len(topos))
        width = 0.8 / len(learners)
        for k, policy in enumerate(learners):
            vals = [100 * per_load[t][policy]["ratio_to_oracle"] for t in topos]
            errs = [100 * per_load[t][policy]["ratio_std"] for t in topos]
            ax.bar(x + (k - (len(learners) - 1) / 2) * width, vals, width * 0.92, yerr=errs, color=colors[k],
                   label=policy, error_kw={"elinewidth": 0.8, "ecolor": "#52514e", "capsize": 2})
        ax.axhline(100, color="#0b0b0b", linewidth=1.2, linestyle="--", label="Genie Oracle (100%)")
        ax.set_xticks(x, topos, color="#0b0b0b")
        ax.set_ylabel("Throughput (% of oracle)", color="#52514e")
        ax.set_title(f"Poisson load λ = {load} packets / STA / slot", loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(ncol=5, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
