"""
experiments/oracle_gap.py
Oracle gap on env v2 (qh_v2): every baseline and learner as a share of the exact per-slot oracle.

Baselines come from src/training/baseline_policies.py (random, greedy/round_robin with ARF,
the all-transmit SINR heuristic "oracle", co_tdma, exact_oracle). Learners are 0-bit MAPPO
(local actors) and, optionally, CentralPPO (∞-bit, autoregressive joint actor). The env runs
with allow_mute=True so learners can stay silent, and with a fixed Poisson rate per STA
(poisson_lambda_min = poisson_lambda_max = load) instead of traffic_load_ratio.

    python experiments/oracle_gap.py                       # default sweep
    python experiments/oracle_gap.py --learners mappo central --loads 5.0 --topos T3 T6
    python experiments/oracle_gap.py --mcs arf genie        # learners pick STA/mute only; MCS by ARF / genie
    python experiments/oracle_gap.py --quick               # smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.mappo import MAPPO, CentralPPO, train_mappo
from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.env_factory import make_env
from src.envs.mcs_control import MCS_CONTROL
from src.envs.topology_registry import get_registry
from src.training.baseline_policies import BaselinePolicy

BASELINES = ["random", "greedy", "round_robin", "oracle", "co_tdma", "exact_oracle"]
LEARNERS = {"mappo": ("MAPPO (0-bit)", MAPPO), "central": ("CentralPPO (∞-bit)", CentralPPO)}
REFERENCE = "exact_oracle"
LABELS = {
    "random": "Random", "greedy": "Greedy (ARF)", "round_robin": "Round-robin (ARF)",
    "oracle": "All-transmit SINR heuristic", "co_tdma": "Co-TDMA", "exact_oracle": "Exact oracle",
}
MCS_LABELS = {"learned": "", "arf": " + ARF MCS", "genie": " + genie MCS"}


def label(policy: str) -> str:
    """'mappo+arf/sampled' -> 'MAPPO (0-bit) + ARF MCS, sampled'."""
    if policy in LABELS:
        return LABELS[policy]
    learner, mode = policy.split("/")
    key, mcs = learner.split("+")
    return f"{LEARNERS[key][0]}{MCS_LABELS[mcs]}, {mode}"


def make_oracle_env(args, topo, load: float):
    experiment = ExperimentConfig.from_dict(load_config(args.config))
    traffic = dict(experiment.env.traffic, traffic_load_ratio=0.0,
                   poisson_lambda_min=load, poisson_lambda_max=load, init_fill_ratio=0.0)
    env_cfg = replace(experiment.env, impl="qh_v2", max_episodes=args.episode_len,
                      traffic=traffic, allow_mute=True, fading={"model": args.fading})
    return make_env(env_cfg, topology=topo, topology_file=experiment.topology_file)


def run_episode(env, act, observe, seed: int) -> tuple[float, float]:
    """Returns (system throughput, arrival bound = initial backlog + all arrivals)."""
    obs, _ = env.reset(seed=seed)
    bound = float(env.queues().sum())
    total, done = 0.0, False
    while not done:
        bound += float(env.peek_arrivals().sum())
        obs, _, done, _, info = env.step(act(obs))
        if observe:
            observe(info)
        total += info["system_throughput"]
    return total, bound


def job_name(job: tuple) -> str:
    return "_".join(str(part) for part in job)


def run_job(args, job: tuple) -> list[dict]:
    """job = ("baselines", load, topo, seed) or ("learner", load, topo, seed, learner, mcs).

    Each finished job is written to <out>/jobs/<name>.json; a re-run returns that file instead of
    recomputing, so an interrupted sweep resumes where it stopped.
    """
    path = os.path.join(args.out, "jobs", job_name(job) + ".json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)["rows"]

    kind, load, topo_id, seed = job[:4]
    torch.set_num_threads(1)
    topo = get_registry().get(topo_id)
    env = make_oracle_env(args, topo, load)
    eval_seed = 10_000 + seed
    rows, extra = [], {}

    def record(policy, thp, bound):
        rows.append({"load": load, "topology": topo_id, "seed": seed, "policy": policy,
                     "throughput": thp, "arrival_bound": bound})

    if kind == "baselines":
        for name in BASELINES:
            policy = BaselinePolicy(name, seed=eval_seed)
            record(name, *run_episode(env, lambda o, p=policy: p.act(env), policy.observe, eval_seed))
    else:
        key, mcs = job[4:]
        _, cls = LEARNERS[key]
        wrap = MCS_CONTROL[mcs] or (lambda e: e)
        # With link adaptation the learner only picks STA / mute: a single dummy MCS output.
        n_mcs = env.mcs_table.n_mcs if mcs == "learned" else 1
        learner = cls(env.obs_dim, env.n_aps, env.k_stas, env.valid_sta_counts, n_mcs=n_mcs, seed=seed)
        start = time.time()
        curve = train_mappo(wrap(make_oracle_env(args, topo, load)), learner, args.episodes,
                            reward_scale=1.0 / (10.0 * env.n_aps), seed=seed,
                            episodes_per_update=args.episodes_per_update)
        extra = {"train_seconds": time.time() - start, "curve": curve}
        eval_env = wrap(env)
        for mode in ("argmax", "sampled"):
            torch.manual_seed(eval_seed)
            act = lambda o, d=(mode == "argmax"): learner.act(o, deterministic=d)[0]
            record(f"{key}+{mcs}/{mode}", *run_episode(eval_env, act, None, eval_seed))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump({"job": list(job), "rows": rows, **extra}, handle)
    os.replace(path + ".tmp", path)
    print(f"[{job_name(job)}] " + " | ".join(f"{r['policy']}: {r['throughput']:.0f}" for r in rows)
          + (f" | train {extra['train_seconds']:.0f}s" if extra else ""), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topos", nargs="+", default=[f"T{i}" for i in range(1, 7)])
    parser.add_argument("--loads", nargs="+", type=float, default=[1.0, 5.0],
                        help="Poisson arrivals per STA per slot")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--learners", nargs="*", default=["mappo"], choices=list(LEARNERS))
    parser.add_argument("--mcs", nargs="+", default=["learned"], choices=list(MCS_CONTROL),
                        help="MCS control for learners: learned (action head), arf, genie")
    parser.add_argument("--fading", default="none", choices=["none", "rayleigh"],
                        help="small-scale fading: none (deterministic path loss) or per-slot Rayleigh block fading")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--out", default="experiments/oracle_gap_results_envv2")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.topos, args.loads, args.seeds, args.episode_len, args.episodes = ["T3", "T6"], [1.0], 1, 50, 8

    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for load in args.loads:
        for t in args.topos:
            for seed in range(args.seeds):
                jobs.append(("baselines", load, t, seed))
                jobs += [("learner", load, t, seed, key, mcs) for key in args.learners for mcs in args.mcs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for job_rows in pool.map(run_job, [args] * len(jobs), jobs) for row in job_rows]

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
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    summary = {"config": {k: v for k, v in vars(args).items() if k != "out"},
               "policies": policies, "results": {}}
    for load in sorted({r["load"] for r in rows}):
        per_load = {}
        for topo in dict.fromkeys(r["topology"] for r in rows):
            sel = [r for r in rows if r["load"] == load and r["topology"] == topo]
            ref = np.array([r["throughput"] for r in sel if r["policy"] == REFERENCE])
            entry = {"arrival_bound": float(np.mean([r["arrival_bound"] for r in sel]))}
            for policy in policies:
                thp = np.array([r["throughput"] for r in sel if r["policy"] == policy])
                ratio = thp / np.maximum(ref, 1e-9)
                entry[policy] = {"mean": float(thp.mean()), "std": float(thp.std()),
                                 "ratio_to_oracle": float(ratio.mean()), "ratio_std": float(ratio.std())}
            per_load[topo] = entry
        summary["results"][str(load)] = per_load
    return summary


def print_table(summary: dict) -> None:
    policies = summary["policies"]
    for load, per_load in summary["results"].items():
        print(f"\n== load λ={load} (throughput, % of exact oracle) ==")
        print(f"{'topology':8s} " + " ".join(f"{p[:15]:>15s}" for p in policies) + f" {'arrivals':>9s}")
        for topo, entry in per_load.items():
            cells = [f"{entry[p]['mean']:6.0f} ({100 * entry[p]['ratio_to_oracle']:3.0f}%)" for p in policies]
            print(f"{topo:8s} " + " ".join(f"{c:>15s}" for c in cells) + f" {entry['arrival_bound']:9.0f}")


def plot(summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference categorical slots in fixed order, keyed by policy (never by rank).
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    order = [p for p in summary["policies"] if p != REFERENCE]
    colors = {p: palette[i % len(palette)] for i, p in enumerate(order)}
    loads = list(summary["results"])
    fig, axes = plt.subplots(len(loads), 1, figsize=(12, 3.6 * len(loads)), squeeze=False)
    for ax, load in zip(axes[:, 0], loads):
        per_load = summary["results"][load]
        topos = list(per_load)
        x = np.arange(len(topos))
        width = 0.84 / len(order)
        for k, policy in enumerate(order):
            vals = [100 * per_load[t][policy]["ratio_to_oracle"] for t in topos]
            errs = [100 * per_load[t][policy]["ratio_std"] for t in topos]
            ax.bar(x + (k - (len(order) - 1) / 2) * width, vals, width * 0.9, yerr=errs, color=colors[policy],
                   label=label(policy), error_kw={"elinewidth": 0.8, "ecolor": "#52514e", "capsize": 2})
        ax.axhline(100, color="#0b0b0b", linewidth=1.2, linestyle="--", label="Exact oracle (100%)")
        ax.set_xticks(x, topos, color="#0b0b0b")
        ax.set_ylabel("Throughput (% of oracle)", color="#52514e")
        ax.set_title(f"env v2, Poisson load λ = {load} packets / STA / slot", loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.36))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
