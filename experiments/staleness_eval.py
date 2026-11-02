"""
experiments/staleness_eval.py
Policy staleness under mobility: a learner trained on one topology, evaluated after the STAs move.

Each job trains one learner on the base topology (as in oracle_gap.py), freezes it, and evaluates it on
copies of the topology in which every STA is displaced by `shift` meters in a uniformly random direction.
Associations and overlap sets stay as in the base topology, i.e., the STAs have not roamed yet. Co-TDMA
and the exact oracle are evaluated on the same displaced topology and the same arrivals; both recompute
their decisions from the current state, so the learner's ratio to the oracle isolates the staleness.

    python experiments/staleness_eval.py                                   # default: T3 T4 T6, lambda = 8
    python experiments/staleness_eval.py --quick                           # smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_gap import LEARNERS, make_oracle_env, run_episode  # noqa: E402
from src.agents.mappo import train_mappo  # noqa: E402
from src.envs.mcs_control import MCS_CONTROL  # noqa: E402
from src.envs.topology_registry import get_registry  # noqa: E402
from src.training.baseline_policies import BaselinePolicy  # noqa: E402

BASELINES = ["co_tdma", "exact_oracle"]


def displace(topo, shift: float, rng: np.random.Generator):
    """Copy of `topo` with every STA moved by `shift` m in a random direction; association unchanged."""
    moved = []
    for stas in topo.sta_positions:
        row = []
        for pos in stas:
            theta = rng.uniform(0.0, 2.0 * np.pi)
            row.append(np.asarray(pos, dtype=float) + shift * np.array([np.cos(theta), np.sin(theta)]))
        moved.append(row)
    return replace(topo, sta_positions=moved, name=f"{topo.name}+{shift:g}m")


def misassociated_share(topo) -> float:
    """Share of STAs whose nearest AP is not their serving AP."""
    aps = np.asarray(topo.ap_positions, dtype=float)
    wrong = [int(np.argmin(np.linalg.norm(aps - np.asarray(p), axis=1)) != i)
             for i, stas in enumerate(topo.sta_positions) for p in stas]
    return float(np.mean(wrong)) if wrong else 0.0


def job_name(job: tuple) -> str:
    return "_".join(str(part) for part in job)


def run_job(args, job: tuple) -> list[dict]:
    """job = (load, topo, seed, learner, mcs). Finished jobs are cached in <out>/jobs/<name>.json."""
    path = os.path.join(args.out, "jobs", job_name(job) + ".json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)["rows"]

    load, topo_id, seed, key, mcs = job
    torch.set_num_threads(1)
    base = get_registry(args.topology_file).get(topo_id)
    env = make_oracle_env(args, base, load)
    wrap = MCS_CONTROL[mcs] or (lambda e: e)
    n_mcs = env.mcs_table.n_mcs if mcs == "learned" else 1
    learner = LEARNERS[key][1](env.obs_dim, env.n_aps, env.k_stas, env.valid_sta_counts, n_mcs=n_mcs, seed=seed)
    start = time.time()
    train_mappo(wrap(make_oracle_env(args, base, load)), learner, args.episodes,
                reward_scale=1.0 / (10.0 * env.n_aps), seed=seed, episodes_per_update=args.episodes_per_update)
    train_seconds = time.time() - start

    rows = []
    for shift in args.shifts:
        for draw in range(1 if shift == 0 else args.draws):
            rng = np.random.default_rng((seed, draw, int(round(100 * shift))))
            topo = base if shift == 0 else displace(base, shift, rng)
            eval_env = make_oracle_env(args, topo, load)
            eval_seed = 10_000 + 100 * seed + draw
            row = {"load": load, "topology": topo_id, "seed": seed, "learner": f"{key}+{mcs}",
                   "shift_m": shift, "draw": draw, "misassociated": misassociated_share(topo)}
            for name in BASELINES:
                policy = BaselinePolicy(name, seed=eval_seed)
                row[name] = run_episode(eval_env, lambda o, p=policy: p.act(eval_env), policy.observe, eval_seed)[0]
            wrapped = wrap(eval_env)
            for mode in ("argmax", "sampled"):
                torch.manual_seed(eval_seed)
                act = lambda o, d=(mode == "argmax"): learner.act(o, deterministic=d)[0]
                row[mode] = run_episode(wrapped, act, None, eval_seed)[0]
            rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump({"job": list(job), "rows": rows, "train_seconds": train_seconds}, handle)
    os.replace(path + ".tmp", path)
    print(f"[{job_name(job)}] train {train_seconds:.0f}s | " + " | ".join(
        f"{r['shift_m']:g}m: {r['sampled'] / r['exact_oracle']:.0%}" for r in rows if r["draw"] == 0), flush=True)
    return rows


def summarize(rows: list[dict]) -> None:
    """Mean ratio to the oracle per (topology, learner, shift), over seeds and draws."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["load"], r["topology"], r["learner"], r["shift_m"])].append(r)
    print(f"\n{'load':>5} {'topo':<5} {'learner':<15} {'shift':>6} {'mis-assoc':>9} {'Co-TDMA':>8}"
          f" {'sampled':>8} {'argmax':>8}   (% of exact oracle on the displaced topology)")
    for (load, topo, learner, shift), rs in sorted(groups.items()):
        ratio = lambda k: 100 * np.mean([r[k] / r["exact_oracle"] for r in rs])
        print(f"{load:>5} {topo:<5} {learner:<15} {shift:>5g}m {100 * np.mean([r['misassociated'] for r in rs]):>8.0f}%"
              f" {ratio('co_tdma'):>7.1f}% {ratio('sampled'):>7.1f}% {ratio('argmax'):>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topology-file", default="configs/topologies.yaml")
    parser.add_argument("--topos", nargs="+", default=["T3", "T4", "T6"])
    parser.add_argument("--loads", nargs="+", type=float, default=[8.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--learners", nargs="+", default=["mappo", "central"], choices=list(LEARNERS))
    parser.add_argument("--mcs", nargs="+", default=["genie"], choices=list(MCS_CONTROL))
    parser.add_argument("--shifts", nargs="+", type=float, default=[0.0, 1.0, 3.0, 5.0], help="STA displacement (m)")
    parser.add_argument("--draws", type=int, default=5, help="random displacements per shift")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--fading", default="none", choices=["none", "rayleigh", "gauss_markov"])
    parser.add_argument("--fading-rho", type=float, default=0.0)
    parser.add_argument("--out", default="experiments/staleness_results")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.topos, args.seeds, args.learners, args.episodes, args.episode_len, args.draws = \
            ["T3"], 1, ["mappo"], 8, 50, 2

    jobs = [(load, t, seed, key, mcs) for load in args.loads for t in args.topos for seed in range(args.seeds)
            for key in args.learners for mcs in args.mcs]
    os.makedirs(args.out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for job_rows in pool.map(run_job, [args] * len(jobs), jobs) for row in job_rows]
    with open(os.path.join(args.out, "staleness.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
