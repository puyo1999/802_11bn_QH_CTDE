"""
experiments/cellular_learn.py
Learned schedulers of paper3 (E2-E5): training of MAPPO (0-bit, shared actor) and Central PPO (full information)
on one drop of experiments/cellular_gap.py, and evaluation over a full episode (300 warm-up + 1000 measured slots)
against the oracle and the references with the metrics of LA0/LA1.

    python experiments/cellular_learn.py pilot --drop 0 --episodes 300 --train-len 400

Training episodes are shorter than the evaluation episode and start from the PF averages of
CellularSim.initial_avg_rate, which are near their steady state; the training seeds differ from the evaluation
seed, so the learner is evaluated on fading and arrival sequences that it has not seen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cellular_gap import Meter, decide, run_jobs                     # noqa: E402
from src.agents.mappo import CentralPPO, SharedCentralPPO, SharedMAPPO, train_mappo   # noqa: E402
from src.cellular.linkadapt import LinkAdapter                       # noqa: E402
from src.cellular.marl_env import CellularMARLEnv                    # noqa: E402
from src.cellular.sim import CellularConfig, CellularSim             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cellular_results", "learn_jobs")
LEARNERS = {"mappo": SharedMAPPO, "central": CentralPPO, "central_shared": SharedCentralPPO}
EVAL_SEED = 0          # the evaluation seed of LA0/LA1, so that references and learners see the same sequences


def evaluate_learner(cfg: CellularConfig, drop: int, learner, cqi_mode: str, deterministic: bool) -> dict:
    env = CellularMARLEnv(cfg, drop, cqi_mode)
    obs, _ = env.reset(seed=EVAL_SEED)
    meter, done = Meter(env.sim), False
    torch.manual_seed(EVAL_SEED)
    while not done:
        measuring, arrivals = env.sim.measuring, env.sim.arrivals.copy()
        action = learner.act(obs, deterministic=deterministic)[0]
        obs, _, done, _, info = env.step(action)
        if measuring:
            meter.record(arrivals, info["targets"], info["cqis"], info["out"])
    return meter.result()


def evaluate_reference(cfg: CellularConfig, drop: int, policy: str, cqi_mode: str) -> dict:
    sim = CellularSim(cfg, drop_seed=drop)
    sim.reset(seed=EVAL_SEED)
    adapter = LinkAdapter(sim, "genie" if cqi_mode == "ideal" else "local")
    meter = Meter(sim)
    while not sim.done:
        measuring, arrivals = sim.measuring, sim.arrivals.copy()
        targets = decide(sim, policy, sim.t)[0]
        cqis = adapter.select(targets)
        out = sim.step(targets, cqis)
        adapter.update(targets, cqis, out["success"])
        if measuring:
            meter.record(arrivals, targets, cqis, out)
    return meter.result()


def learn_job(job: tuple) -> dict:
    """job = (drop, delay, speed, learner, cqi_mode, episodes, train_len, seed[, gamma]); saved to learn_jobs/.

    gamma is the discount of PPO (default 0.99 of the companion study); gamma = 0 makes the learner maximize the
    expected slot utility, i.e., the objective of the per-slot oracle."""
    drop, delay, speed, key, cqi_mode, episodes, train_len, seed = job[:8]
    gamma = job[8] if len(job) > 8 else 0.99
    name = f"d{drop}_delay{delay}_v{speed:g}_{key}_{cqi_mode}_E{episodes}_L{train_len}_s{seed}"
    if gamma != 0.99:
        name += f"_g{gamma:g}"
    path = os.path.join(OUT, name + ".json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    torch.set_num_threads(1)
    cfg = CellularConfig(cluster_size=3, ues_per_cell=4, n_subbands=4, objective="pf", cqi_delay=delay,
                         speed_kmh=speed)
    train_cfg = replace(cfg, episode_len=train_len, warmup_slots=0)
    env = CellularMARLEnv(train_cfg, drop, cqi_mode)
    learner = LEARNERS[key](env.obs_dim, env.n_aps, env.k_stas, env.valid_sta_counts, n_mcs=env.n_mcs, seed=seed,
                          gamma=gamma)
    start = time.time()
    curve = train_mappo(env, learner, episodes, reward_scale=1.0, seed=seed + 1, reward_fn="env")
    train_s = time.time() - start
    result = {"job": list(job), "train_seconds": train_s, "curve": curve,
              "sampled": evaluate_learner(cfg, drop, learner, cqi_mode, deterministic=False),
              "argmax": evaluate_learner(cfg, drop, learner, cqi_mode, deterministic=True)}
    os.makedirs(OUT, exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(result, handle)
    os.replace(path + ".tmp", path)
    return result


def learn_rows(job: tuple) -> list[dict]:
    return [learn_job(job)]


def pilot(args) -> None:
    """One drop: references and both learners with ideal CQI, relative to the oracle with genie CQI."""
    cfg = CellularConfig(cluster_size=3, ues_per_cell=4, n_subbands=4, objective="pf", cqi_delay=args.delay,
                         speed_kmh=args.speed)
    refs = {p: evaluate_reference(cfg, args.drop, p, "ideal") for p in ("oracle", "full_reuse", "greedy_muting")}
    jobs = [(args.drop, args.delay, args.speed, key, "ideal", args.episodes, args.train_len, 0, args.gamma)
            for key in args.learners]
    if args.workers > 1:
        run_jobs(learn_rows, jobs, args.workers)
    results = [learn_job(j) for j in jobs]          # read back from learn_jobs/ in job order
    base = refs["oracle"]

    def rel(r):
        return 100 * np.exp((r["log_utility"] - base["log_utility"]) / r["n_ues"])

    print(f"\n== pilot drop {args.drop}, d={args.delay}, v={args.speed:g} km/h, ideal CQI: geometric-mean rate / "
          f"oracle (%)")
    for p, r in refs.items():
        print(f"{p:>16}: {rel(r):6.1f}")
    summary = {"refs": {p: rel(r) for p, r in refs.items()}}
    for job, res in zip(jobs, results):
        curve = np.array(res["curve"])
        k = max(len(curve) // 10, 1)
        print(f"{job[3] + f' g={job[8]:g}':>16}: sampled {rel(res['sampled']):6.1f}, argmax {rel(res['argmax']):6.1f}  "
              f"(train {res['train_seconds'] / 60:.0f} min; slot throughput first/last 10%: "
              f"{curve[:k].mean() / args.train_len:.3f} / {curve[-k:].mean() / args.train_len:.3f})")
        summary[f"{job[3]}_g{job[8]:g}"] = {"sampled": rel(res["sampled"]), "argmax": rel(res["argmax"]),
                           "train_seconds": res["train_seconds"]}
    with open(os.path.join(os.path.dirname(OUT), f"learn_pilot_d{args.drop}_g{args.gamma:g}.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


REFERENCES = ("oracle", "full_reuse", "greedy_muting", "causal_full_reuse", "causal_greedy")


def e2_refs(job: tuple) -> list[dict]:
    """job = (drop, delay, speed): references with genie and with local (OLLA) CQI, log utility per policy."""
    drop, delay, speed = job
    path = os.path.join(os.path.dirname(OUT), "e2_refs", f"d{drop}_delay{delay}_v{speed:g}.json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    cfg = CellularConfig(cluster_size=3, ues_per_cell=4, n_subbands=4, objective="pf", cqi_delay=delay,
                         speed_kmh=speed)
    rows = []
    for cqi_mode in ("ideal", "olla"):
        for p in REFERENCES:
            r = evaluate_reference(cfg, drop, p, cqi_mode)
            rows.append({"drop": drop, "policy": p, "cqi_mode": cqi_mode, "log_utility": r["log_utility"],
                         "n_ues": r["n_ues"]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(rows, handle)
    return rows


def e3_unit(job: tuple) -> list[dict]:
    """A reference job ("refs", drop, delay, speed) or a learner job of learn_job, for one process pool."""
    if job[0] == "refs":
        return e2_refs(job[1:])
    return learn_rows(job)


E3_CONDITIONS = [(d, v) for d in (0, 4, 8) for v in (3.0, 30.0, 60.0)]


def e3(args) -> None:
    """E3 (reduced): E2 for every CQI delay and UE speed of LA0 on drops 0..D-1; one E2 summary per condition."""
    drops = list(range(args.drops))
    jobs = [("refs", d, delay, speed) for delay, speed in E3_CONDITIONS for d in drops]
    # learner jobs first by cost: the pool then ends with short jobs
    jobs = [(d, delay, speed, key, mode, args.episodes, args.train_len, 0, args.gamma)
            for key in args.learners for delay, speed in E3_CONDITIONS for d in drops for mode in args.cqi_modes] + jobs
    run_jobs(e3_unit, jobs, args.workers)
    for delay, speed in E3_CONDITIONS:
        sub = argparse.Namespace(**{**vars(args), "delay": delay, "speed": speed, "workers": 1})
        e2(sub)


def e2(args) -> None:
    """Reduced E2: learners on drops 0..D-1 with ideal, OLLA and learned CQI, against the oracle with genie CQI
    and the references (current-channel and causal) with genie and OLLA CQI. The learned-CQI learner is compared
    with the references under OLLA, which also choose the CQI from the delayed reports."""
    drops = list(range(args.drops))
    jobs = [(d, args.delay, args.speed, key, mode, args.episodes, args.train_len, 0, args.gamma)
            for d in drops for key in args.learners for mode in args.cqi_modes]
    if args.workers > 1:
        run_jobs(learn_rows, jobs, args.workers)
    results = {j: learn_job(j) for j in jobs}
    refs = [row for d in drops for row in e2_refs((d, args.delay, args.speed))]
    oracle = {r["drop"]: r["log_utility"] for r in refs if r["policy"] == "oracle" and r["cqi_mode"] == "ideal"}

    def rel(d, r):
        return float(100 * np.exp((r["log_utility"] - oracle[d]) / r["n_ues"]))

    table = {}
    for r in refs:
        table.setdefault(f"{r['policy']}|{r['cqi_mode']}", {})[r["drop"]] = rel(r["drop"], r)
    for j, res in results.items():
        for how in ("sampled", "argmax"):
            table.setdefault(f"{j[3]}|{j[4]}|{how}", {})[j[0]] = rel(j[0], res[how])
    print(f"\n== E2 (reduced): {len(drops)} drops, d={args.delay}, v={args.speed:g} km/h, {args.episodes} episodes, "
          f"gamma={args.gamma:g}: geometric-mean rate / oracle with genie CQI (%), mean and per drop")
    for k, v in table.items():
        x = np.array([v[d] for d in drops])
        print(f"{k:>34}: {x.mean():6.1f}  [" + " ".join(f"{y:5.1f}" for y in x) + "]")
    name = f"e2_D{len(drops)}_delay{args.delay}_v{args.speed:g}_E{args.episodes}_g{args.gamma:g}.json"
    with open(os.path.join(os.path.dirname(OUT), name), "w") as handle:
        json.dump({"args": vars(args), "table": table,
                   "train_seconds": {"|".join(map(str, j)): r["train_seconds"] for j, r in results.items()}},
                  handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", choices=["pilot", "e2", "e3"])
    parser.add_argument("--drop", type=int, default=0)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-len", type=int, default=400)
    parser.add_argument("--learners", nargs="+", default=["mappo", "central"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--drops", type=int, default=5)
    parser.add_argument("--cqi-modes", nargs="+", default=["ideal", "olla", "learned"])
    args = parser.parse_args()
    {"pilot": pilot, "e2": e2, "e3": e3}[args.exp](args)


if __name__ == "__main__":
    main()
