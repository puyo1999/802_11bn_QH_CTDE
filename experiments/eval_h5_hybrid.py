"""H5 harsh-regime check: learned STA/defer decision combined with ARF link adaptation.

No retraining. Each trained checkpoint is evaluated on the same episodes as
eval_policy_comparison.py (eval seed = seed + 10000) in three MCS modes:

  learned : the checkpoint as trained (MCS from the policy) -- reproduces the reference run
  arf     : STA/defer from the policy, MCS from a per-AP ARF state machine
  cap     : STA/defer from the policy, MCS = min(policy MCS, ARF MCS)

ARF is the same state machine as the greedy/round-robin baselines
(src/training/baseline_policies.py: ARF_UP, ARF_DOWN, interference-free start MCS).

Outputs (in --output): h5_summary.csv, one row per (topology, condition, mode, seed).

Usage:
  python experiments/eval_h5_hybrid.py --topologies T1 T2 T3 T4 T6 --conditions C1 C2 C3 C4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.topology_registry import load_topologies
from src.training.baseline_policies import ARF_DOWN, ARF_UP, _interference_free_mcs
from src.training.centralized_execution import CentralizedExecutor
from src.training.oracle_policy import prospective_harsh
from experiments.eval_policy_comparison import EVAL_SEED_OFFSET, find_checkpoint
from experiments.train_ablation import CONDITIONS, build_trainer, new_env

MODES = ("learned", "arf", "cap")


class ARFState:
    """Per-AP ARF state machine, identical to BaselinePolicy's ARF."""

    def __init__(self, n_aps: int, max_mcs: int):
        self.max_mcs = max_mcs
        self.mcs = None
        self.n_ok = [0] * n_aps
        self.n_fail = [0] * n_aps
        self.probing = [False] * n_aps

    def update(self, attempted, thp):
        for i, tried in enumerate(attempted):
            if not tried:
                continue
            if float(thp[i]) > 0:
                self.n_ok[i] += 1
                self.n_fail[i] = 0
                self.probing[i] = False
                if self.n_ok[i] >= ARF_UP:
                    self.mcs[i] = min(self.mcs[i] + 1, self.max_mcs)
                    self.n_ok[i] = 0
                    self.probing[i] = True
            else:
                self.n_fail[i] += 1
                self.n_ok[i] = 0
                if self.probing[i] or self.n_fail[i] >= ARF_DOWN:
                    self.mcs[i] = max(self.mcs[i] - 1, 0)
                    self.n_fail[i] = 0
                self.probing[i] = False


def decodable_bound(env, ap, sta, transmitting, harsh):
    """Highest decodable MCS of (ap, sta) for the given concurrent transmitters; -1 if none or blocked."""
    saved = (env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count)
    try:
        r = env.interference.compute(ap, sta, transmitting, is_harsh=harsh)
    finally:
        env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count = saved
    if r.blocked_by_overlap or r.sinr_db < env.mcs_table.required_sinr_db(0):
        return -1
    return env.mcs_table.best_mcs_for(r.sinr_db)


def run(topo, cfg, exp, cond, seed, ckpt, mode, steps, trace=None):
    eval_seed = seed + EVAL_SEED_OFFSET
    env = new_env(topo, cfg, steps, seed_offset=eval_seed)
    model = build_trainer(env, cfg, cond).model
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    counts = [len(s) for s in topo.sta_positions]
    ex = CentralizedExecutor(model, counts, threshold=exp.agent.gamma_margin, device="cpu",
                             execution_mode=CONDITIONS[cond]["execution_mode"])
    arf = ARFState(env.n_aps, env.mcs_table.n_mcs - 1)
    obs, _ = env.reset(seed=eval_seed)
    tot = dict(normal=0.0, harsh=0.0)
    tx = dict(normal=0, harsh=0)
    ok = dict(normal=0, harsh=0)
    for t in range(steps):
        harsh = prospective_harsh(env)
        with torch.no_grad():
            a = np.asarray(ex.act(obs, deterministic=True).actions, dtype=np.int64).reshape(-1).copy()
        attempted = []
        for i in range(env.n_aps):
            sta = int(a[2 * i])
            real = sta < counts[i]
            if arf.mcs is None:
                arf.mcs = [0] * env.n_aps
            if t == 0:
                # Start MCS: interference-free MCS of the policy's first STA (or STA 0 on defer).
                arf.mcs[i] = _interference_free_mcs(env, i, sta if real else 0, harsh)
            if mode == "arf":
                a[2 * i + 1] = arf.mcs[i]
            elif mode == "cap":
                a[2 * i + 1] = min(int(a[2 * i + 1]), arf.mcs[i])
            attempted.append(bool(real and env._aps[i].queues[sta] > 0))
        counters = [(ap.pkts_sent, ap.pkts_dropped) for ap in env._aps]
        obs, _, done, _, info = env.step(a.astype(np.int32))
        if trace is not None:
            # Actual transmissions of this step (after arrivals): the env counts every attempt as
            # sent or dropped, so the counter deltas identify the transmitting set exactly.
            sent = [(ap.pkts_sent, ap.pkts_dropped) != c for ap, c in zip(env._aps, counters)]
            tx_set = [i for i in range(env.n_aps) if sent[i]]
            for i in range(env.n_aps):
                bound = decodable_bound(env, i, int(a[2 * i]), tx_set, harsh) if sent[i] else ""
                trace.append(dict(step=t, ap=i, mode=mode, harsh=int(harsh), attempted=int(sent[i]),
                                  mcs=int(a[2 * i + 1]), bound=bound,
                                  success=int(float(info["thp"][i]) > 0)))
        arf.update(attempted, info["thp"])
        reg = "harsh" if harsh else "normal"
        tot[reg] += float(info["system_throughput"])
        tx[reg] += sum(attempted)
        ok[reg] += sum(1 for i, x in enumerate(attempted) if x and float(info["thp"][i]) > 0)
        if done:
            break
    env.close()
    return dict(throughput_packets=tot["normal"] + tot["harsh"],
                throughput_normal=tot["normal"], throughput_harsh=tot["harsh"],
                attempts_harsh=tx["harsh"], success_rate_harsh=ok["harsh"] / max(tx["harsh"], 1))


def write_trace(args, cfg, exp, steps):
    """Per-step trace of the selected MCS and its decodable bound (input of the harsh-MCS figure)."""
    rows = []
    for spec in args.trace:
        tid, cond, seed = spec.split(":")
        topo = load_topologies([tid], cfg.get("topology_file", "configs/topologies.yaml"))[0]
        ckpt = find_checkpoint(args.checkpoints, tid, cond, int(seed))
        if ckpt is None:
            sys.exit(f"no checkpoint for {spec}")
        for mode in args.modes:
            trace = []
            run(topo, cfg, exp, cond, int(seed), ckpt, mode, steps, trace=trace)
            rows += [dict(topology=tid, condition=cond, seed=int(seed), **r) for r in trace]
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "h5_trace.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/qh_ctde.yaml")
    p.add_argument("--topologies", nargs="+", default=["T1", "T2", "T3", "T4", "T5", "T6"])
    p.add_argument("--conditions", nargs="+", default=["C2", "C4"], choices=list(CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--checkpoints", nargs="+", default=["results_csv/main_full", "results_csv/topo_c2c4"])
    p.add_argument("--output", default="results_csv/harsh_tuning")
    p.add_argument("--trace", nargs="+", metavar="TOPO:COND:SEED",
                   help="write per-step, per-AP MCS traces (h5_trace.csv) for these runs instead")
    args = p.parse_args()

    torch.set_num_threads(1)
    cfg = load_config(args.config)
    exp = ExperimentConfig.from_dict(cfg)
    steps = exp.train.eval_steps
    if args.trace:
        write_trace(args, cfg, exp, steps)
        return
    topos = load_topologies(args.topologies, cfg.get("topology_file", "configs/topologies.yaml"))
    os.makedirs(args.output, exist_ok=True)
    rows = []
    for topo in topos:
        tid = topo.metadata.get("id", topo.name)
        for cond in args.conditions:
            for seed in args.seeds:
                ckpt = find_checkpoint(args.checkpoints, tid, cond, seed)
                if ckpt is None:
                    continue
                for mode in args.modes:
                    r = run(topo, cfg, exp, cond, seed, ckpt, mode, steps)
                    rows.append(dict(topology=tid, condition=cond, mode=mode, seed=seed, **r))
                    print(f"{tid} {cond} s{seed} {mode:7s} total={r['throughput_packets']:5.0f} "
                          f"harsh={r['throughput_harsh']:4.0f} ok_harsh={r['success_rate_harsh']:.2f}",
                          flush=True)
    path = os.path.join(args.output, "h5_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


if __name__ == "__main__":
    main()
