"""
experiments/train.py
학습 진입점 — Attention-DRLCA / FC-DRLCA 공통

Usage:
    python experiments/train.py --config configs/drlca_attn.yaml
    python experiments/train.py --config configs/drlca_fc.yaml --mode cca
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import yaml
import torch
from tqdm import tqdm

from src.envs.obss_env import OBSSEnv
from src.envs.topology import get_all_fixed_topos, random_topology
from src.agents.attn_drlca_agent import AttnDRLCAAgent
from src.utils.metrics import aggregate_metrics, d95_reduction
from src.utils.logger import Logger


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # base config 병합 (단순 flat merge)
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("_base_"))
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def build_agents(n_aps: int, cfg: dict, mode: str) -> list[AttnDRLCAAgent]:
    n_actions = 7 if mode == "cw" else 5
    acfg = cfg["agent"]
    return [
        AttnDRLCAAgent(
            ap_id      = i,
            state_dim  = acfg["state_dim"],
            fused_dim  = acfg["fused_dim"],
            n_actions  = n_actions,
            history_len= acfg["history_len"],
            hidden_dim = acfg.get("hidden_dim", 32),
            n_heads    = acfg.get("n_heads", 2),
            lr         = acfg["lr"],
            gamma      = acfg["gamma"],
            tau        = acfg["tau"],
            eps_start  = acfg["eps_start"],
            eps_end    = acfg["eps_end"],
            eps_decay  = acfg["eps_decay"],
            batch_size = acfg["batch_size"],
            mem_size   = acfg["mem_size"],
            device     = cfg["train"]["device"],
            dueling    = acfg.get("dueling", False),
        )
        for i in range(n_aps)
    ]


def train_on_topology(topo, cfg: dict, mode: str, run_id: int,
                      logger: Logger | None = None) -> dict:
    """단일 토폴로지에서 학습 루프"""
    ecfg = cfg["env"]
    env  = OBSSEnv(
        topology       = topo,
        mode           = mode,
        interact_slots = ecfg["interact_slots"],
        max_episodes   = ecfg["max_episodes"],
        seed           = ecfg["seed"] + run_id,
    )
    agents = build_agents(topo.n_aps, cfg, mode)
    info_log = []
    reward_log = []

    obs, _ = env.reset(seed=ecfg["seed"] + run_id)

    for ep in tqdm(range(ecfg["max_episodes"]),
                   desc=f"[{topo.name}] run={run_id}", leave=False):
        # ── 1. 각 AP: fusion → 현재 obs 구성 ──────────────────
        prev_obs_list = []
        for i, agent in enumerate(agents):
            nbr_ids    = topo.neighbors[i]
            nbr_states = [obs[j] for j in nbr_ids]
            agent.fuse(obs[i], nbr_states)
            prev_obs_list.append(agent.get_obs().copy())   # (T, F)

        # ── 2. 행동 선택 ────────────────────────────────────────
        if mode == "cw":
            actions = np.array([a.select_action(prev_obs_list[i])
                                 for i, a in enumerate(agents)], dtype=np.int32)
        else:
            actions = np.array([a.select_action(prev_obs_list[i])
                                 for i, a in enumerate(agents)], dtype=np.int32)

        # ── 3. 환경 step ────────────────────────────────────────
        next_obs, reward, done, _, info = env.step(actions)

        # ── 4. 다음 obs fusion ──────────────────────────────────
        for i, agent in enumerate(agents):
            nbr_ids    = topo.neighbors[i]
            nbr_states = [next_obs[j] for j in nbr_ids]
            agent.fuse(next_obs[i], nbr_states)
            next_obs_i = agent.get_obs().copy()

            # ── 5. 저장 & 학습 ────────────────────────────────
            agent.store(prev_obs_list[i], actions[i], reward, next_obs_i, done)
            agent.learn()
            agent.decay_epsilon()

        obs = next_obs
        info_log.append(info)
        reward_log.append(reward)

        if logger and ep % cfg["train"]["log_interval"] == 0:
            m = aggregate_metrics(info_log[-cfg["train"]["log_interval"]:])
            logger.log({
                f"{topo.name}/d95": m["d95_mean"],
                f"{topo.name}/thp": m["thp_mean"],
                f"{topo.name}/per": m["per_mean"],
                f"{topo.name}/reward": float(np.mean(reward_log[-10:])),
                f"{topo.name}/eps": agents[0].eps,
            }, step=ep + run_id * ecfg["max_episodes"])

        if done:
            obs, _ = env.reset()

    summary = aggregate_metrics(info_log)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/drlca_attn.yaml")
    parser.add_argument("--mode",   default=None,
                        help="override env.mode (cw|cca|joint)")
    parser.add_argument("--random_topos", type=int, default=0,
                        help="추가 랜덤 토폴로지 수")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    mode = args.mode or cfg["env"].get("mode", "cw")
    lcfg = cfg.get("logging", {})
    run_name = lcfg.get("run_name", "run")

    logger = Logger(
        run_name  = run_name,
        use_wandb = lcfg.get("use_wandb", False),
        use_tb    = lcfg.get("use_tb", True),
        log_dir   = lcfg.get("log_dir", "runs"),
        config    = cfg,
    )

    topos = get_all_fixed_topos()
    if args.random_topos > 0:
        rng = np.random.default_rng(cfg["env"]["seed"])
        topos += [random_topology(rng=rng) for _ in range(args.random_topos)]

    all_results = []
    for run_id in range(cfg["train"]["n_runs"]):
        run_results = {}
        for topo in topos:
            result = train_on_topology(topo, cfg, mode, run_id, logger)
            run_results[topo.name] = result
            print(f"[run {run_id}] {topo.name}: "
                  f"d95={result['d95_mean']*1000:.1f}ms  "
                  f"thp={result['thp_mean']:.3f}  "
                  f"per={result['per_mean']:.3f}")
        all_results.append(run_results)

    # 최종 평균 출력
    print("\n=== 최종 결과 (mean ± std across runs) ===")
    for topo in topos:
        d95s = [r[topo.name]["d95_mean"] * 1000 for r in all_results]
        print(f"{topo.name}: d95 = {np.mean(d95s):.2f} ± {np.std(d95s):.2f} ms")

    logger.close()


if __name__ == "__main__":
    main()
