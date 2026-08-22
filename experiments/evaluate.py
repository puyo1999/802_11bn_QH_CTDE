"""
experiments/evaluate.py
평가 스크립트 — cross-topology 테스트 + attention score 시각화

Usage:
    python experiments/evaluate.py \
        --config configs/drlca_attn.yaml \
        --checkpoint runs/attn_drlca_cw/best.pt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from attention_drlca.src.envs.obss_env import OBSSEnv
from attention_drlca.src.envs.topology import get_all_fixed_topos, random_topology
from attention_drlca.src.agents.attn_drlca_agent import AttnDRLCAAgent
from attention_drlca.src.utils.metrics import aggregate_metrics, d95_reduction
from attention_drlca.experiments.train import load_config, build_agents


def evaluate_topology(topo, agents: list[AttnDRLCAAgent],
                      cfg: dict, mode: str) -> dict:
    """탐색 없이 평가"""
    ecfg = cfg["env"]
    env  = OBSSEnv(topo, mode=mode,
                   interact_slots=ecfg["interact_slots"],
                   max_episodes=cfg["eval"]["n_episodes"],
                   seed=0)
    for agent in agents:
        agent.eps = 0.0  # greedy

    obs, _ = env.reset()
    info_log = []
    attn_log  = []   # (ep, n_aps, N_nbr) attention 저장

    for ep in range(cfg["eval"]["n_episodes"]):
        ep_attn = []
        for i, agent in enumerate(agents):
            nbr_ids    = topo.neighbors[i]
            nbr_states = [obs[j] for j in nbr_ids]
            agent.fuse(obs[i], nbr_states)
            ep_attn.append(agent.get_last_attention())

        attn_log.append(ep_attn)

        actions = np.array([a.select_action(a.get_obs())
                             for a in agents], dtype=np.int32)
        obs, _, done, _, info = env.step(actions)
        info_log.append(info)
        if done:
            obs, _ = env.reset()

    summary = aggregate_metrics(info_log)
    summary["attn_log"] = attn_log
    return summary


def plot_attention_heatmap(
    attn_log: list,
    topo_name: str,
    n_aps: int,
    save_dir: str = "results",
):
    """
    AP별 이웃 attention 가중치 시계열 히트맵.
    논문 contribution: 해석 가능성 실험 Figure
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, n_aps, figsize=(4 * n_aps, 3))
    if n_aps == 1:
        axes = [axes]

    for i in range(n_aps):
        scores = []
        for ep_attn in attn_log:
            a = ep_attn[i]
            if a is not None:
                scores.append(a if a.ndim > 0 else np.array([a]))
        if not scores:
            continue
        max_len = max(len(s) for s in scores)
        mat = np.zeros((len(scores), max_len))
        for t, s in enumerate(scores):
            mat[t, :len(s)] = s

        axes[i].imshow(mat.T, aspect="auto", cmap="Blues",
                       vmin=0, vmax=1, interpolation="nearest")
        axes[i].set_title(f"AP{i+1} attn")
        axes[i].set_xlabel("Episode")
        axes[i].set_ylabel("Neighbor idx")

    fig.suptitle(f"Attention weights — {topo_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"attn_{topo_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/drlca_attn.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode",       default=None)
    parser.add_argument("--random_topos", type=int, default=0)
    parser.add_argument("--save_dir",   default="results")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    mode = args.mode or cfg["env"].get("mode", "cw")

    topos = get_all_fixed_topos()
    if args.random_topos > 0:
        rng = np.random.default_rng(99)
        topos += [random_topology(rng=rng) for _ in range(args.random_topos)]

    print(f"\n=== Evaluation: {mode.upper()} mode ===")
    for topo in topos:
        agents = build_agents(topo.n_aps, cfg, mode)
        if args.checkpoint:
            # 하나의 공유 체크포인트에서 로드 (분산학습 단순화)
            for agent in agents:
                try:
                    agent.load(args.checkpoint)
                except Exception:
                    pass

        result = evaluate_topology(topo, agents, cfg, mode)
        print(f"  {topo.name}: "
              f"d95={result['d95_mean']*1000:.1f}ms  "
              f"thp={result['thp_mean']:.3f}  "
              f"per={result['per_mean']:.3f}")

        plot_attention_heatmap(
            result["attn_log"], topo.name, topo.n_aps, args.save_dir
        )


if __name__ == "__main__":
    main()
