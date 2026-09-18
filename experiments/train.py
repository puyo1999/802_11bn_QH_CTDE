"""
experiments/train.py
학습 진입점 — Attention-DRLCA / FC-DRLCA 공통

Usage:
    python experiments/train.py --config configs/drlca_attn.yaml
    python experiments/train.py --config configs/drlca_fc.yaml --mode cca
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from src.envs.obss_env import OBSSEnv
from src.config import load_config
from src.envs.topology import get_all_fixed_topos, random_topology
from src.agents.attn_drlca_agent import AttnDRLCAAgent
from src.utils.metrics import aggregate_metrics, d95_reduction
from src.utils.logger import Logger


def build_agents(n_aps: int, cfg: dict, mode: str) -> list[AttnDRLCAAgent]:
    if mode == "cw":
        n_actions = 7
    elif mode == "joint":
        n_actions = 35  # 7 * 5
    else:
        n_actions = 5
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

    # 에피소드별 기록을 저장할 리스트 (또는 TensorBoard writer 활용)
    history = []

    for ep in tqdm(range(ecfg["max_episodes"]),
                   desc=f"[{topo.name}] run={run_id}", leave=False):
        # ── 1. 각 AP: fusion → 현재 obs 구성 ──────────────────
        prev_obs_list = []
        for i, agent in enumerate(agents):
            nbr_ids    = topo.neighbors[i]
            nbr_states = [obs[j] for j in nbr_ids]
            agent.fuse(obs[i], nbr_states)
            prev_obs_list.append(agent.get_obs().copy())   # (T, F)

        # ── 2. 행동 선택 (Joint 모드 완벽 대응) ─────────────────
        agent_actions = []  # 에이전트가 선택한 원본 액션 (0~34) 저장용
        if mode == "cw":
            actions = np.array([a.select_action(prev_obs_list[i])
                                 for i, a in enumerate(agents)], dtype=np.int32)

        elif mode == "joint":
            joint_actions = []
            for i, a in enumerate(agents):
                act = a.select_action(prev_obs_list[i])  # 0~34 사이 값 출력
                agent_actions.append(act)

                # 환경이 원하는 [cw, cca, cw, cca...] 구조로 쪼개서 결합
                joint_actions.append(act // 5)  # CW exponent (0~6)
                joint_actions.append(act % 5)  # CCA delta (0~4)
            actions = np.array(joint_actions, dtype=np.int32)

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

        # ❌ [기존 위치 - 삭제] 여기에 있던 np.save() 로직을 복사해서 루프 바깥으로 이동합니다.
        # 매 에피소드마다 덮어쓰기 되던 문제를 해결하기 위함입니다.

    # ─────────────────────────────────────────────────────────
    #  loop가 완전히 끝난 지점 (indentation 주의!)
    # ─────────────────────────────────────────────────────────
    # --- [수정 및 이동] 에피소드 전체의 reward_log를 한 번에 저장 ---
    save_dir = f"./results/{mode}"
    os.makedirs(save_dir, exist_ok=True)

    topo_name = getattr(topo, 'name', f"Topo_{run_id}")
    file_path = f"{save_dir}/{topo_name}_run{run_id}.npy"

    # 단일 스칼라(reward) 대신, 누적된 리스트(reward_log)를 넘파이 배열로 변환하여 저장
    np.save(file_path, np.array(reward_log))
    # ─────────────────────────────────────────────────────────

    summary = aggregate_metrics(info_log)
    return summary


def run_random_on_topology(topo, cfg, run_id, logger):
    """지정된 토폴로지에서 무작위 액션(Random Baseline)으로 시뮬레이션을 수행합니다."""
    ecfg = cfg["env"]

    # [핵심 피드백] OBSSEnv가 인식할 수 있도록 yaml 설정을 따르거나 기본값 'cw'를 매핑합니다.
    # 이렇게 해야 AssertionError를 우회할 수 있습니다.
    env_action_mode = ecfg.get("mode", "cw")
    if env_action_mode == "Random":
        env_action_mode = "cw"

    base_seed = ecfg.get("seed", 42)
    run_seed = base_seed + run_id if base_seed is not None else None

    # 환경 생성 ("Random" 대신 "cw", "cca", "joint" 중 하나가 들어감)
    env = OBSSEnv(
        topology=topo,
        mode=env_action_mode,
        interact_slots=ecfg.get("interact_slots", 55_556),
        max_episodes=ecfg.get("max_episodes", 600),
        seed=run_seed,
        traffic_type=ecfg.get("traffic_type", "saturated"),
        poisson_arrival_rate=ecfg.get("poisson_arrival_rate", 0.1),
        queue_max_size=ecfg.get("queue_max_size", 100)
    )

    env.reset()
    # --- 추세 그래프용 롱텀 히스토리 리스트 ---
    ep_rewards = []
    ep_d95s, ep_thps, ep_pers = [], [], []

    for episode in range(env.max_episodes):
        # 환경의 Action Space 규칙에 맞춰 완벽한 무작위 값 추출
        actions = env.action_space.sample()
        _, reward, done, _, info = env.step(actions)

        # 1. 매 에피소드의 리워드를 순서대로 기록 (길게 추세를 보기 위함)
        ep_rewards.append(reward)

        ep_d95s.append(np.mean(info["d95"]))
        ep_thps.append(np.mean(info["thp"]))
        ep_pers.append(np.mean(info["per"]))

        if done:
            break
    # --- [핵심] 논문 그래프용 데이터 일괄 저장 ---
    save_dir = "./results/Random"
    os.makedirs(save_dir, exist_ok=True)

    # 파일명에 토폴로지와 run_id를 명시하여 DRL 결과와 1:1 매핑 가능하도록 저장
    reward_file_path = f"{save_dir}/{topo.name}_run{run_id}_rewards.npy"
    np.save(reward_file_path, np.array(ep_rewards))

    # (선택) TensorBoard에도 Random 추세가 실시간으로 길게 찍히도록 연동
    if logger and hasattr(logger, "writer") and logger.use_tb:
        for ep_idx, r_val in enumerate(ep_rewards):
            logger.writer.add_scalar(f"Baseline_Random/{topo.name}_run{run_id}", r_val, ep_idx)
    return {
        "d95_mean": float(np.mean(ep_d95s)),
        "thp_mean": float(np.mean(ep_thps)),
        "per_mean": float(np.mean(ep_pers)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/drlca_attn.yaml")
    parser.add_argument("--mode",   default=None,
                        help="override env.mode (cw|cca|joint|random)")
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
            # 인자로 "Random"이 들어왔을 때 에전트 레벨에서 무작위 샘플링 루프로 우회
            if mode.lower() == "random":
                result = run_random_on_topology(topo, cfg, run_id, logger)
            else:
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
