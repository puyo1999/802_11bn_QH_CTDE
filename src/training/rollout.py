"""
src/training/rollout.py
단일 에피소드 롤아웃. 학습과 평가가 동일 경로를 쓰도록 train 플래그로 분기한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.agents.dsm_gate import DSMMode
from src.agents.qh_ctde_agent import QHCTDEAgent


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    mean_throughput: float
    overlap_blocked: int
    reward_breakdown: dict[str, float] = field(default_factory=dict)


def run_episode(
    env,
    agents: list[QHCTDEAgent],
    train: bool,
    dsm_mode: DSMMode = "auto",
    seed: int | None = None,
) -> EpisodeResult:
    """
    한 에피소드를 끝까지 실행한다.

    train=True  : eps-greedy 탐험 유지, 매 스텝 store_transition + update + decay_epsilon
    train=False : eps=0 강제(그리디), 학습 없음
    """
    obs, _ = env.reset(seed=seed)
    n_aps = env.n_aps

    saved_eps = None
    if not train:
        saved_eps = [a.eps for a in agents]
        for a in agents:
            a.eps = 0.0
        for a in agents:
            a.set_eval()

    total_reward = 0.0
    thp_sum = 0.0
    overlap_blocked = 0
    breakdown_sum: dict[str, float] = {}
    done = False
    steps = 0

    try:
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                valid_sta_cnt = len(env.topo.sta_positions[i])
                act = agent.select_action(obs[i], valid_sta_cnt, dsm_mode=dsm_mode)
                actions.extend(act.tolist())

            next_obs, reward, done, _, info = env.step(np.array(actions, dtype=np.int32))

            if train:
                for i, agent in enumerate(agents):
                    agent.store_transition(obs[i], (actions[2 * i], actions[2 * i + 1]),
                                            reward, next_obs[i], done)
                    agent.update(dsm_mode=dsm_mode)
                    agent.decay_epsilon()

            total_reward += reward
            thp_sum += info.get("system_throughput", 0.0)
            overlap_blocked += info.get("overlap_blocked_step", 0)
            for k, v in info.items():
                if k.startswith("reward_"):
                    breakdown_sum[k] = breakdown_sum.get(k, 0.0) + v

            obs = next_obs
            steps += 1
    finally:
        if not train:
            for a, e in zip(agents, saved_eps):
                a.eps = e
            for a in agents:
                a.set_train()

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        mean_throughput=thp_sum / max(steps, 1),
        overlap_blocked=overlap_blocked,
        reward_breakdown={k: v / max(steps, 1) for k, v in breakdown_sum.items()},
    )
