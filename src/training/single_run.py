"""
src/training/single_run.py
토폴로지 1개 · 시드 1개에 대한 학습 실행 단위.

책임: 환경/에이전트 생성 -> N 에피소드 학습 -> 에피소드별 보상 이력 반환.
멀티 토폴로지·멀티 시드 오케스트레이션은 src/training/experiment.py가 담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.agents.agent_factory import build_agents
from src.agents.qh_ctde_agent import QHCTDEAgent
from src.config.schema import ExperimentConfig
from src.envs.env_factory import make_env
from src.envs.space_validator import validate_spaces
from src.envs.topology import Topology
from src.training.rollout import run_episode


@dataclass
class RunResult:
    topology_id: str
    run_id: int
    reward_history: np.ndarray
    agents: list[QHCTDEAgent]


def train_single_run(
    topology: Topology,
    topo_id: str,
    exp_cfg: ExperimentConfig,
    run_id: int,
    topology_file: str,
    progress: bool = False,
) -> RunResult:
    """단일 (토폴로지, 시드) 조합에 대한 전체 학습을 실행한다."""
    started_at = perf_counter()
    env = make_env(exp_cfg.env, topology=topology, seed_offset=run_id, topology_file=topology_file)
    space = validate_spaces(env, topology, declared_state_dim=exp_cfg.agent.state_dim, strict=True)
    agents = build_agents(space, exp_cfg.agent)

    n_episodes = exp_cfg.train.max_episodes
    reward_history = np.zeros(n_episodes, dtype=np.float64)

    for ep in range(n_episodes):
        result = run_episode(env, agents, train=True, seed=exp_cfg.env.seed + run_id)
        reward_history[ep] = result.total_reward
        if progress and (ep % max(1, n_episodes // 10) == 0):
            print(f"    [{topo_id} run={run_id}] ep={ep:4d}/{n_episodes}  reward={result.total_reward:.2f}")

    if progress:
        elapsed_seconds = perf_counter() - started_at
        print(
            f"    [{topo_id} run={run_id}] complete  "
            f"elapsed={elapsed_seconds:.1f}s  "
            f"avg={elapsed_seconds / max(n_episodes, 1):.3f}s/episode"
        )

    return RunResult(topology_id=topo_id, run_id=run_id, reward_history=reward_history, agents=agents)
