"""
src/training/experiment.py
토폴로지 선택자 x n_runs 시드에 걸쳐 학습을 반복하고 결과를 저장한다.
"""

from __future__ import annotations

from src.config.schema import ExperimentConfig
from src.envs.topology_registry import get_registry
from src.training.result_store import save_reward_history
from src.training.single_run import train_single_run


def run_experiment(
    exp_cfg: ExperimentConfig,
    topology_file: str = "configs/topologies.yaml",
    progress: bool = True,
) -> dict[str, list[str]]:
    """
    exp_cfg.env.topology 선택자(단일 ID / 리스트 / 'group:x' / 'all')와
    exp_cfg.train.n_runs 만큼 학습을 반복하고, 저장된 파일 경로를 topo_id별로 반환한다.
    """
    registry = get_registry(topology_file)
    topologies = registry.resolve(exp_cfg.env.topology)
    saved_paths: dict[str, list[str]] = {}

    for topo in topologies:
        topo_id = topo.metadata.get("id", topo.name)
        saved_paths[topo_id] = []
        if progress:
            print(f"[{topo_id}] {topo.name} — {exp_cfg.train.n_runs} runs x "
                  f"{exp_cfg.train.max_episodes} episodes")

        for run_id in range(exp_cfg.train.n_runs):
            result = train_single_run(
                topology=topo,
                topo_id=topo_id,
                exp_cfg=exp_cfg,
                run_id=run_id,
                topology_file=topology_file,
                progress=progress,
            )
            path = save_reward_history(
                exp_cfg.train.result_dir, exp_cfg.logging.run_name, topo_id,
                run_id, result.reward_history,
            )
            saved_paths[topo_id].append(path)

    return saved_paths
