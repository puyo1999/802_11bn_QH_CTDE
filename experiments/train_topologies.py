"""
experiments/train_topologies.py
설정 파일 하나로 토폴로지 선택 x 멀티시드 학습 x 리워드 그래프 생성까지 실행한다.

Usage:
    python experiments/train_topologies.py --config configs/qh_ctde.yaml
    python experiments/train_topologies.py --config configs/qh_ctde.yaml \
        --set env.topology=group:with_overlap train.n_runs=3 train.max_episodes=200
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig, load_config
from src.envs.topology_registry import get_registry
from src.plot.reward_curves import plot_single_topology, plot_topology_grid, summarize_final_performance
from src.training.experiment import run_experiment
from src.training.result_store import load_reward_histories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topology-file", default="configs/topologies.yaml")
    parser.add_argument("--plot-dir", default="./plots")
    parser.add_argument(
        "--set", nargs="*", default=None,
        help="dot-path 오버라이드, 예: train.n_runs=3 env.topology=group:with_overlap",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    exp_cfg = ExperimentConfig.from_dict(cfg)

    print(f"실험: {exp_cfg.logging.run_name}")
    print(f"  topology={exp_cfg.env.topology}  n_runs={exp_cfg.train.n_runs}  "
          f"max_episodes={exp_cfg.train.max_episodes}")
    print(f"  traffic_load_ratio={exp_cfg.env.traffic.get('traffic_load_ratio')}  "
          f"overlap={exp_cfg.env.overlap}")

    saved = run_experiment(exp_cfg, topology_file=args.topology_file, progress=True)

    registry = get_registry(args.topology_file)
    per_topo_histories = {
        topo_id: load_reward_histories(exp_cfg.train.result_dir, exp_cfg.logging.run_name, topo_id)
        for topo_id in saved
    }
    topo_names = {topo_id: registry.get(topo_id).name for topo_id in saved}

    print("\n토폴로지별 개별 그래프 생성 중...")
    for topo_id, histories in per_topo_histories.items():
        path = plot_single_topology(
            histories, topo_id, topo_names[topo_id],
            save_path=os.path.join(args.plot_dir, exp_cfg.logging.run_name, f"{topo_id}_reward.png"),
        )
        print(f"  {path}")

    if len(per_topo_histories) > 1:
        grid_path = plot_topology_grid(
            per_topo_histories, topo_names,
            save_path=os.path.join(args.plot_dir, exp_cfg.logging.run_name, "all_topologies_grid.png"),
        )
        print(f"\n통합 그리드: {grid_path}")

    print("\n최종 성능 요약 (마지막 tail 20 에피소드 평균):")
    for topo_id, stats in summarize_final_performance(per_topo_histories).items():
        print(f"  {topo_id} ({topo_names[topo_id]}): "
              f"{stats['final_mean']:.3f} ± {stats['final_std']:.3f}  (n_runs={stats['n_runs']})")


if __name__ == "__main__":
    main()
