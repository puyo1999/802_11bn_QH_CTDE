"""
src/training/result_store.py
토폴로지×시드 보상 이력을 디스크에 저장/로딩한다.

경로 규칙: {result_dir}/{run_name}/{topo_id}/run{run_id}_rewards.npy
"""

from __future__ import annotations

import os

import numpy as np


def result_dir_for(base_dir: str, run_name: str, topo_id: str) -> str:
    return os.path.join(base_dir, run_name, topo_id)


def save_reward_history(base_dir: str, run_name: str, topo_id: str, run_id: int,
                         reward_history: np.ndarray) -> str:
    directory = result_dir_for(base_dir, run_name, topo_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"run{run_id}_rewards.npy")
    np.save(path, reward_history)
    return path


def load_reward_histories(base_dir: str, run_name: str, topo_id: str) -> list[np.ndarray]:
    directory = result_dir_for(base_dir, run_name, topo_id)
    if not os.path.isdir(directory):
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith("_rewards.npy"))
    return [np.load(os.path.join(directory, f)) for f in files]
