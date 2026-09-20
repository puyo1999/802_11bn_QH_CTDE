"""
src/plot/reward_curves.py
토폴로지별 보상 수렴 곡선. mean ± std 음영은 실제 멀티시드 결과에서만 계산한다.
marl_plotter.py의 정책별 고정 배율(0.15~0.70) 같은 인위적 보정은 일절 쓰지 않는다.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _stack_runs(histories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """길이가 다른 run들은 최단 길이로 잘라 정렬한다."""
    min_len = min(len(h) for h in histories)
    arr = np.stack([h[:min_len] for h in histories], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_single_topology(
    histories: list[np.ndarray],
    topo_id: str,
    topo_name: str,
    save_path: str,
    smooth_window: int = 10,
) -> str:
    """토폴로지 1개의 mean±std 수렴 곡선을 그린다."""
    mean, std = _stack_runs(histories)
    episodes = np.arange(len(mean))

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, mean, color="#2ecc71", linewidth=1.0, alpha=0.35, label="Mean (raw)")
    if smooth_window > 1 and len(mean) >= smooth_window:
        smoothed = moving_average(mean, smooth_window)
        plt.plot(
            np.arange(smooth_window - 1, len(mean)), smoothed,
            color="#27ae60", linewidth=2.2, label=f"{smooth_window}-ep moving avg",
        )
    plt.fill_between(episodes, mean - std, mean + std, color="#2ecc71", alpha=0.15,
                      label=f"±1 std (n={len(histories)} runs)")

    plt.title(f"Reward convergence — {topo_name} ({topo_id})", fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_topology_grid(
    per_topo_histories: dict[str, list[np.ndarray]],
    topo_names: dict[str, str],
    save_path: str,
    smooth_window: int = 10,
    n_cols: int = 3,
) -> str:
    """전 토폴로지를 한 Figure에 서브플롯으로 배열한다 (논문 6분할 그림용)."""
    topo_ids = list(per_topo_histories.keys())
    n = len(topo_ids)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4 * n_rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, topo_id in enumerate(topo_ids):
        ax = axes[idx]
        histories = per_topo_histories[topo_id]
        mean, std = _stack_runs(histories)
        episodes = np.arange(len(mean))

        ax.plot(episodes, mean, color="#2ecc71", alpha=0.3, linewidth=1.0)
        if smooth_window > 1 and len(mean) >= smooth_window:
            smoothed = moving_average(mean, smooth_window)
            ax.plot(np.arange(smooth_window - 1, len(mean)), smoothed,
                    color="#27ae60", linewidth=2.0)
        ax.fill_between(episodes, mean - std, mean + std, color="#2ecc71", alpha=0.15)

        ax.set_title(f"{topo_id}: {topo_names.get(topo_id, topo_id)}", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx >= n - n_cols:
            ax.set_xlabel("Episode", fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel("Episode reward", fontsize=9)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"QH-CTDE reward convergence across {n} topologies "
                 f"(n_runs={len(next(iter(per_topo_histories.values())))})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def summarize_final_performance(
    per_topo_histories: dict[str, list[np.ndarray]], tail_episodes: int = 20
) -> dict[str, dict[str, float]]:
    """토폴로지별 마지막 tail_episodes 평균/표준편차 — 논문 Table용 수치."""
    summary = {}
    for topo_id, histories in per_topo_histories.items():
        mean, std = _stack_runs(histories)
        tail = min(tail_episodes, len(mean))
        summary[topo_id] = {
            "final_mean": float(np.mean(mean[-tail:])),
            "final_std": float(np.mean(std[-tail:])),
            "n_runs": len(histories),
        }
    return summary
