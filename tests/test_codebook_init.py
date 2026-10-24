"""
tests/test_codebook_init.py
reset 기반 초기화의 코드북 붕괴와 k-means 초기화의 개선을 검증한다.

실행:
    python tests/test_codebook_init.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.agents.codebook_init import collect_observations, kmeans_centers, quantization_report
from src.config import ExperimentConfig, load_config
from src.envs.env_factory import make_env
from src.envs.topology_registry import get_registry

_failures = []


def check(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        _failures.append(label)


def main() -> int:
    exp = ExperimentConfig.from_dict(load_config("configs/qh_ctde.yaml"))
    exp.env.max_episodes = 128
    env = make_env(exp.env, topology=get_registry("configs/topologies.yaml").get("T3"),
                   topology_file="configs/topologies.yaml")

    print("\n[1] reset 관측은 시드와 무관하게 같다 (기존 초기화가 붕괴하는 이유)")
    reset_rows = np.concatenate([env.reset(seed=43 + k)[0] for k in range(8)], axis=0)
    check(len(np.unique(reset_rows.round(6), axis=0)) <= 4, "reset 8회 샘플의 서로 다른 벡터 수 <= 4")

    print("\n[2] 에피소드 전체 관측은 다양하다")
    data = collect_observations(env, n_episodes=4, base_seed=100)
    check(len(np.unique(data.round(6), axis=0)) > 100, "서로 다른 관측 벡터 > 100")

    print("\n[3] k-means 중심은 16개 코드를 모두 쓰고 재구성 오차가 훨씬 작다")
    centers = kmeans_centers(data, 16, seed=0)
    heldout = collect_observations(env, n_episodes=4, base_seed=900)
    cur = np.unique(reset_rows.round(6), axis=0)
    cur = np.concatenate([cur, np.tile(cur[:1], (16 - len(cur), 1))])
    a, b = quantization_report(heldout, cur), quantization_report(heldout, centers)
    print(f"    reset 기반 {a}\n    k-means   {b}")
    check(len(centers) == 16, "중심 16개")
    check(b["codes_used"] == 16, "held-out에서 16개 코드 모두 사용")
    check(b["mse"] < 0.1 * a["mse"], "MSE가 기존의 10% 미만")

    print("\n[4] 같은 시드면 같은 결과 (재현성)")
    c1 = kmeans_centers(collect_observations(env, 2, base_seed=7), 16, seed=3)
    c2 = kmeans_centers(collect_observations(env, 2, base_seed=7), 16, seed=3)
    check(np.allclose(c1, c2), "동일 시드에서 중심이 동일")

    print("\n" + "=" * 60)
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        return 1
    print("전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
