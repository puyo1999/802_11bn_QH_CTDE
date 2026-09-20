"""
tests/test_traffic_ratio.py
연속 부하 비율(traffic_load_ratio) 트래픽 모델과, 정책이 실제 큐/보상에
영향을 미치는지(= saturated 고정 버그가 해소되었는지) 검증한다.

실행:
    python tests/test_traffic_ratio.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import ExperimentConfig, load_config
from src.envs.env_factory import make_env
from src.envs.topology_registry import get_registry

CONFIG = "configs/qh_ctde.yaml"
TOPO_FILE = "configs/topologies.yaml"

_failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        _failures.append(label)


def base_config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_config(CONFIG))


def test_lambda_interpolation() -> None:
    print("\n[1] traffic_load_ratio -> effective_lambda 보간")
    from src.envs.traffic import TrafficConfig

    lo = TrafficConfig(traffic_load_ratio=0.0, poisson_lambda_min=0.1, poisson_lambda_max=3.0)
    hi = TrafficConfig(traffic_load_ratio=1.0, poisson_lambda_min=0.1, poisson_lambda_max=3.0)
    mid = TrafficConfig(traffic_load_ratio=0.5, poisson_lambda_min=0.1, poisson_lambda_max=3.0)

    check(abs(lo.effective_lambda - 0.1) < 1e-9, "ratio=0 -> lambda_min")
    check(abs(hi.effective_lambda - 3.0) < 1e-9, "ratio=1 -> lambda_max")
    check(abs(mid.effective_lambda - 1.55) < 1e-9, "ratio=0.5 -> 선형 중간값")

    try:
        TrafficConfig(traffic_load_ratio=1.5)
        check(False, "범위 밖 ratio는 예외를 발생시킨다")
    except ValueError:
        check(True, "범위 밖 ratio는 예외를 발생시킨다")


def test_queue_always_responds_to_policy() -> None:
    print("\n[2] 성공 전송 시 큐가 실제로 차감되는지 (saturated 고정 버그 해소 확인)")
    # 나이브 휴리스틱 정책 간 비교는 다중 AP 간섭 하에서 신뢰할 수 없다 — 간섭을
    # 고려하지 않고 고른 높은 MCS는 실제로는 자주 디코딩에 실패할 수 있고, 그 자체가
    # RL이 필요한 이유이지 버그가 아니다. 여기서는 도착률을 0으로 고정한 결정론적
    # 조건에서, 전송이 성공할 때 큐가 정확히 tx_pkts만큼 줄어드는지를 직접 계측한다.
    exp = base_config()
    topo = get_registry(TOPO_FILE).get("T1")

    for ratio, label in [(1.0, "ratio=1.0"), (0.3, "ratio=0.3")]:
        # effective_lambda는 traffic_load_ratio에 따라 달라지지만, 도착 자체를
        # 끄면(둘 다 lambda=0) ratio 값과 무관하게 큐 차감 메커니즘만 격리된다.
        exp.env.traffic = {
            "traffic_load_ratio": ratio, "poisson_lambda_min": 0.0,
            "poisson_lambda_max": 0.0, "init_fill_ratio": 0.5,
        }
        env = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
        env.reset(seed=1)
        env.harsh_mode_enabled = False

        # 오라클 MCS: 실제 동시 전송 시나리오의 SINR을 미리 계산해 확실히
        # 디코딩 가능한 MCS를 선택한다 (간섭 인지 링크 적응).
        queue_before = [float(env._aps[i].queues[0]) for i in range(env.n_aps)]
        all_transmitting = list(range(env.n_aps))
        action = []
        expected_tx = []
        for i in range(env.n_aps):
            sinr = env.interference.compute(i, 0, all_transmitting, is_harsh=False).sinr_db
            mcs = env.mcs_table.best_mcs_for(sinr)
            action.extend([0, mcs])
            expected_tx.append(min(queue_before[i], env.mcs_table.capacity(mcs)))

        _, _, _, _, info = env.step(np.array(action, dtype=np.int32))
        queue_after = [float(env._aps[i].queues[0]) for i in range(env.n_aps)]

        actual_drop = [b - a for b, a in zip(queue_before, queue_after)]
        matches = all(abs(d - e) < 1e-6 for d, e in zip(actual_drop, expected_tx))
        print(
            f"    [{label}] 이전 큐={queue_before}  이후 큐={queue_after}  "
            f"기대 차감={expected_tx}"
        )
        check(matches, f"{label}: 도착 0 조건에서 큐 차감량이 오라클 MCS 전송량과 정확히 일치")
        check(sum(actual_drop) > 0, f"{label}: 실제로 0이 아닌 차감이 발생했다 (saturated 고정 아님)")


def test_ratio_controls_saturation_level() -> None:
    print("\n[3] ratio가 실제 포화 수준을 제어하는지 (무작위 정책 기준)")
    exp = base_config()
    topo = get_registry(TOPO_FILE).get("T1")

    def mean_queue_under_random_policy(ratio: float) -> float:
        exp.env.traffic = {"traffic_load_ratio": ratio, "poisson_lambda_min": 0.1,
                            "poisson_lambda_max": 3.0, "init_fill_ratio": 0.0}
        env = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
        env.reset(seed=7)
        env.harsh_mode_enabled = False
        means = []
        for _ in range(60):
            _, _, _, _, info = env.step(env.action_space.sample())
            means.append(np.mean(info["queue_mean"]))
        return float(np.mean(means))

    q_low = mean_queue_under_random_policy(0.05)
    q_high = mean_queue_under_random_policy(1.0)
    print(f"    ratio=0.05 평균 큐={q_low:.1f}   ratio=1.0 평균 큐={q_high:.1f}")
    check(q_high > q_low, "ratio가 높을수록 평균 큐 점유가 커진다")


def test_reward_reasonable_scale() -> None:
    print("\n[4] 리워드 스케일 — throughput 항이 latency 항에 완전히 묻히지 않는지")
    exp = base_config()
    topo = get_registry(TOPO_FILE).get("T3")
    exp.env.traffic = {"traffic_load_ratio": 0.7, "poisson_lambda_min": 0.1,
                        "poisson_lambda_max": 3.0, "init_fill_ratio": 0.3}
    exp.env.max_episodes = 50

    env = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
    env.reset(seed=3)
    rewards = []
    for _ in range(50):
        _, reward, done, _, info = env.step(env.action_space.sample())
        rewards.append(reward)

    mean_abs = float(np.mean(np.abs(rewards)))
    print(f"    무작위 정책 평균 |reward| = {mean_abs:.2f} (참고: 구 버그 상태에서는 ~10000+)")
    check(mean_abs < 200.0, "평균 |reward|가 이전 폭주 수준(수천~수만)보다 훨씬 작다")
    check(np.std(rewards) > 0.0, "보상에 정책/트래픽에 따른 변동이 존재한다")


def main() -> int:
    print("=" * 70)
    print("traffic_load_ratio 트래픽 모델 검증")
    print("=" * 70)
    test_lambda_interpolation()
    test_queue_always_responds_to_policy()
    test_ratio_controls_saturation_level()
    test_reward_reasonable_scale()

    print("\n" + "=" * 70)
    if _failures:
        print(f"실패 {len(_failures)}건:")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
