"""
tests/test_overlap_interference.py
중첩 STA 명시적 제약과 동적 관측 차원 검증.

실행:
    python tests/test_overlap_interference.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import ExperimentConfig, load_config
from src.envs.env_factory import make_env
from src.envs.overlap import OverlapConfig, OverlapPolicy
from src.envs.space_validator import build_adjacency, validate_spaces
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


def test_overlap_policy() -> None:
    print("\n[1] OverlapPolicy 판정 로직")
    topo = get_registry(TOPO_FILE).get("T6")
    policy = OverlapPolicy(topo, OverlapConfig(mode="hard"))

    check(policy.is_active(), "T6에서 정책이 활성화된다")
    check(policy.peers_of(3, 1) == [1, 2], "AP3의 중첩 STA가 AP1/AP2와 연결된다")
    check(policy.peers_of(0, 0) == [], "비중첩 STA는 상대 AP가 없다")

    check(
        policy.blocks_transmission(3, 1, transmitting_aps=[3, 1]),
        "중첩 상대 AP가 전송 중이면 차단된다",
    )
    check(
        not policy.blocks_transmission(3, 1, transmitting_aps=[3, 0]),
        "비중첩 AP만 전송 중이면 차단되지 않는다",
    )
    check(
        not policy.blocks_transmission(3, 0, transmitting_aps=[3, 1, 2]),
        "비중첩 STA는 어떤 조합에서도 차단되지 않는다",
    )

    soft = OverlapPolicy(topo, OverlapConfig(mode="soft", soft_gain_db=10.0))
    check(not soft.blocks_transmission(3, 1, [3, 1]), "soft 모드는 전송을 차단하지 않는다")
    check(
        abs(soft.interference_gain(3, 1, 1, [3, 1]) - 10.0) < 1e-9,
        "soft 모드는 중첩 AP 간섭에 10 dB 이득을 적용한다",
    )
    check(
        abs(soft.interference_gain(3, 1, 0, [3, 0]) - 1.0) < 1e-9,
        "비중첩 AP 간섭에는 이득을 적용하지 않는다",
    )


def test_observation_dim() -> None:
    print("\n[2] 동적 관측 차원 (overlap 지시자 포함 여부)")
    exp = base_config()
    topo = get_registry(TOPO_FILE).get("T6")

    exp.env.overlap = {"enabled": True, "mode": "hard", "expose_in_obs": True}
    env_with = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
    k = env_with.k_stas
    check(env_with.obs_dim == 3 * k + 2, f"지시자 포함 시 obs_dim = 3K+2 = {3 * k + 2}")

    exp.env.overlap = {"enabled": True, "mode": "hard", "expose_in_obs": False}
    env_without = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
    check(env_without.obs_dim == 2 * k + 2, f"지시자 제외 시 obs_dim = 2K+2 = {2 * k + 2}")

    layout = env_with.feature_layout()
    names = [name for name, _, _ in layout]
    check("overlap_indicator" in names, "feature_layout에 overlap_indicator가 포함된다")

    obs, _ = env_with.reset()
    start, length = next((s, l) for n, s, l in layout if n == "overlap_indicator")
    ap3_indicator = obs[3][start:start + length]
    check(ap3_indicator[1] > 0.0, "AP3의 중첩 STA 지시자가 0보다 크다")
    check(ap3_indicator[0] == 0.0, "AP3의 비중첩 STA 지시자는 0이다")


def test_hard_constraint_effect() -> None:
    print("\n[3] hard 제약의 전송 결과 반영")
    exp = base_config()
    exp.env.traffic_type = "saturated"
    exp.env.max_episodes = 40
    exp.env.harsh_mode_enabled = False
    topo = get_registry(TOPO_FILE).get("T6")

    # 전 AP가 중첩 STA를 동시에 지정하는 결정론적 액션
    def run(overlap_cfg: dict) -> tuple[float, int]:
        exp.env.overlap = overlap_cfg
        env = make_env(exp.env, topology=topo, seed_offset=0, topology_file=TOPO_FILE)
        env.reset()
        total_thp, total_blocked = 0.0, 0
        action = []
        for i in range(env.n_aps):
            # AP1/AP2/AP3은 중첩 STA(로컬 인덱스 1), AP0은 유일 STA(0)를 지정
            sta = 1 if len(topo.sta_positions[i]) > 1 else 0
            action.extend([sta, 4])
        action = np.array(action, dtype=np.int32)

        for _ in range(20):
            _, _, _, _, info = env.step(action)
            total_thp += info["system_throughput"]
            total_blocked += info["overlap_blocked_step"]
        return total_thp, total_blocked

    thp_hard, blocked_hard = run({"enabled": True, "mode": "hard", "expose_in_obs": True})
    thp_off, blocked_off = run({"enabled": False, "mode": "hard", "expose_in_obs": True})

    print(f"    hard 제약 ON : throughput={thp_hard:.0f}  blocked={blocked_hard}")
    print(f"    hard 제약 OFF: throughput={thp_off:.0f}  blocked={blocked_off}")

    check(blocked_hard > 0, "중첩 STA 동시 지정 시 차단이 발생한다")
    check(blocked_off == 0, "제약 비활성화 시 차단이 발생하지 않는다")
    check(thp_hard < thp_off, "hard 제약이 처리량을 실제로 제한한다")


def test_space_validation() -> None:
    print("\n[4] 공간 정합성 검증 (T1~T6)")
    exp = base_config()
    reg = get_registry(TOPO_FILE)

    for topo_id in reg.ids():
        topo = reg.get(topo_id)
        env = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
        spec = validate_spaces(env, topo, declared_state_dim=None)
        expected = 3 * env.k_stas + 2
        ok = spec.local_obs_dim == expected and spec.global_state_dim == expected * topo.n_aps
        print(f"    {topo_id}: {spec.describe()}")
        check(ok, f"{topo_id} 관측/전역 상태 차원 정합")

    # 선언값 불일치 시 예외 발생 여부
    topo = reg.get("T1")
    env = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)
    try:
        validate_spaces(env, topo, declared_state_dim=9, strict=True)
        check(False, "선언 차원 불일치 시 예외가 발생한다")
    except ValueError:
        check(True, "선언 차원 불일치 시 예외가 발생한다")

    adj = build_adjacency(reg.get("T1"))
    check(adj.shape == (3, 3) and adj[0, 0] == 1.0, "인접 행렬 생성 (자기 연결 포함)")


def test_rollout_smoke() -> None:
    print("\n[5] 무작위 정책 롤아웃 (T3)")
    exp = base_config()
    exp.env.max_episodes = 30
    topo = get_registry(TOPO_FILE).get("T3")
    env = make_env(exp.env, topology=topo, topology_file=TOPO_FILE)

    obs, _ = env.reset()
    rewards, harsh_seen, blocked_total = [], False, 0
    done = False
    while not done:
        obs, reward, done, _, info = env.step(env.action_space.sample())
        rewards.append(reward)
        harsh_seen = harsh_seen or info["is_harsh_mode"]
        blocked_total += info["overlap_blocked_step"]

    check(len(rewards) == 30, "max_episodes 스텝 후 종료된다")
    check(np.all(np.isfinite(rewards)), "보상이 모두 유한하다")
    check(harsh_seen, "에피소드 후반에 가혹 조건이 활성화된다")
    check(obs.shape == (topo.n_aps, env.obs_dim), "관측 형상이 유지된다")
    print(f"    평균 보상={np.mean(rewards):.3f}  누적 중첩 차단={blocked_total}")


def main() -> int:
    print("=" * 70)
    print("Overlap 제약 / 간섭 모델 / 공간 차원 검증")
    print("=" * 70)
    test_overlap_policy()
    test_observation_dim()
    test_hard_constraint_effect()
    test_space_validation()
    test_rollout_smoke()

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
