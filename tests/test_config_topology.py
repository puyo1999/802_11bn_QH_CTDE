"""
tests/test_config_topology.py
Configuration / Topology 계층 단위 검증.

실행:
    python tests/test_config_topology.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig, apply_overrides, deep_merge, load_config
from src.envs.topology_registry import get_registry

CONFIG = "configs/qh_ctde.yaml"
TOPO_FILE = "configs/topologies.yaml"

_failures: list[str] = []


def check(condition: bool, label: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        _failures.append(label)


def test_deep_merge() -> None:
    print("\n[1] deep merge — 부분 재정의 시 형제 키 보존")
    base = {"env": {"mode": "qh", "seed": 42, "k_stas": 4}, "train": {"n_runs": 5}}
    child = {"env": {"mode": "cw"}}
    merged = deep_merge(base, child)
    check(merged["env"]["mode"] == "cw", "자식 값이 우선 적용된다")
    check(merged["env"]["seed"] == 42, "재정의하지 않은 형제 키가 보존된다")
    check(merged["train"]["n_runs"] == 5, "다른 섹션이 보존된다")
    check(base["env"]["mode"] == "qh", "원본 딕셔너리가 변형되지 않는다")


def test_config_loading() -> None:
    print("\n[2] 설정 로딩 및 _base_ 상속")
    cfg = load_config(CONFIG)
    exp = ExperimentConfig.from_dict(cfg)
    check(exp.env.impl == "qh_v2", "자식 설정의 env.impl 적용")
    check(exp.env.interact_slots == 55556, "base.yaml의 env.interact_slots 상속")
    check(exp.env.queue_max_size == 100, "base.yaml의 env.queue_max_size 상속")
    check(exp.agent.vq_bits == 4, "agent 섹션이 정상 파싱된다")
    check(exp.agent.use_dsm is True, "agent.use_dsm이 train 하위로 새지 않는다")
    check(exp.agent.state_dim is None, "state_dim은 미지정 상태로 자동 추론에 위임된다")
    check(exp.train.max_episodes == 1000, "train.max_episodes 재정의 적용")


def test_overrides() -> None:
    print("\n[3] dot-path 오버라이드")
    cfg = load_config(CONFIG, overrides=["agent.lr=1e-4", "env.topology=T3", "agent.use_dsm=false"])
    exp = ExperimentConfig.from_dict(cfg)
    check(abs(exp.agent.lr - 1e-4) < 1e-12, "float 오버라이드")
    check(exp.env.topology == "T3", "문자열 오버라이드")
    check(exp.agent.use_dsm is False, "bool 오버라이드")


def test_topology_registry() -> None:
    print("\n[4] 토폴로지 레지스트리 T1~T6")
    reg = get_registry(TOPO_FILE)
    check(reg.ids() == ["T1", "T2", "T3", "T4", "T5", "T6"], "6종 토폴로지 로드")

    t1 = reg.get("T1")
    check(t1.n_aps == 3 and t1.max_stas_per_ap() == 1, "T1: 3-AP, AP당 1 STA")
    check(t1.n_overlap_stas() == 0, "T1: overlap STA 없음")

    t5 = reg.get("T5")
    check(t5.n_stas(0) == 2, "T5: AP0이 2개 STA 서빙")


def test_overlap_stas() -> None:
    print("\n[5] T3 / T6 overlap STA 조건")
    reg = get_registry(TOPO_FILE)

    t3 = reg.get("T3")
    check(t3.n_overlap_stas() == 2, "T3: overlap STA 2개")
    check(t3.is_overlap_sta(0, 1), "T3: AP0의 두 번째 STA가 overlap")
    check(t3.sta_overlap[0][1] == [1], "T3: AP0 overlap STA가 AP1과 중첩")
    check(not t3.is_overlap_sta(0, 0), "T3: 기본 STA는 비중첩")

    t6 = reg.get("T6")
    check(t6.n_overlap_stas() == 3, "T6: overlap STA 3개")
    check(t6.sta_overlap[3][1] == [1, 2], "T6: 3중 중첩 STA가 AP1/AP2와 중첩")
    check(t6.max_stas_per_ap() == 2, "T6: AP당 최대 2 STA")

    print("\n  토폴로지 요약:")
    for row in reg.summary():
        print(
            f"    {row['id']:3s} {row['name']:10s} AP={row['n_aps']} "
            f"STA={row['n_stas']:2d} maxK={row['max_stas_per_ap']} "
            f"overlap={row['n_overlap_stas']} deg={row['mean_degree']:.2f}"
        )


def test_group_resolution() -> None:
    print("\n[6] 토폴로지 그룹 선택자")
    reg = get_registry(TOPO_FILE)
    overlap_topos = reg.resolve("group:with_overlap")
    check([t.metadata["id"] for t in overlap_topos] == ["T3", "T6"], "group:with_overlap 해석")
    check(len(reg.resolve("all")) == 6, "all 선택자 해석")
    check(len(reg.resolve(["T1", "T5"])) == 2, "리스트 선택자 해석")
    check(reg.groups_of("T6") == ["high_density", "with_overlap"], "T6 소속 그룹 역조회")


def main() -> int:
    print("=" * 66)
    print("Configuration / Topology 계층 검증")
    print("=" * 66)
    test_deep_merge()
    test_config_loading()
    test_overrides()
    test_topology_registry()
    test_overlap_stas()
    test_group_resolution()

    print("\n" + "=" * 66)
    if _failures:
        print(f"실패 {len(_failures)}건:")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
