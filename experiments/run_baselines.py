"""
experiments/run_baselines.py
3가지 핵심 통신 Baseline (Random, Greedy, Round-Robin) 구동 및 Throughput 성능 집계 스크립트
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.envs.obss_env import OBSSEnv
from src.envs.topology import FIXED_TOPOS_CW  # 고정 토폴로지 6종 로드


def run_eval(env, policy_type: str, test_steps: int = 50) -> float:
    """지정된 정책 유형에 따라 환경을 실행하고 총 성공 패킷 수를 반환"""
    env.reset()
    n_aps = env.n_aps
    total_system_packets = 0

    for step_idx in range(test_steps):
        if policy_type == "Random":
            # 1. 무작위 파라미터 조합 선택
            action = env.action_space.sample()

        elif policy_type == "Greedy":
            # 2. 이기적/공격적 기법: 최저 CW (0), 최고 CCA 임계치 (4) 강제 적용
            action = np.array([0, 4] * n_aps, dtype=np.int32)

        elif policy_type == "Round-Robin":
            # 3. 시간 분할 조율 기법: 타임스텝마다 교대로 특정 AP 1개만 공격성을 가지고, 나머지는 최고 양보 정책 유지
            action = np.zeros(n_aps * 2, dtype=np.int32)
            # 기본적으로 모든 AP는 양보 상태 (CW 최고=6, CCA 최저=0)
            for i in range(n_aps):
                action[2 * i] = 6
                action[2 * i + 1] = 0
            # 이번 차례의 AP에게만 독점적 전송 기회 위임 (CW=0, CCA=4)
            active_ap = step_idx % n_aps
            action[2 * active_ap] = 0
            action[2 * active_ap + 1] = 4

        # 환경 상호작용 진행
        _, _, _, _, info = env.step(action)

        # 시스템 전체 성공 패킷수 누적
        total_system_packets += sum(info["thp"])

    return float(total_system_packets)


def main():
    test_steps = 50
    policies = ["Random", "Greedy", "Round-Robin"]

    # 결과 저장용 딕셔너리
    results = {p: [] for p in policies}

    # ── [수정] 함수들을 호출하여 실제 Topology 인스턴스 리스트를 생성합니다 ──
    topo_instances = [topo_fn() for topo_fn in FIXED_TOPOS_CW]
    topo_names = [topo.name for topo in topo_instances]

    print("=== [Wi-Fi 8 OBSS 표준 실험] Baseline 3종 벤치마크 테스트 개시 ===")

    # ── [수정] 생성된 topo_instances를 순회합니다 ──
    for topo in topo_instances:
        print(f"\n[토폴로지 평가 중] -> {topo.name}")

        # 새롭게 정제된 4-State Joint Mode 환경 인스턴스화
        env = OBSSEnv(topology=topo, mode="joint", interact_slots=55_556, max_episodes=test_steps)

        for p in policies:
            total_packets = run_eval(env, policy_type=p, test_steps=test_steps)
            results[p].append(total_packets)
            print(f"  └─ 정책 [{p:11s}]: 누적 시스템 처리량 = {total_packets:.0f} pks")

    # ── 4️⃣ 결과 시각화 (Throughput Bar Chart 그래프 드로잉) ──
    x = np.arange(len(topo_names))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, results["Random"], width, label="Random Allocation", color="#95a5a6")
    plt.bar(x, results["Greedy"], width, label="Greedy (Aggressive OBSS-PD)", color="#e74c3c")
    plt.bar(x + width, results["Round-Robin"], width, label="Round-Robin (TDM Heuristic)", color="#3498db")

    plt.xlabel("Network Topologies", fontsize=12, fontweight='bold')
    plt.ylabel("System Cumulative Throughput (Total Successful Packets)", fontsize=12, fontweight='bold')
    plt.title("Baseline Performance Comparison across IEEE 802.11bn OBSS Scenarios", fontsize=14, fontweight='bold')
    plt.xticks(x, topo_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    output_path = "./baseline_throughput_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[실험 완료] Baseline 성능 비교 차트가 성공적으로 저장되었습니다: {output_path}")



if __name__ == "__main__":
    main()