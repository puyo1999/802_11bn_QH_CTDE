from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

# 논문용 폰트 및 스타일 설정 (맑은 고딕 또는 기본 sans-serif 계열 추천)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def plot_cumulative_policy_comparison(
        results_dict: dict[str, np.ndarray],
        topology_name: str,
        event_step: int = 100,
        save_path: str = "./plot_results/"
) -> None:
    """
    [스타일 1] 누적 성공 패킷(Cumulative Successful Packets) 비교 그래프
    - 업로드된 'ultimate_5_policy_comparison_CW_Topo6.png' 형식 반영

    Args:
        results_dict: {"Policy Name": np.array([step 0~N의 누적 혹은 개별 값들]), ...}
        topology_name: 토폴로지 이름 (예: "CW_Topo6")
        event_step: 환경 변화나 이벤트가 발생한 시점 (수직선 표시용, 예: 100)
        save_path: 저장 디렉토리 경로
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)

    # 정책별 스타일 매핑 (색상 및 선 모양 정의)
    style_map = {
        "Greedy": {"color": "#ff7f0e", "linestyle": "dotted", "linewidth": 2.2},
        "Random": {"color": "#7f7f7f", "linestyle": "dashed", "linewidth": 2.2},
        "Round-Robin": {"color": "#6baed6", "linestyle": "dashdot", "linewidth": 2.2},
        "CTDE Basic": {"color": "#0072bd", "linestyle": "solid", "linewidth": 2.5},
        "CTDE Adaptive": {"color": "#2ca02c", "linestyle": "solid", "linewidth": 2.5},
    }

    for policy_name, values in results_dict.items():
        # 누적합 계산 (필요시 그대로 전달받아도 무방)
        cum_values = np.cumsum(values) if "Cumulative" not in policy_name else values

        style = style_map.get(policy_name, {"color": "black", "linestyle": "solid", "linewidth": 2})
        plt.plot(
            range(len(cum_values)),
            cum_values,
            label=policy_name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"]
        )

    # 이벤트 수직선 추가 (예: 백하울 대역폭 변화 시점)
    if event_step is not None:
        plt.axvline(x=event_step, color="#d62728", linestyle="solid", alpha=0.6, linewidth=1.5)

    plt.title(f"Comprehensive Network Capacity Comparison ({topology_name})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Simulation Time Steps", fontsize=11, fontweight="bold")
    plt.ylabel("Total Cumulative Successful Packets", fontsize=11, fontweight="bold")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="upper left")
    plt.tight_layout()

    out_file = os.path.join(save_path, f"cumulative_comparison_{topology_name}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[Plot Saved] {out_file}")


def plot_throughput_pulse_verification(
        baseline_throughput: np.ndarray,
        adaptive_throughput: np.ndarray,
        pulse_indices: list[int],
        topology_name: str,
        event_step: int = 100,
        save_path: str = "./plot_results/"
) -> None:
    """
    [스타일 2] 실시간 모드 전환 펄스 검증 (Instantaneous System Throughput) 그래프
    - 업로드된 'fig5a_pulse_good_backhaul.png' 및 'ctde_pulse_switching_timeline_CW_Topo6.png' 형식 반영

    Args:
        baseline_throughput: 기본 모델의 스텝별 처리량 배열
        adaptive_throughput: 적응형 전환 모델의 스텝별 처리량 배열
        pulse_indices: 모드 전환 펄스가 발동된 타임스텝 리스트
        topology_name: 토폴로지 이름
        event_step: 환경 전환 시점 (수직선)
        save_path: 저장 디렉토리 경로
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 5), dpi=300)
    steps = range(len(baseline_throughput))

    # 기본 및 적응형 라인 플롯
    plt.plot(steps, baseline_throughput, label="Proposed CTDE (Basic Setup)", color="#aec7e8", linewidth=1.8)
    plt.plot(steps, adaptive_throughput, label="Proposed CTDE with Fast Backhaul (Adaptive)", color="#2ca02c",
             linewidth=2.0)

    # 모드 전환 펄스 포인트 산점도 표시
    if pulse_indices:
        pulse_y = [adaptive_throughput[idx] for idx in pulse_indices if idx < len(adaptive_throughput)]
        plt.scatter(
            pulse_indices[:len(pulse_y)], pulse_y,
            color="#2ca02c", s=35, zorder=5, label="Mode Switching Pulse"
        )

    # 이벤트 수직선
    if event_step is not None:
        plt.axvline(x=event_step, color="#d62728", linestyle="dashed", linewidth=1.5)

    plt.title(f"Real-time Mode Switching Pulse Verification ({topology_name})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Simulation Time Steps", fontsize=11, fontweight="bold")
    plt.ylabel("Instantaneous System Throughput (Pkts/Step)", fontsize=11, fontweight="bold")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="lower left")
    plt.tight_layout()

    out_file = os.path.join(save_path, f"pulse_verification_{topology_name}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[Plot Saved] {out_file}")