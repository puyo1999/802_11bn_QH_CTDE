import numpy as np
import matplotlib.pyplot as plt


def plot_topology_tradeoffs():
    alpha = 0.5
    beta = 0.5
    D_0 = 50.0  # VQ 왜곡 스케일
    R_max = 4.0  # 최대 비트율

    # tau_bh 범위 설정 (0 ~ 10)
    tau_bh = np.linspace(0.1, 10.0, 600)

    # 토폴로지별 파라미터 (Topo 1: 여유로움, Topo 3: 보통, Topo 6: 밀집/가혹)
    # 임계값 tau_bh가 커질수록 -> 오프로딩 시도율이 감소함
    topologies = {
        "Topo 1 (Sparse / Low Interference)": {
            "lambda_0": 0.5, "mu_bh": 1.2, "color": "#2ca02c", "optimal": 3.2
        },
        "Topo 3 (Moderate / Default)": {
            "lambda_0": 0.7, "mu_bh": 1.1, "color": "#1f77b4", "optimal": 4.8
        },
        "Topo 6 (Dense / Harsh Mode)": {
            "lambda_0": 0.9, "mu_bh": 1.0, "color": "#d62728", "optimal": 6.5
        }
    }

    plt.figure(figsize=(9, 6))

    for name, params in topologies.items():
        l_0 = params["lambda_0"]
        mu = params["mu_bh"]
        c = params["color"]
        opt_tau = params["optimal"]

        # 1. VQ 양자화 오차 비용: 임계값이 높을수록(로컬 처리가 많을수록) 오차가 커짐
        # cost_vq 가증치 함수 (tau_bh에 대해 증가 함수)
        cost_vq = alpha * D_0 * (2.0 ** (-2.0 * R_max * (1.0 - (tau_bh / 10.0))))
        # 정교한 VQ Cost: tau가 커지면 VQ bit가 줄어들어 Distortion 증가
        cost_vq = alpha * D_0 * np.exp(-0.4 * (10.0 - tau_bh))

        # 2. 백홀 대기열 지연 비용: 임계값이 낮을수록(오프로딩 남발) 백홀 폭발
        # tau_bh가 opt_tau보다 작으면 대기열 폭발, 크면 안정화되지만 VQ 폭발
        # M/M/1 기반 모사 텀
        offload_prob = 1.0 / (1.0 + np.exp(1.2 * (tau_bh - opt_tau)))
        denominator = np.maximum(mu - l_0 * offload_prob, 1e-3)
        cost_delay = beta * (l_0 * offload_prob) / denominator * 5.0

        # 3. 총 시스템 손실 (Convex 형태 유도)
        cost_total = cost_vq + cost_delay

        # 최소점 찾기
        min_idx = np.argmin(cost_total)
        tau_star_empirical = tau_bh[min_idx]

        # 곡선 플로팅
        plt.plot(tau_bh, cost_total, color=c, linewidth=2.5, label=f"{name}")

        # 최적점 마커 및 수직선 표시
        plt.axvline(x=tau_star_empirical, color=c, linestyle='--', linewidth=1.5, alpha=0.8)
        plt.scatter([tau_star_empirical], [cost_total[min_idx]], color=c, s=120, edgecolor='black', zorder=5)

    plt.title('Topology-wise Theoretical Trade-off Analysis ($\mathcal{C}_{total}$)', fontsize=14, fontweight='bold')
    plt.xlabel('Backhaul Quality Threshold ($\tau_{bh}$)', fontsize=12)
    plt.ylabel('Total System Cost ($\mathcal{C}_{total}$)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(0, 10)
    plt.ylim(0, max(cost_total) * 1.2 if 'cost_total' in locals() else 100)

    plt.tight_layout()
    plt.savefig('topology_tradeoff_comparison.png', dpi=300)
    print("Corrected graph saved successfully!")


if __name__ == '__main__':
    plot_topology_tradeoffs()