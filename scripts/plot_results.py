import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def plot_convergence(mode_list, topo_name, max_episodes, window_size=10):
    plt.figure(figsize=(10, 6))

    # 여러 알고리즘 모드(예: ['drlca_fc', 'Greedy', 'Random'])를 순회하며 그리기
    for mode in mode_list:
        # 1. 수정된 파일명 포맷(*_run*_rewards.npy)에 맞게 패턴 지정
        file_pattern = f"./results/{mode}/{topo_name}_run*_rewards.npy"
        file_list = glob.glob(file_pattern)

        if not file_list:
            print(f"[{mode}] 데이터를 찾을 수 없습니다: {file_pattern}")
            continue

        # 모든 Run 데이터 로드 -> shape: (num_runs, max_episodes)
        run_data = [np.load(f) for f in file_list]
        run_data = np.array(run_data)

        # Run 축(axis=0)을 기준으로 에피소드별 평균과 표준편차 계산
        mean_rewards = np.mean(run_data, axis=0)
        std_rewards = np.std(run_data, axis=0)
        episodes = np.arange(len(mean_rewards))

        # 💡 [추가] 리워드 변동이 심할 때 추세를 잘 보기 위해 이동 평균(Moving Average) 적용
        if window_size > 1:
            # 원래 데이터의 경계면을 유지하기 위해 'same' 모드 사용 또는 보정
            smoothed_mean = np.convolve(mean_rewards, np.ones(window_size) / window_size, mode='same')
            smoothed_std = np.convolve(std_rewards, np.ones(window_size) / window_size, mode='same')
        else:
            smoothed_mean = mean_rewards
            smoothed_std = std_rewards

        # 2. 평균 수렴 곡선 그리기 (이동평균 데이터 사용)
        line, = plt.plot(episodes, smoothed_mean, label=f"{mode}", linewidth=2)

        # 3. 표준편차만큼 음영(Shaded Area)으로 신뢰구간 표시
        plt.fill_between(
            episodes,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            alpha=0.15,
            color=line.get_color()
        )

    # 그래프 세부 설정
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.title(f"Convergence Curve - {topo_name}", fontsize=14, fontweight='bold')
    plt.xlim(0, max_episodes)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()

    # 4. 결과 그래프 시각화 및 이미지 파일 저장
    os.makedirs("./plots", exist_ok=True)
    save_path = f"./plots/{topo_name}_convergence.png"
    plt.savefig(save_path, dpi=300)
    print(f"[성공] 그래프가 저장되었습니다: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────
# 스크립트 단독 실행을 위한 코드
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 비교하고 싶으신 모드(폴더명) 리스트를 작성하세요.
    # 예: 학습 기반 모드('drlca_fc' 또는 'drlca_attn')와 'Random' 모드 비교
    modes = ["drlca_fc", "Random"]

    # 그래프를 띄울 대상 토폴로지 이름 설정
    target_topology = "CW_Topo1"

    # 최대 에피소드 길이 설정 (600개 에피소드)
    max_eps = 600

    plot_convergence(mode_list=modes, topo_name=target_topology, max_episodes=max_eps, window_size=10)