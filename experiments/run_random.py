# run_random.py (프로젝트 루트에 생성)
import os
import numpy as np
import yaml
from src.envs.obss_env import OBSSEnv
from src.envs.topology import Topology  # 토폴로지 로드용 클래스 (실제 명칭에 맞게 수정)


def run_random_baseline(topo_name, num_runs=5):
    # 1. 설정 파일 로드
    with open("configs/drlca_fc.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ecfg = cfg["env"]
    mode = "cw"  # 혹은 'cca', 'joint' 중 테스트하고 싶은 액션 타입 지정

    # 2. 결과 저장 디렉토리 생성
    save_dir = f"./results/Random"
    os.makedirs(save_dir, exist_ok=True)

    # 가상의 토폴로지 객체 생성 (기존 train.py의 토폴로지 생성 로직 적용)
    # 예: topo = Topology.load_by_name(topo_name)
    topo = Topology(name=topo_name)  # 본인의 토폴로지 초기화 코드에 맞추세요.

    print(f"▶ Random Baseline 시작 ({topo_name} / Mode: {mode})")

    for run_id in range(num_runs):
        # 환경 생성 (Seed 세팅으로 재현성 확보)
        env = OBSSEnv(
            topology=topo,
            mode=mode,
            interact_slots=ecfg["interact_slots"],
            max_episodes=ecfg["max_episodes"],
            seed=ecfg["seed"] + run_id,
            traffic_type=ecfg.get("traffic_type", "saturated")
        )

        reward_history = []
        obs, _ = env.reset()

        for episode in range(ecfg["max_episodes"]):
            # --- [핵심] Gymnasium의 내장 함수로 무작위 Action 샘플링 ---
            actions = env.action_space.sample()

            # 환경 step 진행
            next_obs, reward, done, _, info = env.step(actions)

            reward_history.append(reward)
            obs = next_obs

        # .npy 파일로 저장 (추후 plot_results.py에서 불러올 수 있도록 포맷 동기화)
        file_path = f"{save_dir}/{topo_name}_run{run_id}.npy"
        np.save(file_path, np.array(reward_history))
        print(f"   [Run {run_id}] 완료 -> {file_path}")


if __name__ == "__main__":
    # 테스트할 토폴로지 이름 지정
    run_random_baseline(topo_name="CW_Topo1", num_runs=5)