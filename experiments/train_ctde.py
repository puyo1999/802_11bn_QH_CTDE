"""
experiments/train_ctde.py (Reward Trace Graph 고도화 버전)
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 상위 경로를 path에 추가하여 패키지 임포트 유연성 확보
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from src.envs.obss_env import OBSSEnv
#from src.envs.ctde_obss_env import OBSSEnv
#from src.envs.qh_obss_env import OBSSEnv
from src.envs.qh_obss_env_mod1 import OBSSEnv
from src.envs.topology import FIXED_TOPOS_CW
from experiments.run_baselines import run_eval as run_baseline_eval

# 신설된 plot 폴더에서 플로터 클래스 로드
from experiments.plot.marl_plotter import MARLPlotter

# ── [중략: CentralizedReplayBuffer & MAAttentionQNetwork & unpack_action 은 이전과 동일] ──
# (코드 결합 시 이전 본문의 Buffer와 Network 클래스를 그대로 위 공간에 두시면 됩니다.)

class CentralizedReplayBuffer:
    def __init__(self, capacity: int, n_aps: int, obs_dim: int):
        self.capacity = capacity
        self.n_aps = n_aps
        self.obs_dim = obs_dim
        self.pointer = 0
        self.size = 0
        self.obs_buf = np.zeros((capacity, n_aps, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, n_aps), dtype=np.int64)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_aps, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

    def store(self, obs, acts, rew, next_obs, done):
        self.obs_buf[self.pointer] = obs
        self.act_buf[self.pointer] = acts
        self.rew_buf[self.pointer] = rew
        self.next_obs_buf[self.pointer] = next_obs
        self.done_buf[self.pointer] = float(done)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs_buf[indices]),
            torch.LongTensor(self.act_buf[indices]),
            torch.FloatTensor(self.rew_buf[indices]),
            torch.FloatTensor(self.next_obs_buf[indices]),
            torch.FloatTensor(self.done_buf[indices])
        )

class MAAttentionQNetwork(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden_dim: int = 32, n_actions: int = 35):
        super().__init__()
        self.n_actions = n_actions
        self.feature_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer   = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, joint_obs: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, n_aps, _ = joint_obs.size()
        feat = self.feature_net(joint_obs)
        Q = self.query_layer(feat)
        K = self.key_layer(feat)
        V = self.value_layer(feat)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        mask = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        combined = torch.cat([feat, context], dim=-1)
        return self.q_head(combined)

def unpack_action(action_35: int, n_aps: int) -> np.ndarray:
    env_actions = []
    for act in action_35:
        env_actions.extend([act // 5, act % 5])
    return np.array(env_actions, dtype=np.int32)


# ── [수정] Reward History 기록을 리턴하도록 변경 ──────────────────────────
def train_proposed_ctde(topo, max_episodes=100, batch_size=32) -> tuple[MAAttentionQNetwork, list[float]]:
    """중앙 집중식 학습을 수행하며 각 에피소드별 누적 Reward 추이를 추적 반환"""
    env = OBSSEnv(topology=topo, mode="joint", interact_slots=55_556, max_episodes=1)
    n_aps = env.n_aps
    obs_dim = env.observation_space.shape[1]

    # ─── [수정 후] 차원 불일치(3,9 -> 3,4) 해결 코드 ───────────────────
    # 환경이 실제로 리턴하는 observation의 마지막 차원 크기(9)를 동적으로 읽어옴
    init_obs, _ = env.reset()
    actual_obs_dim = init_obs.shape[-1]  # <- 9차원을 동적으로 획득


    adj_matrix = np.zeros((n_aps, n_aps), dtype=np.float32)
    for i in range(n_aps):
        adj_matrix[i, i] = 1.0
        for nbr in topo.neighbors[i]: adj_matrix[i, nbr] = 1.0
    adj_tensor = torch.FloatTensor(adj_matrix)

    online_net = MAAttentionQNetwork(obs_dim=actual_obs_dim)
    target_net = MAAttentionQNetwork(obs_dim=actual_obs_dim)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
    buffer = CentralizedReplayBuffer(capacity=10000, n_aps=n_aps, obs_dim=actual_obs_dim)

    epsilon = 1.0
    eps_decay = 0.95
    eps_min = 0.05

    # 리워드 추이 저장 배열
    reward_history = []
    print(f" -> [{topo.name}] Proposed CTDE 가치 믹서 신경망 학습 개시...")

    for episode in range(max_episodes):
        obs, _ = env.reset()
        done = False
        ep_cumulative_reward = 0.0

        while not done:
            if random.random() < epsilon:
                #local_actions = [random.randint(0, 34) for _ in range(n_aps)]
                # 💡 [수정 1] Random 탐험 시에도 실제 존재하는 STA 범위 내 액션만 선택
                local_actions = []
                for i in range(n_aps):
                    valid_sta_cnt = len(topo.sta_positions[i])
                    # target_sta (act // 5) 가 valid_sta_cnt 보다 작은 유효 액션만 필터링
                    valid_actions = [act for act in range(35) if (act // 5) < valid_sta_cnt]
                    local_actions.append(random.choice(valid_actions))
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    q_vals = online_net(obs_t, adj_tensor).squeeze(0)
                    #local_actions = q_vals.argmax(dim=-1).tolist()
                    # 💡 [수정 2] Exploitation 시 무효 STA 액션 Q-Value 마스킹 (-1e9)
                    masked_q_vals = q_vals.clone()
                    for i in range(n_aps):
                        valid_sta_cnt = len(topo.sta_positions[i])
                        for act in range(35):
                            if (act // 5) >= valid_sta_cnt:
                                masked_q_vals[i, act] = -1e9  # 선택 불가능하도록 차단

                    local_actions = masked_q_vals.argmax(dim=-1).tolist()

            env_action = unpack_action(local_actions, n_aps)
            next_obs, system_reward, done, _, _ = env.step(env_action)

            buffer.store(obs, local_actions, system_reward, next_obs, done)
            obs = next_obs
            ep_cumulative_reward += system_reward

            if buffer.size > batch_size:
                b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(batch_size)
                q_values = online_net(b_obs, adj_tensor)
                chosen_q = q_values.gather(2, b_act.unsqueeze(-1)).squeeze(-1)
                q_tot_curr = chosen_q.sum(dim=-1)

                with torch.no_grad():
                    next_q_values = target_net(b_next_obs, adj_tensor)
                    max_next_q = next_q_values.max(dim=-1)[0]
                    q_tot_next = max_next_q.sum(dim=-1)
                    target_y = b_rew + 0.95 * q_tot_next * (1.0 - b_done)

                loss = nn.MSELoss()(q_tot_curr, target_y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()

        if episode % 5 == 0:
            target_net.load_state_dict(online_net.state_dict())
        epsilon = max(eps_min, epsilon * eps_decay)

        # 에피소드별 리워드 저장
        reward_history.append(ep_cumulative_reward)

    return online_net, reward_history

def eval_proposed_ctde(env, trained_model, topo, test_steps=50) -> float:
    obs, _ = env.reset()
    n_aps = env.n_aps
    total_system_packets = 0
    adj_matrix = np.zeros((n_aps, n_aps), dtype=np.float32)
    for i in range(n_aps):
        adj_matrix[i, i] = 1.0
        for nbr in topo.neighbors[i]: adj_matrix[i, nbr] = 1.0
    adj_tensor = torch.FloatTensor(adj_matrix)

    trained_model.eval()
    for _ in range(test_steps):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            q_vals = trained_model(obs_t, adj_tensor).squeeze(0)
            #local_actions = q_vals.argmax(dim=-1).tolist()

            # 💡 [수정] 평가 단계 Action Masking 적용
            masked_q_vals = q_vals.clone()
            for i in range(n_aps):
                valid_sta_cnt = len(topo.sta_positions[i])
                for act in range(35):
                    if (act // 5) >= valid_sta_cnt:
                        masked_q_vals[i, act] = -1e9

            local_actions = masked_q_vals.argmax(dim=-1).tolist()

        env_action = unpack_action(local_actions, n_aps)
        next_obs, _, _, _, info = env.step(env_action)
        total_system_packets += sum(info["thp"])
        obs = next_obs
    return float(total_system_packets)


# ── 6. 통합 메인 루프 및 시각화 고도화 ────────────────────────────────────
'''
def main():
    test_steps = 50
    max_episodes = 100
    topo_instances = [topo_fn() for topo_fn in FIXED_TOPOS_CW]
    topo_names = [topo.name for topo in topo_instances]

    final_results = {"Random": [], "Greedy": [], "Round-Robin": [], "Proposed CTDE": []}

    # [추가] 각 토폴로지별 학습 리워드 추이를 저장할 공간
    all_topo_rewards = {}

    print("==========================================================================")
    for topo in topo_instances:
        print(f"\n[Scenario Config] 통신 토폴로지 구동: {topo.name}")
        env = OBSSEnv(topology=topo, mode="joint", interact_slots=55_556, max_episodes=test_steps)

        # 1. Baseline 측정
        for base_policy in ["Random", "Greedy", "Round-Robin"]:
            score = run_baseline_eval(env, policy_type=base_policy, test_steps=test_steps)
            final_results[base_policy].append(score)

        # 2. Proposed 학습 진행 및 리워드 히스토리 수집
        trained_agent_net, rew_history = train_proposed_ctde(topo, max_episodes=max_episodes)
        all_topo_rewards[topo.name] = rew_history

        # 3. 모델 평가
        ctde_score = eval_proposed_ctde(env, trained_agent_net, topo, test_steps=test_steps)
        final_results["Proposed CTDE"].append(ctde_score)
        print(f"  ▶ 종합 성과 확인 -> Proposed CTDE 스코어: {ctde_score:.0f} 패킷 성공")

    print("\n==========================================================================")
    print("★ [그래프 생성 작업 1] 누적 Reward 추이 수렴 곡선 (Time/Episode 축) 드로잉...")

    # ── 6분할 Subplot 구조로 시나리오별 수렴성 그래프 생성 ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(topo_names):
        ax = axes[idx]
        rewards = all_topo_rewards[name]

        # 원본 데이터와 가독성을 위한 이동 평균(Smooth) 데이터 병해 출력
        ax.plot(rewards, color='#2ecc71', alpha=0.3, label='Raw Reward')

        # 5개 에피소드 윈도우 이동 평균 계산
        smoothed_rew = np.convolve(rewards, np.ones(5)/5, mode='valid')
        ax.plot(range(4, max_episodes), smoothed_rew, color='#27ae60', linewidth=2, label='5-Ep Moving Avg')

        ax.set_title(f"Convergence: {name}", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx >= 3:
            ax.set_xlabel("Training Episodes", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("System Total Reward", fontsize=10)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("Proposed MARL (CTDE) Learning Curves Across 802.11bn OBSS Topologies", fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    reward_graph_path = "./ctde_reward_convergence.png"
    plt.savefig(reward_graph_path, dpi=300)
    print(f"  └─ 수렴 추이 그래프 저장 완료: {reward_graph_path}")

    # ── [기존] 4종 처리량 비교 바 차트 드로잉 및 저장 조치 ──
    x = np.arange(len(topo_names))
    width = 0.18
    plt.figure(figsize=(13, 6.5))
    plt.bar(x - width*1.5, final_results["Random"], width, label="Random Allocation", color="#b2bec3")
    plt.bar(x - width*0.5, final_results["Greedy"], width, label="Greedy (Aggressive)", color="#ff7675")
    plt.bar(x + width*0.5, final_results["Round-Robin"], width, label="Round-Robin (TDM)", color="#74b9ff")
    plt.bar(x + width*1.5, final_results["Proposed CTDE"], width, label="Proposed Basic CTDE", color="#2ecc71", edgecolor="#27ae60")
    plt.xlabel("Standard Wi-Fi 8 Network Topologies", fontsize=12, fontweight='bold')
    plt.ylabel("System Cumulative Throughput (Total Successful Packets)", fontsize=12, fontweight='bold')
    plt.title("Performance Verification: Proposed CTDE Framework vs. Communication Baselines (802.11bn)", fontsize=14, fontweight='bold')
    plt.xticks(x, topo_names)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()

    final_output_path = "./final_throughput_comparison.png"
    plt.savefig(final_output_path, dpi=300)
    print(f"  └─ 최종 성능 비교 바 차트 저장 완료: {final_output_path}")

'''
def main():
    test_steps = 200
    max_episodes = 300

    # 1. 팩토리 함수들을 호출하여 실제 Topology 인스턴스 리스트 생성
    topo_instances = [topo_fn() for topo_fn in FIXED_TOPOS_CW]
    topo_names = [topo.name for topo in topo_instances]

    # ── [확인] 에러의 원인이었던 전역 결과 저장소 딕셔너리 선언 확실히 명시 ──
    final_results = {
        "Random": [],
        "Greedy": [],
        "Round-Robin": [],
        "Proposed CTDE": []
    }

    # 각 토폴로지별 학습 리워드 추이를 저장할 공간
    all_topo_rewards = {}
    # 클래스 객체 깔끔하게 선언
    plotter = MARLPlotter(topo_names, test_steps, max_episodes)

    print("==========================================================================")
    for topo in topo_instances:
        print(f"\n[Scenario Config] 통신 토폴로지 구동: {topo.name}")

        # 매 에피소드(스텝)마다 0.5초(55,556 slots)씩 시뮬레이션하는 환경 인스턴스
        env = OBSSEnv(topology=topo, mode="joint", interact_slots=55_556, max_episodes=test_steps)

        # 2. 기존 통신 Baseline 3종 측정 구동 및 결과 축적
        for base_policy in ["Random", "Greedy", "Round-Robin"]:
            score = run_baseline_eval(env, policy_type=base_policy, test_steps=test_steps)
            final_results[base_policy].append(score)
            print(f"  └─ 대조군 [{base_policy:11s}] 측정 완료: {score:.0f} pks")

        # 3. Proposed 기법 중앙 학습 진행 및 에피소드별 리워드 히스토리 수집
        trained_agent_net, rew_history = train_proposed_ctde(topo, max_episodes=max_episodes)
        all_topo_rewards[topo.name] = rew_history

        # 분리된 plot 폴더 내부 메서드로 그래프 위임 (unpack_action 전달)
        print(f"  ▶ [{topo.name}] 5종 정책 통합 누적 시계열 시뮬레이션 개시...")
        plotter.plot_all_5_policies_cumulative(env, trained_agent_net, topo, unpack_action_fn=unpack_action)

        env.is_harsh_interference = False
        # 4. 학습 완료된 모델 분산 구동 및 최종 성능(Throughput) 평가
        ctde_score = eval_proposed_ctde(env, trained_agent_net, topo, test_steps=test_steps)
        final_results["Proposed CTDE"].append(ctde_score)
        print(f"  ▶ 종합 성과 확인 -> Proposed CTDE 스코어: {ctde_score:.0f} 패킷 성공")

    print("\n==========================================================================")

    # 루프 종료 후 종합 장표 호출
    print("★ [시각화 1] 6종 토폴로지별 학습 리워드 수렴 곡선 (ctde_reward_convergence.png) 생성 중...")
    plotter.plot_reward_convergence(all_topo_rewards)

    # ─── ⭐ [이 위치에 추가] 펄스 스파이크 확인용 실시간 타임라인 단독 플롯 호출 ───
    print(f"  ▶ [{topo.name}] 실시간 모드 스위칭 펄스 검증 그래프 생성...")
    plotter.plot_mode_switching_timeline(env, trained_agent_net, topo, unpack_action_fn=unpack_action)
    # ───────────────────────────────────────────────────────────────────────────

    print("\n==========================================================================")
    print("★ [시각화 2] 논문 게재용 4종 멀티 바 차트 (final_throughput_comparison.png) 생성 중...")
    plotter.plot_final_bar_chart(final_results)

    print("==========================================================================")

if __name__ == "__main__":
    main()