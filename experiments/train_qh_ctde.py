"""
experiments/train_qh_ctde.py (QH-CTDE VQ & DSM 통합 고도화 버전)
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.qh_obss_env_mod1 import OBSSEnv
from src.config import load_config
from src.envs.topology import FIXED_TOPOS_CW
from experiments.plot.marl_plotter import MARLPlotter
from experiments.run_baselines import run_eval as run_baseline_eval

from src.agents.qh_ctde_agent import QHCTDEAgent

def build_agents(n_aps: int, cfg: dict) -> list[QHCTDEAgent]:
    """
    YAML 설정 파일(qh_ctde.yaml)의 train 섹션을 바탕으로
    모든 AP에 대한 QH-CTDE 에이전트 리스트를 생성합니다.
    """
    acfg = cfg["agent"]
    tcfg = cfg["train"]

    state_dim = acfg.get("state_dim", 10)
    hidden_dim = acfg.get("hidden_dim", 128)
    use_quantization = acfg.get("use_quantization", True)
    vq_bits = acfg.get("vq_bits", 4)
    use_dsm = acfg.get("use_dsm", True)
    gamma_margin = acfg.get("gamma_margin", 0.8)

    lr = acfg.get("lr", tcfg.get("learning_rate", 0.0003))
    gamma = acfg.get("gamma", 0.99)
    tau = acfg.get("tau", 0.005)

    eps_start = acfg.get("eps_start", 1.0)
    eps_end = acfg.get("eps_end", 0.01)
    eps_decay = tcfg.get("epsilon_decay", acfg.get("eps_decay", 0.995))
    device = tcfg.get("device", "cpu")

    k_stas = cfg["env"].get("k_stas", 4)

    return [
        QHCTDEAgent(
            ap_id=i,
            state_dim=state_dim,
            n_stas=k_stas,
            n_mcs=16,
            hidden_dim=hidden_dim,
            use_quantization=use_quantization,
            vq_bits=vq_bits,
            use_dsm=use_dsm,
            gamma_margin=gamma_margin,
            lr=lr,
            gamma=gamma,
            tau=tau,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            batch_size=acfg.get("batch_size", 32),
            mem_size=acfg.get("mem_size", 5000),
            device=device,
        )
        for i in range(n_aps)
    ]

# ── 1. 적응형 벡터 양자화 (Adaptive Vector Quantizer with STE) ───────────
class AdaptiveVectorQuantizer(nn.Module):
    def __init__(self, max_embeddings=256, embedding_dim=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x: torch.Tensor, bit_width: int = 4):
        """
        x: [batch_size, n_aps, obs_dim]
        bit_width: 4 (K = 2^4 = 16) 또는 8 (K = 2^8 = 256)
        """
        num_embeddings = 2 ** bit_width
        active_embedding = self.embedding.weight[:num_embeddings, :] # [K, dim]

        batch_size, n_aps, dim = x.shape
        flat_x = x.reshape(-1, dim) # [B * N, dim]

        # L2 Distance 계산
        d = torch.sum(flat_x**2, dim=1, keepdim=True) + \
            torch.sum(active_embedding**2, dim=1) - \
            2 * torch.matmul(flat_x, active_embedding.t())

        encoding_indices = torch.argmin(d, dim=1)
        quantized = active_embedding[encoding_indices].reshape(batch_size, n_aps, dim)

        # Straight-Through Estimator (STE): Forward는 양자화값, Backward는 원본 x 전달
        quantized_st = x + (quantized - x).detach()

        # VQ Loss (Codebook Loss + Commitment Loss)
        codebook_loss = nn.MSELoss()(quantized, x.detach())
        commitment_loss = nn.MSELoss()(quantized.detach(), x)
        vq_loss = codebook_loss + 0.25 * commitment_loss

        return quantized_st, vq_loss


# ── 2. QH-CTDE 전용 Attention Q-Network (Asymmetric Critic) ──────────────
class QHAttentionQNetwork(nn.Module):
    def __init__(self, obs_dim: int = 10, hidden_dim: int = 128, k_stas: int = 4):
        super().__init__()
        self.k_stas = k_stas
        # QH 모드 액션: 각 AP당 STA(K개) * MCS(16개) = K * 16 조합 (또는 분리형 헤드)
        self.n_actions_per_ap = k_stas * 16 

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer   = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions_per_ap)
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
        return self.q_head(combined) # [batch_size, n_aps, n_actions_per_ap]


# ── 3. Centralized Replay Buffer ──────────────────────────────────────────
class CentralizedReplayBuffer:
    def __init__(self, capacity: int, n_aps: int, obs_dim: int):
        self.capacity = capacity
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


def unpack_qh_action(flat_actions: list[int], k_stas: int) -> np.ndarray:
    """평탄화된 액션을 (STA, MCS) 튜플 환경 액션 배열로 변환"""
    env_actions = []
    for act in flat_actions:
        sta_idx = act // 16
        mcs_idx = act % 16
        env_actions.extend([min(sta_idx, k_stas - 1), mcs_idx])
    return np.array(env_actions, dtype=np.int32)


# ── 4. QH-CTDE 학습 메인 함수 (build_agents 활용) ────────────────────────
def train_qh_ctde(topo, cfg: dict) -> tuple[list[QHCTDEAgent], list[float]]:
    ecfg = cfg.get("env", {})
    tcfg = cfg.get("train", {})

    max_episodes = tcfg.get("max_episodes", 300)
    k_stas = ecfg.get("k_stas", 4)

    env = OBSSEnv(
        topology=topo,
        mode=ecfg.get("mode", "qh"),
        interact_slots=ecfg.get("interact_slots", 55556),
        max_episodes=1,
        k_stas_per_ap=k_stas
    )
    n_aps = env.n_aps

    # 💡 새로 구성한 build_agents 함수를 호출하여 각 AP 에이전트 생성
    agents = build_agents(n_aps, cfg)
    reward_history = []

    print(f" -> [{topo.name}] QH-CTDE (Agent Modularized VQ + DSM) 학습 개시...")

    for episode in range(max_episodes):
        obs, _ = env.reset()
        done = False
        ep_cumulative_reward = 0.0

        while not done:
            # 1. 각 AP별 에이전트가 자율적으로 행동 결정 (유효 STA 마스킹 반영)
            actions_list = []
            obs_dict = {}
            actions_dict = {}
            for i, agent in enumerate(agents):
                valid_sta_cnt = len(topo.sta_positions[i])
                act = agent.select_action(obs[i], valid_sta_cnt=valid_sta_cnt)
                actions_list.extend(act)  # [sta_0, mcs_0, sta_1, mcs_1, ...]

                # 에이전트별 저장용 딕셔너리 매핑
                obs_dict[i] = obs[i]
                actions_dict[i] = act

            env_action = np.array(actions_list, dtype=np.int32)
            next_obs, system_reward, done, _, info = env.step(env_action)

            # 성능 어느 정도 개선되는지 검증
            for i, agent in enumerate(agents):
                # 1. 버퍼에 저장
                agent.store_transition(obs[i], actions_dict[i], system_reward, next_obs[i], done)
                # 2. 매 스텝 미니배치 학습 수행
                agent.update()

            agent.decay_epsilon()

            obs = next_obs
            ep_cumulative_reward += system_reward

        reward_history.append(ep_cumulative_reward)

    return agents, reward_history

# ── 3. 학습 완료 모델 평가 함수 ──────────────────────────────────────────
def eval_qh_ctde(env, agents: list[QHCTDEAgent], topo, test_steps=200) -> float:
    obs, _ = env.reset()
    total_system_packets = 0

    for agent in agents:
        agent.actor.eval()
        agent.sta_head.eval()
        agent.mcs_head.eval()

    for _ in range(test_steps):
        actions_list = []
        for i, agent in enumerate(agents):
            valid_sta_cnt = len(topo.sta_positions[i])
            # 평가 시 탐험(Epsilon)을 0으로 고정하여 최적 정책 수행
            orig_eps = agent.eps
            agent.eps = 0.0
            act = agent.select_action(obs[i], valid_sta_cnt=valid_sta_cnt)
            agent.eps = orig_eps
            actions_list.extend(act)

        env_action = np.array(actions_list, dtype=np.int32)
        _, _, _, _, info = env.step(env_action)
        total_system_packets += sum(info["thp"])

    return float(total_system_packets)


# ── 5. 통합 메인 실행 루프 ────────────────────────────────────────────────
def main():
    cfg = load_config("configs/qh_ctde.yaml")
    test_steps = 200
    topo_instances = [topo_fn() for topo_fn in FIXED_TOPOS_CW]
    topo_names = [topo.name for topo in topo_instances]

    final_results = {
        "Random": [],
        "Greedy": [],
        "Round-Robin": [],
        "Proposed CTDE": []
    }

    all_topo_rewards = {}
    plotter = MARLPlotter(topo_names, test_steps, cfg["train"].get("max_episodes", 300))

    print("==========================================================================")
    for topo in topo_instances:
        print(f"\n[Scenario Config] 무선 통신 토폴로지 구동: {topo.name}")
        env = OBSSEnv(topology=topo, mode="qh", interact_slots=55556, max_episodes=test_steps)

        # 1. 전통적 통신 Baseline 측정
        for base_policy in ["Random", "Greedy", "Round-Robin"]:
            score = run_baseline_eval(env, policy_type=base_policy, test_steps=test_steps)
            final_results[base_policy].append(score)
            print(f"  └─ 대조군 [{base_policy:11s}] 측정 완료: {score:.0f} pks")

        # 2. QH-CTDE 학습 구동 (build_agents 내부 포함)
        trained_agents, rew_history = train_qh_ctde(topo, cfg)
        all_topo_rewards[topo.name] = rew_history

        # 3. 모델 성능 평가
        qh_score = eval_qh_ctde(env, trained_agents, topo, test_steps=test_steps)
        final_results["Proposed CTDE"].append(qh_score)
        print(f"  ▶ 종합 성과 확인 -> Proposed CTDE 스코어: {qh_score:.0f} 패킷 성공")

    print("\n==========================================================================")
    print("★ [시각화 1] 토폴로지별 학습 리워드 수렴 곡선 생성 중...")
    plotter.plot_reward_convergence(all_topo_rewards)

    print("\n==========================================================================")
    print("★ [시각화 2] 논문 게재용 4종 멀티 바 차트 생성 중...")
    plotter.plot_final_bar_chart(final_results)
    print("==========================================================================")

if __name__ == "__main__":
    main()
