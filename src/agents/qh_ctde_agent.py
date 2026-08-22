"""
src/agents/qh_ctde_agent.py (다차원 액션 버퍼 및 STE VQ 학습 고도화 버전)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QHCTDEAgent:
    """
    IEEE 802.11bn MAPC Coordinated Scheduling 환경을 위한 QH-CTDE 에이전트.
    - State Dim: 10 (환경 실측치 반영)
    - Action: MultiDiscrete ([STA 선택, MCS 선택])
    - VQ (Vector Quantization): 시그널링 오버헤드 제약을 위한 코드북 양자화 (STE 적용)
    - DSM (Dynamic Switching Module): 채널 상태 변화에 따른 모드 적응형 제어
    """

    def __init__(
        self,
        ap_id: int,
        state_dim: int = 10,
        n_stas: int = 4,
        n_mcs: int = 16,
        hidden_dim: int = 128,
        use_quantization: bool = True,
        vq_bits: int = 4,
        use_dsm: bool = True,
        gamma_margin: float = 0.8,
        lr: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        batch_size: int = 32,
        mem_size: int = 5000,
        device: str = "cpu",
    ):
        self.ap_id = ap_id
        self.state_dim = state_dim
        self.n_stas = n_stas
        self.n_mcs = n_mcs
        self.gamma = gamma
        self.tau = tau
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.use_quantization = use_quantization
        self.vq_bits = vq_bits
        self.codebook_size = 2 ** vq_bits  # 4-bit -> 16개 코드북 벡터
        self.use_dsm = use_dsm
        self.gamma_margin = gamma_margin

        # ── 신경망 구조 (Actor 네트워크) ──
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        # 액션 출력 헤드 (STA 선택지 K개 + MCS 선택지 16개)
        self.sta_head = nn.Linear(hidden_dim, n_stas).to(self.device)
        self.mcs_head = nn.Linear(hidden_dim, n_mcs).to(self.device)

        # 벡터 양자화(VQ) 코드북 정의 (학습 가능한 파라미터)
        if self.use_quantization:
            self.codebook = nn.Parameter(
                torch.randn(self.codebook_size, hidden_dim).to(self.device)
            )

        # 최적화기 설정
        param_groups = [
            {"params": self.actor.parameters()},
            {"params": self.sta_head.parameters()},
            {"params": self.mcs_head.parameters()},
        ]
        if self.use_quantization:
            param_groups.append({"params": [self.codebook]})

        self.optimizer = optim.Adam(param_groups, lr=lr)

        # ── 다차원 액션 호환 자체 메모리 버퍼 (deque) ──
        self.memory = deque(maxlen=mem_size)

    def select_action(self, state: np.ndarray, valid_sta_cnt: int = 4) -> np.ndarray:
        """ε-greedy 및 유효 STA 마스킹을 적용한 액션 선택 ([STA, MCS])"""
        if random.random() < self.eps:
            sta_act = random.randrange(valid_sta_cnt)
            mcs_act = random.randrange(self.n_mcs)
            return np.array([sta_act, mcs_act], dtype=np.int32)

        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.actor(s_t)

            # VQ(Vector Quantization) 적용 로직 (추론 시)
            if self.use_quantization:
                flat_feat = feat.squeeze(0)
                distances = torch.sum((flat_feat.unsqueeze(0) - self.codebook) ** 2, dim=1)
                idx = torch.argmin(distances)
                feat = self.codebook[idx].unsqueeze(0)

            sta_logits = self.sta_head(feat)
            mcs_logits = self.mcs_head(feat)

            # 유효하지 않은 STA 범위 마스킹 (-1e9 부여)
            if valid_sta_cnt < self.n_stas:
                sta_logits[0, valid_sta_cnt:] = -1e9

            sta_act = sta_logits.argmax(dim=-1).item()
            mcs_act = mcs_logits.argmax(dim=-1).item()

        return np.array([sta_act, mcs_act], dtype=np.int32)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """스텝별 트랜지션 저장 (액션 배열 그대로 보존)"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self) -> float | None:
        """Replay Buffer에서 미니배치를 샘플링하여 정책 경사 역방향 전파 수행"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([item[0] for item in batch])).to(self.device)
        actions = np.array([item[1] for item in batch])  # (B, 2) 형태의 배열
        sta_a_t = torch.LongTensor(actions[:, 0]).to(self.device)
        mcs_a_t = torch.LongTensor(actions[:, 1]).to(self.device)
        rewards = torch.FloatTensor([item[2] for item in batch]).to(self.device)

        # 보상 정규화 (학습 안정화)
        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 순전파 (그라디언트 유지)
        feat = self.actor(states)

        if self.use_quantization:
            # Straight-Through Estimator (STE) 적용
            flat_feat = feat
            distances = torch.sum((flat_feat.unsqueeze(1) - self.codebook.unsqueeze(0)) ** 2, dim=-1)
            indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[indices]
            feat = feat + (quantized - feat).detach()

        sta_logits = self.sta_head(feat)
        mcs_logits = self.mcs_head(feat)

        sta_dist = torch.distributions.Categorical(logits=sta_logits)
        mcs_dist = torch.distributions.Categorical(logits=mcs_logits)

        sta_log_probs = sta_dist.log_prob(sta_a_t)
        mcs_log_probs = mcs_dist.log_prob(mcs_a_t)

        # 정책 경사 손실 함수
        loss = -(sta_log_probs * rewards + mcs_log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) +
            list(self.sta_head.parameters()) +
            list(self.mcs_head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "sta_head": self.sta_head.state_dict(),
            "mcs_head": self.mcs_head.state_dict(),
            "eps": self.eps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.sta_head.load_state_dict(ckpt["sta_head"])
        self.mcs_head.load_state_dict(ckpt["mcs_head"])
        self.eps = ckpt.get("eps", self.eps_end)