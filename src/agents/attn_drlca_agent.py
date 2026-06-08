"""
src/agents/attn_drlca_agent.py
Attention-DRLCA: Self-Attention Fusion + LSTM DDQN 분산 에이전트

논문 Algorithm 1을 확장:
  - FCFusionNet → AttentionFusionNet
  - 가변 이웃 수 처리 (padding_mask)
  - attn_weights 로그 저장 (해석 가능성)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks.fusion_attention import AttentionFusionNet
from src.networks.ddqn import DDQNNet
from src.utils.replay_buffer import ReplayBuffer


class AttnDRLCAAgent:
    """
    단일 AP를 제어하는 Attention-DRLCA 에이전트.
    분산 학습: 각 AP가 독립적인 에이전트 인스턴스를 보유.
    """

    def __init__(
        self,
        ap_id:       int,
        state_dim:   int   = 6,
        fused_dim:   int   = 8,
        n_actions:   int   = 7,       # CW: 7, CCA: 5
        history_len: int   = 10,      # T=10 (논문)
        max_neighbors: int = 5,
        hidden_dim:  int   = 32,
        n_heads:     int   = 2,
        lr:          float = 5e-4,    # 논문 Table II: 0.0005
        gamma:       float = 0.9,     # 논문 Table II
        tau:         float = 0.004,   # soft update
        eps_start:   float = 1.0,
        eps_end:     float = 0.01,
        eps_decay:   float = 0.99,
        batch_size:  int   = 32,
        mem_size:    int   = 500,
        device:      str   = "cpu",
        dueling:     bool  = False,
    ):
        self.ap_id       = ap_id
        self.state_dim   = state_dim
        self.fused_dim   = fused_dim
        self.n_actions   = n_actions
        self.history_len = history_len
        self.gamma       = gamma
        self.tau         = tau
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.batch_size  = batch_size
        self.device      = torch.device(device)

        # ── 네트워크 ────────────────────────────────────────────
        self.fusion = AttentionFusionNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            out_dim=fused_dim,
        ).to(self.device)

        self.ddqn = DDQNNet(
            fused_dim=fused_dim,
            n_actions=n_actions,
            dueling=dueling,
        ).to(self.device)

        # 전체 파라미터 통합 최적화
        self.optimizer = optim.Adam(
            list(self.fusion.parameters()) +
            list(self.ddqn.online.parameters()),
            lr=lr,
        )

        # ── 리플레이 버퍼 ────────────────────────────────────────
        self.memory = ReplayBuffer(mem_size)

        # ── 히스토리 큐 ─────────────────────────────────────────
        # 논문: 과거 T step의 fused info를 시계열로 사용
        self._fused_history: deque[np.ndarray] = deque(
            [np.zeros(fused_dim)] * history_len, maxlen=history_len
        )

        # ── 로그 ────────────────────────────────────────────────
        self._last_attn: np.ndarray | None = None  # (N,) 헤드 평균 attn

    # ── public API ─────────────────────────────────────────────

    def fuse(
        self,
        self_state:      np.ndarray,          # (state_dim,)
        neighbor_states: list[np.ndarray],    # list of (state_dim,)
    ) -> np.ndarray:                          # (fused_dim,)
        """
        현재 상태를 fusion하고 히스토리 큐에 추가.
        beacon frame 수신 후 매 interact interval마다 호출.
        """
        s   = torch.tensor(self_state, dtype=torch.float32,
                           device=self.device).unsqueeze(0)       # (1, D)

        if neighbor_states:
            n = torch.tensor(
                np.stack(neighbor_states), dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)                                        # (1, N, D)
            mask = torch.ones(1, len(neighbor_states),
                              dtype=torch.bool, device=self.device)
        else:
            # 이웃이 없는 경우 (고립 AP)
            n    = torch.zeros(1, 1, self.state_dim,
                               dtype=torch.float32, device=self.device)
            mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

        self.fusion.eval()
        with torch.no_grad():
            fused, attn = self.fusion(s, n, mask)                 # (1,F), (1,h,1,N)

        fused_np = fused.squeeze(0).cpu().numpy()
        self._last_attn = attn.mean(dim=1).squeeze().cpu().numpy()
        self._fused_history.append(fused_np)
        return fused_np

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy 행동 선택. obs는 fused history (T, fused_dim)."""
        if random.random() < self.eps:
            return random.randrange(self.n_actions)

        t = torch.tensor(obs, dtype=torch.float32,
                         device=self.device).unsqueeze(0)    # (1, T, F)
        self.ddqn.eval()
        with torch.no_grad():
            q, _ = self.ddqn(t)
        return int(q.argmax(dim=-1).item())

    def store(
        self,
        obs:      np.ndarray,   # (T, fused_dim)
        action:   int,
        reward:   float,
        next_obs: np.ndarray,   # (T, fused_dim)
        done:     bool,
    ) -> None:
        self.memory.push(obs, action, reward, next_obs, done)

    def learn(self) -> float | None:
        """
        미니배치 샘플링 후 DDQN 업데이트.
        Returns: loss 값 (float) or None (버퍼 부족)
        """
        if len(self.memory) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.memory.sample(
            self.batch_size, self.device
        )
        # obs: (B, T, F), actions: (B,), rewards: (B,), dones: (B,)

        # ── Current Q ─────────────────────────────────────────
        self.ddqn.train()
        q_vals, _ = self.ddqn(obs)                              # (B, n_actions)
        q_curr = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # ── DDQN Target Q ─────────────────────────────────────
        with torch.no_grad():
            q_next_online, _ = self.ddqn(next_obs)
            best_actions = q_next_online.argmax(dim=-1, keepdim=True)
            q_next_target, _ = self.ddqn.target_forward(next_obs)
            q_next = q_next_target.gather(1, best_actions).squeeze(1)
            y = rewards + self.gamma * q_next * (1.0 - dones)

        loss = nn.functional.mse_loss(q_curr, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.fusion.parameters()) +
            list(self.ddqn.online.parameters()), 1.0
        )
        self.optimizer.step()
        self.ddqn.soft_update(self.tau)

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def get_obs(self) -> np.ndarray:
        """히스토리 큐를 (T, fused_dim) 배열로 반환"""
        return np.stack(list(self._fused_history), axis=0)

    def get_last_attention(self) -> np.ndarray | None:
        """마지막 fusion 단계의 이웃별 attention score"""
        return self._last_attn

    def save(self, path: str) -> None:
        torch.save({
            "fusion": self.fusion.state_dict(),
            "ddqn_online": self.ddqn.online.state_dict(),
            "ddqn_target": self.ddqn.target.state_dict(),
            "eps": self.eps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.fusion.load_state_dict(ckpt["fusion"])
        self.ddqn.online.load_state_dict(ckpt["ddqn_online"])
        self.ddqn.target.load_state_dict(ckpt["ddqn_target"])
        self.eps = ckpt.get("eps", self.eps_end)
