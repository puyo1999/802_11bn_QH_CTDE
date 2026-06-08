"""
src/networks/fusion_fc.py
논문 원본 FC Fusion Network (Baseline)

입력: [s_t^i, s_t^j]  — AP 자신 + 이웃 AP 상태 concat
출력: m_t^i  — 1×8 fused vector

구조: FC64 → FC32 → FC16 → FC8  (Fig.4 기준)
"""

import torch
import torch.nn as nn


class FCFusionNet(nn.Module):
    """
    논문 Fig.4의 Fusion Net 그대로 구현.
    이웃이 여러 명일 경우 concat 후 한 번에 처리 (zero-padding 필요).
    """

    def __init__(self, state_dim: int = 6, max_neighbors: int = 3,
                 out_dim: int = 8):
        super().__init__()
        # 자신(1) + 이웃(max_neighbors) → concat
        in_dim = state_dim * (1 + max_neighbors)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 16),     nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(
        self,
        self_state: torch.Tensor,        # (B, state_dim)
        neighbor_states: torch.Tensor,   # (B, max_nbr, state_dim) — zero-padded
    ) -> torch.Tensor:                   # (B, out_dim)
        B = self_state.shape[0]
        nbr_flat = neighbor_states.reshape(B, -1)   # (B, max_nbr * state_dim)
        x = torch.cat([self_state, nbr_flat], dim=-1)
        return self.net(x)
