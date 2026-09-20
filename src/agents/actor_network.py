"""
src/agents/actor_network.py
QH-CTDE 에이전트의 신경망 구성 요소.

VQ/DSM/학습 루프와 분리된 순수 nn.Module. state_dim은 생성자 인자로만 받으며
어디에도 하드코딩하지 않는다 (src.envs.space_validator가 산출한 값을 그대로 받는다).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorFeatureExtractor(nn.Module):
    """관측 → hidden_dim 특징. STA/MCS 헤드의 공통 입력을 만든다."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class SchedulingHeads(nn.Module):
    """공통 특징 → (STA 선택 로짓, MCS 선택 로짓)."""

    def __init__(self, hidden_dim: int, n_stas: int, n_mcs: int):
        super().__init__()
        self.sta_head = nn.Linear(hidden_dim, n_stas)
        self.mcs_head = nn.Linear(hidden_dim, n_mcs)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sta_head(feat), self.mcs_head(feat)
