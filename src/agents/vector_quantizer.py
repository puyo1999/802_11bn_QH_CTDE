"""
src/agents/vector_quantizer.py
Straight-Through Estimator 기반 벡터 양자화 코드북.

QHCTDEAgent에서 분리한 이유: 코드북 학습/거리 계산 로직은 DSM 게이팅이나
행동 선택과 독립적으로 테스트·교체 가능해야 한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    입력 feature를 codebook_size개의 학습 가능한 벡터 중 하나로 양자화한다.
    Forward: 최근접 코드북 벡터. Backward: STE로 원본 그래디언트를 그대로 통과시킨다.
    """

    def __init__(self, codebook_size: int, feature_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.feature_dim = feature_dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, feature_dim))

    def _nearest_indices(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, D) → (B,) 최근접 코드북 인덱스."""
        distances = torch.sum(
            (feat.unsqueeze(1) - self.codebook.unsqueeze(0)) ** 2, dim=-1
        )
        return torch.argmin(distances, dim=1)

    def quantize(self, feat: torch.Tensor, straight_through: bool = True) -> torch.Tensor:
        """
        feat: (B, D) → (B, D) 양자화된 feature.
        straight_through=True이면 backward 시 feat의 그래디언트를 그대로 전달한다(STE).
        False이면 순수 lookup만 수행한다 (no_grad 추론 경로에서 사용).
        """
        indices = self._nearest_indices(feat)
        quantized = self.codebook[indices]
        if straight_through:
            return feat + (quantized - feat).detach()
        return quantized

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.quantize(feat, straight_through=True)
