"""
src/agents/dsm_gate.py
Coherence-Aware Dynamic Switching Module (CA-DSM) 게이트.

환경 관측 벡터의 마지막 차원(tau_ratio = tau_sig/tau_coh)을 코히어런스 신호로
사용하여, 중앙 코디네이션(양자화 feature)과 로컬 폴백(raw feature) 중 어느 것을
쓸지 결정론적으로 판정한다.

  tau_ratio <= gamma_margin  → 백홀/채널이 코히어런트 → 양자화된 코디네이션 정보 사용
  tau_ratio >  gamma_margin  → 코히어런스 붕괴          → 로컬 raw feature로 폴백

이 판정은 상태에 대해 순수 함수이므로 select_action()과 update() 양쪽에서
동일하게 재계산되어, 행동 정책과 그래디언트 경로가 항상 일치한다.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

DSMMode = Literal["auto", "always_quant", "never_quant"]
VALID_MODES = ("auto", "always_quant", "never_quant")

# qh_obss_env_v2.ObservationBuilder 기준 관측 벡터의 마지막 차원이 tau_ratio.
TAU_RATIO_OBS_INDEX = -1


class DSMGate:
    """단일/배치 상태에 대한 코히어런스 기반 양자화 게이트."""

    def __init__(self, use_dsm: bool = True, gamma_margin: float = 0.8):
        self.use_dsm = use_dsm
        self.gamma_margin = gamma_margin

    def decide(self, state: np.ndarray, mode: DSMMode = "auto") -> bool:
        """단일 상태에 대해 양자화 feature 사용 여부(True)를 반환한다."""
        if mode not in VALID_MODES:
            raise ValueError(f"dsm_mode는 {VALID_MODES} 중 하나여야 합니다 (현재: {mode})")
        if mode == "always_quant":
            return True
        if mode == "never_quant":
            return False
        # mode == "auto"
        if not self.use_dsm:
            return True  # DSM 비활성 시 항상 코디네이션 (Basic과 동일 동작)
        tau_ratio = float(state[TAU_RATIO_OBS_INDEX])
        return tau_ratio <= self.gamma_margin

    def decide_batch(self, states: torch.Tensor, mode: DSMMode = "auto") -> torch.Tensor:
        """states: (B, state_dim) → (B,) bool 텐서. update()의 배치 경로에서 사용."""
        if mode not in VALID_MODES:
            raise ValueError(f"dsm_mode는 {VALID_MODES} 중 하나여야 합니다 (현재: {mode})")
        if mode == "always_quant":
            return torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
        if mode == "never_quant":
            return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)
        if not self.use_dsm:
            return torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
        tau_ratio = states[:, TAU_RATIO_OBS_INDEX]
        return tau_ratio <= self.gamma_margin
