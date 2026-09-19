"""
src/envs/link_adaptation.py
MCS 인덱스에 대한 SINR 요구치와 전송 용량 매핑.

MCS가 높을수록 전송 용량은 증가하나 요구 SINR도 상승한다. 에이전트는 이 상충
관계를 명시적으로 학습한다.
"""

from __future__ import annotations

from dataclasses import dataclass

N_MCS_DEFAULT = 16


@dataclass
class MCSTable:
    """선형 근사 MCS 테이블. n_mcs=16 기준 SINR 2.0~24.5 dB, 용량 1~23 패킷."""
    n_mcs: int = N_MCS_DEFAULT
    sinr_base_db: float = 2.0
    sinr_step_db: float = 1.5
    capacity_base: int = 1
    capacity_step: float = 1.5

    def __post_init__(self) -> None:
        self.sinr_thresholds = [
            self.sinr_base_db + self.sinr_step_db * i for i in range(self.n_mcs)
        ]
        self.capacities = [
            self.capacity_base + int(self.capacity_step * i) for i in range(self.n_mcs)
        ]

    def clamp(self, mcs_idx: int) -> int:
        return int(max(0, min(int(mcs_idx), self.n_mcs - 1)))

    def required_sinr_db(self, mcs_idx: int) -> float:
        return self.sinr_thresholds[self.clamp(mcs_idx)]

    def capacity(self, mcs_idx: int) -> int:
        return self.capacities[self.clamp(mcs_idx)]

    def decodable(self, sinr_db: float, mcs_idx: int) -> bool:
        return sinr_db >= self.required_sinr_db(mcs_idx)

    def best_mcs_for(self, sinr_db: float) -> int:
        """주어진 SINR에서 복호 가능한 최고 MCS. 전부 불가하면 0을 반환한다."""
        feasible = [i for i, thr in enumerate(self.sinr_thresholds) if sinr_db >= thr]
        return feasible[-1] if feasible else 0
