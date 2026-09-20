"""
src/envs/traffic.py
연속 부하 비율(traffic_load_ratio) 기반 트래픽 도착 모델.

기존 traffic_type in {"saturated", "poisson"} 이분법을 대체한다.

  ratio=0.0 : 평균 도착률 poisson_lambda_min (가벼운 부하)
  ratio=1.0 : 평균 도착률 poisson_lambda_max (서비스 용량을 상회 → 자연 포화)

두 극단 모두 매 스텝 Poisson(lambda) 표본으로 도착을 생성한다는 점은 동일하다.
차이는 강도(lambda)뿐이며, 큐는 traffic_load_ratio 값과 무관하게 항상
실제 전송 성공분만큼 차감된다. 기존 'saturated' 구현처럼 큐를 강제로
queue_max_size에 고정하는 우회 로직은 두지 않는다 — 그 우회가 정책과
무관한 상수 지연 패널티를 만든 원인이었다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrafficConfig:
    traffic_load_ratio: float = 0.7
    poisson_lambda_min: float = 0.1   # ratio=0 일 때 STA당 평균 도착률 (packets/slot)
    poisson_lambda_max: float = 3.0   # ratio=1 일 때 STA당 평균 도착률
    init_fill_ratio: float = 0.0      # 에피소드 시작 시 큐 초기 점유 비율 (0~1)

    def __post_init__(self) -> None:
        if not 0.0 <= self.traffic_load_ratio <= 1.0:
            raise ValueError(f"traffic_load_ratio는 [0,1] 범위여야 합니다: {self.traffic_load_ratio}")
        if self.poisson_lambda_max < self.poisson_lambda_min:
            raise ValueError("poisson_lambda_max는 poisson_lambda_min 이상이어야 합니다")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrafficConfig":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def effective_lambda(self) -> float:
        """ratio로 선형 보간된 평균 도착률."""
        return (
            self.poisson_lambda_min
            + self.traffic_load_ratio * (self.poisson_lambda_max - self.poisson_lambda_min)
        )


class TrafficModel:
    """AP별 STA 큐에 Poisson 도착을 주입하는 모델."""

    def __init__(self, cfg: TrafficConfig, rng: np.random.Generator):
        self.cfg = cfg
        self._rng = rng

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def initial_queue(self, k_stas: int, queue_max_size: int) -> np.ndarray:
        fill = self.cfg.init_fill_ratio * queue_max_size
        return np.full(k_stas, fill, dtype=np.float32)

    def arrivals(self, k_stas: int) -> np.ndarray:
        """이번 스텝에 각 STA 큐에 도착하는 패킷 수."""
        return self._rng.poisson(self.cfg.effective_lambda, size=k_stas).astype(np.float32)

    def step_queue(self, queue: np.ndarray, queue_max_size: int) -> np.ndarray:
        """도착을 주입하고 상한으로 클리핑한다. 차감(전송 성공)은 호출자가 별도 처리한다."""
        return np.clip(queue + self.arrivals(len(queue)), 0.0, queue_max_size)
