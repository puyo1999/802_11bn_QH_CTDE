"""
src/envs/reward.py
QH-CTDE 협력 보상 산출.

    R = R_thr + lambda_f * R_fair + lambda_l * R_lat

  R_thr  : 비례 공정 로그 처리량 합,  sum_i log(1 + Thp_i)
  R_fair : Jain 공정 지수에서 1을 뺀 값, 범위 [-1, 0]
  R_lat  : UHR 지연 기한 초과에 대한 지수 패널티

중첩 제약으로 차단된 전송에 대한 별도 패널티 항을 선택적으로 포함한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RewardConfig:
    lambda_fair: float = 0.5
    lambda_latency: float = 0.2
    lambda_overlap: float = 0.0     # 0이면 중첩 차단에 대한 명시적 패널티 없음
    latency_deadline_ms: float = 5.0
    latency_exp_scale: float = 0.02   # 0.1 -> 0.02: saturated 트래픽(큐≈100)에서
                                       # exp(0.1*(100-5))≈1.3e4로 throughput 항을 압도하던 문제 완화
    latency_clip: float = -1e6
    slot_ms: float = 1.0


@dataclass
class RewardBreakdown:
    """보상 구성 요소별 값. 학습 진단 및 논문 분석용으로 info에 노출한다."""
    total: float
    r_throughput: float
    r_fairness: float
    r_latency: float
    r_overlap: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reward_total": self.total,
            "reward_thr": self.r_throughput,
            "reward_fair": self.r_fairness,
            "reward_lat": self.r_latency,
            "reward_overlap": self.r_overlap,
        }


def throughput_reward(successful_tx: Sequence[float]) -> float:
    """비례 공정 로그 처리량 합."""
    return float(sum(math.log1p(max(float(tx), 0.0)) for tx in successful_tx))


def jain_fairness(values: Sequence[float]) -> float:
    """Jain 공정 지수. 전 항이 0이면 완전 공정(1.0)으로 간주한다."""
    arr = np.asarray(values, dtype=np.float64)
    sq_sum = float(np.sum(arr ** 2))
    if sq_sum <= 0.0:
        return 1.0
    return float(np.sum(arr) ** 2 / (len(arr) * sq_sum))


def fairness_reward(successful_tx: Sequence[float]) -> float:
    """Jain 지수를 [-1, 0] 범위 보상으로 이동한다."""
    return jain_fairness(successful_tx) - 1.0


def latency_reward(
    queue_lengths: Sequence[Sequence[float]],
    cfg: RewardConfig,
) -> float:
    """
    AP별 평균 큐 길이를 대기 시간으로 환산하고, 기한 초과분에 지수 패널티를 부과한다.
    """
    penalty = 0.0
    for queues in queue_lengths:
        est_delay_ms = float(np.mean(queues)) * cfg.slot_ms
        if est_delay_ms > cfg.latency_deadline_ms:
            penalty -= math.exp(
                cfg.latency_exp_scale * (est_delay_ms - cfg.latency_deadline_ms)
            )
    return max(penalty, cfg.latency_clip)


def compute_reward(
    successful_tx: Sequence[float],
    queue_lengths: Sequence[Sequence[float]],
    cfg: RewardConfig,
    n_overlap_blocked: int = 0,
) -> RewardBreakdown:
    """보상 구성 요소를 모두 산출하고 가중 합을 반환한다."""
    r_thr = throughput_reward(successful_tx)
    r_fair = fairness_reward(successful_tx)
    r_lat = latency_reward(queue_lengths, cfg)
    r_ovl = -float(n_overlap_blocked)

    total = (
        r_thr
        + cfg.lambda_fair * r_fair
        + cfg.lambda_latency * r_lat
        + cfg.lambda_overlap * r_ovl
    )
    return RewardBreakdown(
        total=float(total),
        r_throughput=r_thr,
        r_fairness=r_fair,
        r_latency=r_lat,
        r_overlap=r_ovl,
    )
