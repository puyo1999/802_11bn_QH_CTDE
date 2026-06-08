"""src/utils/metrics.py — Wi-Fi 8 PAR 핵심 지표 계산"""

import numpy as np


def compute_d95(delays: list[float]) -> float:
    """95번째 백분위 지연 (초 단위)"""
    if not delays:
        return 0.0
    return float(np.percentile(delays, 95))


def compute_thp(pkts_sent: int, pkts_total: int) -> float:
    """정규화 처리량 (0~1)"""
    return pkts_sent / pkts_total if pkts_total > 0 else 0.0


def compute_per(pkts_failed: int, pkts_total: int) -> float:
    """패킷 오류율"""
    return pkts_failed / pkts_total if pkts_total > 0 else 0.0


def d95_reduction(d95_ai: float, d95_csma: float) -> float:
    """d95 감소율 (%) — 논문 Fig.8 기준"""
    if d95_csma == 0:
        return 0.0
    return (d95_csma - d95_ai) / d95_csma * 100.0


def aggregate_metrics(info_list: list[dict]) -> dict:
    """
    여러 step의 info를 집계하여 평균 반환.
    info = {"d95": [...], "thp": [...], "per": [...]}
    """
    d95_vals = [np.mean(info["d95"]) for info in info_list if info["d95"]]
    thp_vals = [np.mean(info["thp"]) for info in info_list if info["thp"]]
    per_vals = [np.mean(info["per"]) for info in info_list if info["per"]]
    return {
        "d95_mean": float(np.mean(d95_vals)) if d95_vals else 0.0,
        "thp_mean": float(np.mean(thp_vals)) if thp_vals else 0.0,
        "per_mean": float(np.mean(per_vals)) if per_vals else 0.0,
    }
