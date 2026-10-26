"""
src/envs/physical.py
물리 계층 공통 연산.

기존에는 동일한 _rx_power / _dbm_to_mw / _mw_to_db 정적 메서드가
obss_env.py, ctde_obss_env.py, qh_obss_env.py, qh_obss_env_mod1.py에
네 번 중복 정의되어 있었다. 본 모듈이 단일 출처가 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── 기본 물리 상수 ───────────────────────────────────────────────
DEFAULT_PL_INTERCEPT = -46.67   # dB, 6 GHz 실내 경로 손실 모델 절편
DEFAULT_PL_EXPONENT = 30.0      # dB/decade
DEFAULT_MIN_DISTANCE_M = 1.0
DEFAULT_NOISE_DBM = -95.0
DEFAULT_TX_POWER_DBM = 15.0


def dbm_to_mw(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)


def mw_to_db(mw: float) -> float:
    return 10.0 * math.log10(max(mw, 1e-30))


def db_to_linear(db: float) -> float:
    """dB 이득을 선형 배율로 변환한다 (10 dB → 10배)."""
    return 10.0 ** (db / 10.0)


@dataclass
class PathLossModel:
    """로그 거리 경로 손실 모델: PL(d) = intercept - exponent * log10(d)."""
    intercept_db: float = DEFAULT_PL_INTERCEPT
    exponent: float = DEFAULT_PL_EXPONENT
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M

    def path_loss_db(self, distance_m: float) -> float:
        d = max(float(distance_m), self.min_distance_m)
        return self.intercept_db - self.exponent * math.log10(d)

    def rx_power_dbm(self, tx_pos, rx_pos, tx_power_dbm: float) -> float:
        distance = float(np.linalg.norm(np.asarray(tx_pos, dtype=np.float64)
                                        - np.asarray(rx_pos, dtype=np.float64)))
        return tx_power_dbm + self.path_loss_db(distance)

    def rx_power_mw(self, tx_pos, rx_pos, tx_power_dbm: float) -> float:
        return dbm_to_mw(self.rx_power_dbm(tx_pos, rx_pos, tx_power_dbm))
