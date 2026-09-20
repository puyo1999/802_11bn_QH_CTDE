"""
src/envs/observation.py
AP별 지역 관측 벡터 구성.

    o_i = [ beta_i (K) | gamma_i (K) | overlap_i (K, 선택) | r_bar_i (1) | tau_i (1) ]

  beta_i    : STA별 정규화 버퍼 점유율
  gamma_i   : STA별 정규화 링크 SINR (간섭 제외, 잡음 대비)
  overlap_i : STA별 중첩 지시자. overlap.expose_in_obs=true일 때만 포함된다.
  r_bar_i   : 이웃 AP 평균 수신 전력 정규화 값
  tau_i     : 백홀 지연 대 채널 응집 시간 비율 tau_sig / tau_coh

차원은 설정에 따라 2K+2 또는 3K+2로 결정되며, 하드코딩하지 않고 본 모듈이 산출한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.interference import InterferenceModel
from src.envs.overlap import OverlapPolicy

SINR_NORM_DIVISOR = 40.0     # dB → [0,1] 근사 정규화
RSSI_NORM_DIVISOR = 100.0
TAU_NORMAL = 0.3
TAU_HARSH = 1.5


@dataclass
class ObservationBuilder:
    """관측 벡터 구성기. 환경이 보유하는 상태를 인자로 받아 순수 함수처럼 동작한다."""
    n_aps: int
    k_stas: int
    queue_max_size: int
    interference: InterferenceModel
    overlap: OverlapPolicy

    def __post_init__(self) -> None:
        self._overlap_dim = self.overlap.obs_feature_dim(self.k_stas)
        # 링크 SINR은 토폴로지 고정값이므로 최초 1회만 계산하여 재사용한다.
        self._link_sinr_cache = self._precompute_link_sinr()
        self._overlap_cache = [
            self.overlap.obs_features(i, self.k_stas) for i in range(self.n_aps)
        ]

    @property
    def obs_dim(self) -> int:
        """AP당 관측 차원. 2K + overlap_dim + 2."""
        return 2 * self.k_stas + self._overlap_dim + 2

    def feature_layout(self) -> list[tuple[str, int, int]]:
        """(이름, 시작 인덱스, 길이) 목록. 디버깅 및 논문 Table 기술에 사용한다."""
        layout = [
            ("buffer_ratio", 0, self.k_stas),
            ("link_sinr", self.k_stas, self.k_stas),
        ]
        cursor = 2 * self.k_stas
        if self._overlap_dim:
            layout.append(("overlap_indicator", cursor, self._overlap_dim))
            cursor += self._overlap_dim
        layout.append(("neighbor_rssi", cursor, 1))
        layout.append(("tau_ratio", cursor + 1, 1))
        return layout

    def _precompute_link_sinr(self) -> np.ndarray:
        """AP×STA 링크 SINR 정규화 값. 미존재 STA 슬롯은 0으로 채운다."""
        cache = np.zeros((self.n_aps, self.k_stas), dtype=np.float32)
        for i in range(self.n_aps):
            n_real = len(self.interference.topo.sta_positions[i])
            for k in range(self.k_stas):
                if k < n_real:
                    cache[i, k] = self.interference.link_sinr_db(i, k) / SINR_NORM_DIVISOR
        return cache

    def build(
        self,
        queues: list[np.ndarray],
        is_harsh: bool,
    ) -> np.ndarray:
        """전 AP의 관측 행렬 (n_aps, obs_dim)을 구성한다."""
        obs = np.zeros((self.n_aps, self.obs_dim), dtype=np.float32)
        tau = TAU_HARSH if is_harsh else TAU_NORMAL

        for i in range(self.n_aps):
            parts = [
                np.asarray(queues[i], dtype=np.float32) / self.queue_max_size,
                self._link_sinr_cache[i],
            ]
            if self._overlap_dim:
                parts.append(self._overlap_cache[i])
            parts.append(
                np.array(
                    [self.interference.neighbor_rssi_dbm(i) / RSSI_NORM_DIVISOR, tau],
                    dtype=np.float32,
                )
            )
            obs[i] = np.concatenate(parts)

        return obs
