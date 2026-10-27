"""
src/envs/interference.py
수신 신호 대 간섭 잡음비(SINR) 산출.

간섭 누적 순서를 명시적으로 고정한다.

    I_total = ( sum_j G_overlap(i,k,j) * P_rx(j -> STA) ) * G_harsh + N

    1. 동시 전송 중인 타 AP의 수신 전력 합산. 중첩 관계인 AP의 기여분에는
       soft 모드 이득 G_overlap이 곱해진다.
    2. 가혹 조건 이득 G_harsh는 간섭 성분에만 적용된다.
    3. 열잡음 N은 마지막에 한 번만 가산된다.

기존 ctde_obss_env.py는 _dbm_to_mw(NOISE_DBM)을 두 번 가산하고 있었으며
(ctde_obss_env_mod1.py에서 수정됨), 가혹 이득 적용 시점도 파일마다 달랐다.
본 모듈이 계산 순서의 단일 출처가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.envs.overlap import OverlapPolicy
from src.envs.physical import (
    DEFAULT_NOISE_DBM,
    DEFAULT_TX_POWER_DBM,
    PathLossModel,
    dbm_to_mw,
    mw_to_db,
)


@dataclass
class SINRResult:
    """단일 전송의 SINR 산출 결과."""
    sinr_db: float
    rx_power_mw: float
    interference_mw: float
    blocked_by_overlap: bool
    active_overlap_peers: list[int]

    @property
    def is_hard_blocked(self) -> bool:
        return self.blocked_by_overlap


class InterferenceModel:
    """토폴로지와 overlap 정책에 결합된 SINR 계산기."""

    def __init__(
        self,
        topology,
        overlap_policy: OverlapPolicy,
        path_loss: PathLossModel | None = None,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        noise_dbm: float = DEFAULT_NOISE_DBM,
        harsh_interference_gain: float = 5.0,
    ):
        self.topo = topology
        self.overlap = overlap_policy
        self.path_loss = path_loss or PathLossModel()
        self.tx_power_dbm = tx_power_dbm
        self.noise_mw = dbm_to_mw(noise_dbm)
        self.harsh_interference_gain = harsh_interference_gain
        # 슬롯별 소규모 페이딩 전력 이득 fading_gain[tx_ap, serving_ap, sta] (평균 1).
        # None 이면 결정적 경로손실만 쓴다. 환경이 슬롯마다 갱신한다.
        self.fading_gain: np.ndarray | None = None

    # ── 좌표 조회 ────────────────────────────────────────────────
    _ISOLATED_POSITION = np.array([9999.0, 9999.0], dtype=np.float32)

    def sta_position(self, ap_idx: int, sta_idx: int) -> np.ndarray:
        """
        STA 좌표를 반환한다. 토폴로지에 존재하지 않는 인덱스를 지정한 경우
        시스템에 간섭을 주지 않는 원거리 좌표로 격리한다.
        """
        stas = self.topo.sta_positions[ap_idx]
        if 0 <= sta_idx < len(stas):
            return np.asarray(stas[sta_idx], dtype=np.float32)
        return self._ISOLATED_POSITION

    def rx_power(self, tx_ap: int, rx_pos, tx_power_dbm: float | None = None) -> float:
        power = self.tx_power_dbm if tx_power_dbm is None else tx_power_dbm
        return self.path_loss.rx_power_mw(self.topo.ap_positions[tx_ap], rx_pos, power)

    # ── SINR 산출 ────────────────────────────────────────────────
    def compute(
        self,
        ap_idx: int,
        sta_idx: int,
        transmitting_aps: Iterable[int],
        is_harsh: bool = False,
        tx_powers: Sequence[float] | None = None,
    ) -> SINRResult:
        """
        ap_idx가 sta_idx로 전송할 때의 SINR을 산출한다.

        Args:
            transmitting_aps: 동일 슬롯에 전송 중인 AP 인덱스 집합 (ap_idx 포함 무방)
            is_harsh:         가혹 간섭 조건 활성화 여부
            tx_powers:        AP별 송신 전력(dBm). None이면 공통값을 사용한다.
        """
        transmitting = {j for j in transmitting_aps if j != ap_idx}
        sta_pos = self.sta_position(ap_idx, sta_idx)

        def power_of(j: int) -> float:
            return self.tx_power_dbm if tx_powers is None else tx_powers[j]

        rx_mw = self.rx_power(ap_idx, sta_pos, power_of(ap_idx)) * self._fading(ap_idx, ap_idx, sta_idx)

        # 1. 중첩 제약 판정 (hard 모드)
        active_peers = self.overlap.active_peers(ap_idx, sta_idx, transmitting)
        blocked = self.overlap.blocks_transmission(ap_idx, sta_idx, transmitting)

        # 2. 동시 전송 AP의 간섭 누적 (soft 모드 이득 포함)
        interference_mw = 0.0
        for j in transmitting:
            contribution = self.rx_power(j, sta_pos, power_of(j)) * self._fading(j, ap_idx, sta_idx)
            contribution *= self.overlap.interference_gain(ap_idx, sta_idx, j, transmitting)
            interference_mw += contribution

        # 3. 가혹 조건 이득은 간섭 성분에만 적용
        if is_harsh:
            interference_mw *= self.harsh_interference_gain

        # 4. 열잡음 1회 가산
        interference_mw += self.noise_mw

        return SINRResult(
            sinr_db=mw_to_db(rx_mw / interference_mw),
            rx_power_mw=rx_mw,
            interference_mw=interference_mw,
            blocked_by_overlap=blocked,
            active_overlap_peers=active_peers,
        )

    def _fading(self, tx_ap: int, ap_idx: int, sta_idx: int) -> float:
        """tx_ap 에서 (ap_idx 의 sta_idx) 수신기로 가는 링크의 이번 슬롯 페이딩 이득."""
        gain = self.fading_gain
        if gain is None or not 0 <= sta_idx < gain.shape[2]:
            return 1.0
        return float(gain[tx_ap, ap_idx, sta_idx])

    def link_sinr_db(self, ap_idx: int, sta_idx: int) -> float:
        """간섭 없는 조건의 평균 링크 SNR(잡음 대비, 페이딩 제외). 관측 벡터의 채널 품질 특징에 사용한다."""
        sta_pos = self.sta_position(ap_idx, sta_idx)
        rx_mw = self.rx_power(ap_idx, sta_pos)
        return mw_to_db(rx_mw / self.noise_mw)

    def neighbor_rssi_dbm(self, ap_idx: int) -> float:
        """이웃 AP들로부터의 평균 수신 전력(dBm). 이웃이 없으면 잡음 레벨을 반환한다."""
        neighbors = self.topo.neighbors[ap_idx]
        if not neighbors:
            return mw_to_db(self.noise_mw)
        powers = [
            self.rx_power(j, self.topo.ap_positions[ap_idx])
            for j in neighbors
        ]
        return mw_to_db(float(np.mean(powers)))
