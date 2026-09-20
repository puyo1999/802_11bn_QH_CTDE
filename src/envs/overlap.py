"""
src/envs/overlap.py
중첩 영역 STA(overlap STA)에 대한 명시적 간섭 제약 정책.

토폴로지 사양의 stas[].overlap_with 는 해당 STA가 서빙 AP 외의 다른 AP
커버리지와 물리적으로 중첩된 위치에 있음을 선언한다. 좌표 기반 SINR 계산만으로는
이 조건이 거리 차이에 희석되므로, 본 모듈이 중첩 관계를 독립적인 제약으로 강제한다.

정책 두 가지를 제공한다.

  hard: 중첩 상대 AP 중 하나라도 동일 슬롯에 전송하면 해당 STA로의 전송은 실패한다.
        중첩 영역에서 동시 전송이 구조적으로 불가능한 조건을 결정론적으로 모델링한다.

  soft: 중첩 상대 AP가 유발하는 간섭 전력에 고정 이득을 곱한다. 동시 전송을 금지하지
        않고 성공 확률만 저하시킨다.

에이전트는 실행 시점에 다른 AP의 현재 행동을 관측하지 못한다. 따라서 중첩 여부는
정적 지시자(static indicator)로 관측 벡터에 노출되며, 동시 전송 발생 여부는
보상을 통해서만 간접적으로 학습된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.envs.physical import db_to_linear
from src.envs.topology import Topology

HARD = "hard"
SOFT = "soft"
VALID_MODES = (HARD, SOFT)


@dataclass
class OverlapConfig:
    """overlap 제약 설정. EnvConfig.overlap 하위 딕셔너리와 대응한다."""
    enabled: bool = True
    mode: str = HARD                 # hard | soft
    soft_gain_db: float = 10.0       # soft 모드에서 중첩 AP 간섭에 적용하는 이득
    expose_in_obs: bool = True       # 관측 벡터에 중첩 지시자 K차원을 추가할지 여부

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"overlap.mode는 {VALID_MODES} 중 하나여야 합니다 (현재: {self.mode})")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "OverlapConfig":
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


class OverlapPolicy:
    """특정 토폴로지에 결합된 overlap 제약 판정기."""

    def __init__(self, topology: Topology, cfg: OverlapConfig | None = None):
        self.topo = topology
        self.cfg = cfg or OverlapConfig()
        self._soft_gain_linear = db_to_linear(self.cfg.soft_gain_db)

        # peers[i][k] = i번 AP의 k번째 STA가 중첩된 다른 AP 인덱스 목록
        self.peers: list[list[list[int]]] = topology.sta_overlap

        # 통계 카운터 — 에피소드별 제약 발동 횟수를 info로 노출한다.
        self.blocked_count = 0
        self.soft_penalty_count = 0

    # ── 조회 ────────────────────────────────────────────────────
    def is_active(self) -> bool:
        return self.cfg.enabled and self.topo.n_overlap_stas() > 0

    def peers_of(self, ap_idx: int, sta_idx: int) -> list[int]:
        """해당 STA가 중첩된 상대 AP 목록. 인덱스 범위를 벗어나면 빈 목록."""
        if not 0 <= ap_idx < len(self.peers):
            return []
        per_ap = self.peers[ap_idx]
        if not 0 <= sta_idx < len(per_ap):
            return []
        return per_ap[sta_idx]

    def active_peers(
        self, ap_idx: int, sta_idx: int, transmitting_aps: Iterable[int]
    ) -> list[int]:
        """동일 슬롯에 전송 중인 중첩 상대 AP 목록."""
        if not self.is_active():
            return []
        active = set(transmitting_aps)
        return [j for j in self.peers_of(ap_idx, sta_idx) if j in active]

    # ── 제약 적용 ────────────────────────────────────────────────
    def blocks_transmission(
        self, ap_idx: int, sta_idx: int, transmitting_aps: Iterable[int]
    ) -> bool:
        """
        hard 모드에서 해당 전송이 중첩 제약으로 실패하는지 판정한다.
        soft 모드에서는 항상 False를 반환한다.
        """
        if not self.is_active() or self.cfg.mode != HARD:
            return False
        blocked = bool(self.active_peers(ap_idx, sta_idx, transmitting_aps))
        if blocked:
            self.blocked_count += 1
        return blocked

    def interference_gain(
        self, ap_idx: int, sta_idx: int, peer_idx: int, transmitting_aps: Iterable[int]
    ) -> float:
        """
        soft 모드에서 peer_idx가 해당 STA에 유발하는 간섭에 곱할 배율을 반환한다.
        중첩 관계가 아니거나 hard 모드이면 1.0을 반환한다.
        """
        if not self.is_active() or self.cfg.mode != SOFT:
            return 1.0
        if peer_idx not in self.peers_of(ap_idx, sta_idx):
            return 1.0
        if peer_idx not in set(transmitting_aps):
            return 1.0
        self.soft_penalty_count += 1
        return self._soft_gain_linear

    # ── 관측 특징 ────────────────────────────────────────────────
    def obs_feature_dim(self, k_stas: int) -> int:
        """관측 벡터에 추가되는 차원 수."""
        return k_stas if (self.cfg.enabled and self.cfg.expose_in_obs) else 0

    def obs_features(self, ap_idx: int, k_stas: int) -> np.ndarray:
        """
        AP별 중첩 지시자 벡터 (길이 k_stas).
        값은 해당 STA가 중첩된 상대 AP 수를 전체 AP 수로 정규화한 값이며,
        비중첩 STA와 미존재 STA 슬롯은 0이다.
        """
        if not (self.cfg.enabled and self.cfg.expose_in_obs):
            return np.zeros(0, dtype=np.float32)

        feature = np.zeros(k_stas, dtype=np.float32)
        denom = max(self.topo.n_aps - 1, 1)
        for k in range(k_stas):
            peers = self.peers_of(ap_idx, k)
            if peers:
                feature[k] = len(peers) / denom
        return feature

    # ── 통계 ────────────────────────────────────────────────────
    def reset_stats(self) -> None:
        self.blocked_count = 0
        self.soft_penalty_count = 0

    def stats(self) -> dict[str, int]:
        return {
            "overlap_blocked": self.blocked_count,
            "overlap_soft_penalty": self.soft_penalty_count,
        }
