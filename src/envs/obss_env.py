"""
src/envs/obss_env.py (Updated)
IEEE 802.11bn OBSS 채널 접근 Gymnasium 환경

논문 파라미터 (Table I) 기준:
  - Time slot: 9 μs
  - Packet length: 120 slots
  - MCS: 5 (fixed)
  - Path loss: -46.67 - 30*log10(D)
  - Traffic: Saturated or Poisson (configurable)
  - CCA default: -82 dBm
  - Transmit power P: 10 dBm
  - Noise power: -95 dBm
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.topology import Topology


# ── 물리층 상수 ────────────────────────────────────────────────
SLOT_US      = 9e-6          # time slot (s)
DIFS_SLOTS   = 2             # DIFS = 2 slots (simplified)
PKT_LEN      = 120           # slots per packet
NOISE_DBM    = -95.0         # dBm
TX_POWER_DBM = 10.0          # dBm default
CCA_DEFAULT  = -82.0         # dBm
MCS_IDX      = 5             # fixed MCS
# MCS5 → SINR threshold 약 14 dB (802.11ax 기준 근사)
SINR_THRESH_DB = 14.0
# CW 지수 집합: b ∈ {0,...,6} → CW = 2^(4+b)
CW_EXPS      = list(range(7))   # [0,1,2,3,4,5,6]
# CCA 조정: Δp ∈ {0,5,10,15,20} dBm
CCA_DELTAS   = [0, 5, 10, 15, 20]


@dataclass
class APState:
    """단일 AP의 내부 상태"""
    ap_id:         int
    cw_exp:        int   = 0      # 현재 CW 지수 b
    backoff:       int   = 0      # 남은 backoff counter
    tx_power:      float = TX_POWER_DBM
    cca_thresh:    float = CCA_DEFAULT
    # 통계 (매 interact 구간마다 누적 후 reset)
    pkts_sent:     int   = 0
    pkts_collided: int   = 0
    delays:        list  = field(default_factory=list)
    # Traffic queue (for saturated/poisson traffic)
    pkts_in_queue: int   = 0


class OBSSEnv(gym.Env):
    """
    IEEE 802.11bn OBSS Gymnasium 환경.

    Observation space (per AP, after fusion — raw state vector):
        [prev_action, d_ave, d95, THP, PER, CS_power]   shape=(6,)

    Action space:
        mode='cw'   → Discrete(7)   (CW exponent b)
        mode='cca'  → Discrete(5)   (Δp index)
        mode='joint' → MultiDiscrete([7, 5])

    Step 단위: interact_slots (논문 ˆt = 0.5s → 약 55,555 slots)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: Topology,
        mode: str = "cw",            # 'cw' | 'cca' | 'joint'
        interact_slots: int = 55_556,
        max_episodes: int = 600,
        seed: int | None = None,
        traffic_type: str = "saturated",  # 'saturated' | 'poisson'
        poisson_arrival_rate: float = 0.1,  # packets per slot (for poisson)
        queue_max_size: int = 100,  # max queue size
    ):
        super().__init__()
        assert mode in ("cw", "cca", "joint"), f"Unknown mode: {mode}"
        assert traffic_type in ("saturated", "poisson"), f"Unknown traffic_type: {traffic_type}"

        self.topo               = topology
        self.mode               = mode
        self.interact_slots     = interact_slots
        self.max_episodes       = max_episodes
        self.n_aps              = topology.n_aps
        self._rng               = np.random.default_rng(seed)
        random.seed(seed)

        # Traffic configuration
        self.traffic_type       = traffic_type
        self.poisson_arrival_rate = poisson_arrival_rate
        self.queue_max_size     = queue_max_size

        # ── 공간 정의 ──────────────────────────────────────────
        obs_dim = 6  # [prev_action, d_ave, d95, THP, PER, CS]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_aps, obs_dim), dtype=np.float32
        )
        if mode == "cw":
            self.action_space = spaces.MultiDiscrete([7] * self.n_aps)
        elif mode == "cca":
            self.action_space = spaces.MultiDiscrete([5] * self.n_aps)
        else:  # joint
            self.action_space = spaces.MultiDiscrete([7, 5] * self.n_aps)

        self._aps: list[APState] = []
        self._episode = 0
        self._step    = 0

    # ── public API ─────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._aps = [APState(ap_id=i) for i in range(self.n_aps)]

        # Initialize queue based on traffic type
        if self.traffic_type == "saturated":
            for ap in self._aps:
                ap.pkts_in_queue = self.queue_max_size  # 충분한 패킷
        else:  # poisson
            for ap in self._aps:
                ap.pkts_in_queue = 0  # 도착률 기반으로 생성됨

        self._episode += 1
        self._step = 0
        obs = self._get_obs()
        return obs, {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        actions: shape (n_aps,) for cw/cca, (n_aps*2,) for joint
        """
        self._apply_actions(actions)
        self._simulate(self.interact_slots)
        obs  = self._get_obs()
        rew  = self._compute_reward()
        self._step += 1
        done = self._step >= self.max_episodes
        info = self._get_info()
        self._reset_stats()
        return obs, rew, done, False, info

    # ── 내부 메서드 ─────────────────────────────────────────────

    def _apply_actions(self, actions: np.ndarray) -> None:
        if self.mode == "cw":
            for i, ap in enumerate(self._aps):
                ap.cw_exp = int(actions[i])
                ap.backoff = self._draw_backoff(ap.cw_exp)
        elif self.mode == "cca":
            for i, ap in enumerate(self._aps):
                delta = CCA_DELTAS[int(actions[i])]
                ap.cca_thresh = CCA_DEFAULT + delta
                ap.tx_power   = TX_POWER_DBM - delta
        else:  # joint
            for i, ap in enumerate(self._aps):
                ap.cw_exp     = int(actions[2 * i])
                ap.backoff    = self._draw_backoff(ap.cw_exp)
                delta         = CCA_DELTAS[int(actions[2 * i + 1])]
                ap.cca_thresh = CCA_DEFAULT + delta
                ap.tx_power   = TX_POWER_DBM - delta

    def _simulate(self, n_slots: int) -> None:
        """
        정밀 슬롯 단위(Slot-by-Slot) CSMA/CA 시뮬레이션.
        동시 전송에 의한 패킷 충돌(Collision) 및 Carrier Sensing(백오프 동결)을 완벽히 모델링합니다.
        """
        aps = self._aps
        topo = self.topo

        # 시뮬레이션 상태 관리용 로컬 배열
        tx_rem = [0] * self.n_aps  # 각 AP의 남은 전송 슬롯 수 (0이면 IDLE)
        is_collision = [False] * self.n_aps  # 현재 전송 중인 패킷의 충돌 운명 플래그
        pkt_delay_slots = [0] * self.n_aps  # 현재 패킷이 큐 헤드에서 대기/백오프한 총 슬롯 수

        t = 0
        while t < n_slots:
            # 1. 트래픽 도착 (Poisson 트래픽 적용)
            if self.traffic_type == "poisson":
                for ap in aps:
                    if self._rng.random() < self.poisson_arrival_rate:
                        if ap.pkts_in_queue < self.queue_max_size:
                            ap.pkts_in_queue += 1

            # 2. Carrier Sensing (CS): 각 AP 입장에서 이웃 중 누군가 전송 중인지 확인
            channel_busy = [False] * self.n_aps
            for i in range(self.n_aps):
                for neighbor_id in topo.neighbors[i]:
                    if tx_rem[neighbor_id] > 0:
                        channel_busy[i] = True
                        break

            # 3. 백오프 카운터 다운그레이드 판단
            new_tx_aps = []
            for i, ap in enumerate(aps):
                if tx_rem[i] > 0:
                    continue  # 이미 본인이 전송 중이면 패스
                if ap.pkts_in_queue <= 0:
                    continue  # 보낼 패킷이 없으면 패스

                if channel_busy[i]:
                    # 이웃이 전송 중이면 백오프 동결 (대기 시간 지연은 계속 누적)
                    pkt_delay_slots[i] += 1
                else:
                    # 채널이 깨끗하면 백오프 카운터 감소
                    if ap.backoff > 0:
                        ap.backoff -= 1
                        pkt_delay_slots[i] += 1

                    # 카운터가 0이 되는 순간 '동시 전송 시도 리스트'에 등록
                    if ap.backoff == 0:
                        new_tx_aps.append(i)

            # 4. 신규 전송 처리 및 SINR 기반 충돌(Collision) 판정
            if new_tx_aps:
                for i in new_tx_aps:
                    # 전송 타이머 가동 (Packet Length + DIFS)
                    tx_rem[i] = PKT_LEN + DIFS_SLOTS

                # 동시에 전송을 시작한 AP들을 포함하여 수신단(STA)의 SINR 측정
                for i in new_tx_aps:
                    ap = aps[i]
                    sta_pos = topo.sta_positions[i][0]
                    rx_power = self._rx_power(topo.ap_positions[i], sta_pos, ap.tx_power)

                    # 간섭파 계산 (현재 채널을 점유 중인 모든 AP의 전송 전력 합산)
                    interference = 0.0
                    for j in range(self.n_aps):
                        if j != i and tx_rem[j] > 0:
                            interference += self._rx_power(topo.ap_positions[j], sta_pos, aps[j].tx_power)
                    interference += self._dbm_to_mw(NOISE_DBM)

                    sinr_db = self._mw_to_db(rx_power / interference)

                    # 임계치를 넘지 못하면 이 패킷은 충돌(에러) 예정으로 마킹
                    is_collision[i] = (sinr_db < SINR_THRESH_DB)

            # 5. 슬롯 종료 및 전송 완료 처리
            for i, ap in enumerate(aps):
                if tx_rem[i] > 0:
                    tx_rem[i] -= 1
                    if tx_rem[i] == 0:  # 패킷 전송이 완료된 시점
                        # ── [수정] Saturated 트래픽일 때는 큐를 감소시키지 않고 항상 무한 공급 ──
                        if self.traffic_type != "saturated":
                            ap.pkts_in_queue -= 1
                        # ─────────────────────────────────────────────────────────────────

                        if is_collision[i]:
                            ap.pkts_collided += 1
                            ap.cw_exp = min(ap.cw_exp + 1, 6)  # BEB 페널티 적용
                        else:
                            ap.pkts_sent += 1
                            # 실제 지연 = (대기/백오프 슬롯 + 순수 전송 슬롯) * 슬롯 시간
                            total_delay = (pkt_delay_slots[i] + PKT_LEN + DIFS_SLOTS) * SLOT_US
                            ap.delays.append(total_delay)

                        # 전송이 끝났으므로 다음 패킷을 위한 백오프 및 지연 타이머 리셋
                        ap.backoff = self._draw_backoff(ap.cw_exp)
                        pkt_delay_slots[i] = 0

            t += 1  # 1슬롯씩 정밀하게 이동

    def _simulate_(self, n_slots: int) -> None:
        """
        슬롯 단위 CSMA/CA 시뮬레이션 (Traffic 모델 포함).
        모든 AP가 동시에 backoff 카운터를 감소시키고,
        0에 도달한 AP들이 동시에 전송을 시도한다.
        """
        aps = self._aps
        topo = self.topo

        for ap in aps:
            if ap.backoff == 0:
                ap.backoff = self._draw_backoff(ap.cw_exp)

        t = 0
        while t < n_slots:
            # ── Traffic arrival (Poisson only) ──────────────────
            if self.traffic_type == "poisson":
                for ap in aps:
                    if self._rng.random() < self.poisson_arrival_rate:
                        if ap.pkts_in_queue < self.queue_max_size:
                            ap.pkts_in_queue += 1

            # 채널이 idle한 AP만 backoff 카운터 감소
            transmitting = [ap for ap in aps if ap.backoff == 0]

            if not transmitting:
                # 모든 AP backoff 감소
                for ap in aps:
                    ap.backoff = max(0, ap.backoff - 1)
                t += 1
                continue

            # CS: 이웃 AP 중 transmitting이 있으면 backoff 동결
            active_set: list[APState] = []
            for ap in transmitting:
                neighbors_tx = [
                    o for o in transmitting
                    if o.ap_id != ap.ap_id
                    and o.ap_id in topo.neighbors[ap.ap_id]
                ]
                # ── Queue check: 보낼 패킷이 있어야 함 ──
                if not neighbors_tx and ap.pkts_in_queue > 0:
                    active_set.append(ap)

            if not active_set:
                # 모두 동결
                t += 1
                continue

            # 동시 전송 시도 → SINR 판정
            for ap in active_set:
                # 최선 STA 선택 (nearest)
                sta_pos   = topo.sta_positions[ap.ap_id][0]
                rx_power  = self._rx_power(
                    topo.ap_positions[ap.ap_id], sta_pos, ap.tx_power
                )
                interferers = [
                    o for o in active_set if o.ap_id != ap.ap_id
                ]
                interference = sum(
                    self._rx_power(
                        topo.ap_positions[o.ap_id], sta_pos, o.tx_power
                    )
                    for o in interferers
                ) + self._dbm_to_mw(NOISE_DBM)

                sinr_db = self._mw_to_db(rx_power / interference)

                pkt_start = t
                if sinr_db >= SINR_THRESH_DB:
                    ap.pkts_sent += 1
                    ap.pkts_in_queue -= 1  # ← 패킷 전송 완료
                    delay = (PKT_LEN + DIFS_SLOTS + ap.backoff) * SLOT_US
                    ap.delays.append(delay)
                else:
                    ap.pkts_collided += 1
                    ap.pkts_in_queue -= 1  # ← collision도 패킷 손실
                    ap.cw_exp = min(ap.cw_exp + 1, 6)  # BEB

                ap.backoff = self._draw_backoff(ap.cw_exp)

            t += PKT_LEN
            # 전송 중 다른 AP backoff 동결 해제
            for ap in aps:
                if ap not in active_set:
                    ap.backoff = max(0, ap.backoff - 1)

    def _compute_reward(self) -> float:
        """논문 수식 (5): relative reward"""
        # CSMA/CA baseline은 reset 후 첫 step에 저장된 값 참조
        # (학습 중 offline simulation 값 사용 — 여기선 근사 고정값)
        # 실제 구현 시 configs에서 주입
        thp_sum   = sum(self._ap_thp(ap) for ap in self._aps)
        per_mean  = np.mean([self._ap_per(ap) for ap in self._aps])
        d95_mean  = np.mean([self._ap_d95(ap) for ap in self._aps])

        thp_base  = getattr(self, "_csma_thp", thp_sum * 0.70)
        per_base  = getattr(self, "_csma_per", per_mean * 1.20)

        alpha, beta = 10.0, 10.0 / max(1, self.n_aps)
        m, n = 0.98, 0.95

        if thp_sum >= m * thp_base and per_mean <= n * per_base:
            return float(alpha - beta * d95_mean)
        return -1.0

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_aps, 6), dtype=np.float32)
        for i, ap in enumerate(self._aps):
            obs[i] = [
                float(ap.cw_exp if self.mode != "cca" else
                      CCA_DELTAS.index(int(ap.cca_thresh - CCA_DEFAULT))),
                self._ap_dave(ap),
                self._ap_d95(ap),
                self._ap_thp(ap),
                self._ap_per(ap),
                self._ap_cs(ap),
            ]
        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "d95":  [self._ap_d95(ap)  for ap in self._aps],
            "thp":  [self._ap_thp(ap)  for ap in self._aps],
            "per":  [self._ap_per(ap)  for ap in self._aps],
        }

    def _reset_stats(self) -> None:
        for ap in self._aps:
            ap.pkts_sent = ap.pkts_collided = 0
            ap.delays.clear()

    # ── 통계 헬퍼 ──────────────────────────────────────────────

    def _ap_d95(self, ap: APState) -> float:
        if not ap.delays:
            return 0.0
        return float(np.percentile(ap.delays, 95))

    def _ap_dave(self, ap: APState) -> float:
        return float(np.mean(ap.delays)) if ap.delays else 0.0

    def _ap_thp(self, ap: APState) -> float:
        total = ap.pkts_sent + ap.pkts_collided
        return ap.pkts_sent / total if total > 0 else 0.0

    def _ap_per(self, ap: APState) -> float:
        total = ap.pkts_sent + ap.pkts_collided
        return ap.pkts_collided / total if total > 0 else 0.0

    def _ap_cs(self, ap: APState) -> float:
        """이웃 AP들의 평균 수신 전력 (dBm) — carrier sense power"""
        topo = self.topo
        neighbors = topo.neighbors[ap.ap_id]
        if not neighbors:
            return float(CCA_DEFAULT)
        powers = [
            self._mw_to_db(
                self._rx_power(
                    topo.ap_positions[nb],
                    topo.ap_positions[ap.ap_id],
                    self._aps[nb].tx_power,
                )
            )
            for nb in neighbors
        ]
        return float(np.mean(powers))

    # ── 물리층 헬퍼 ────────────────────────────────────────────

    @staticmethod
    def _draw_backoff(cw_exp: int) -> int:
        cw = 2 ** (4 + cw_exp)
        return random.randint(0, cw - 1)

    @staticmethod
    def _rx_power(tx_pos: np.ndarray, rx_pos: np.ndarray,
                  tx_power_dbm: float) -> float:
        """로그 거리 경로 손실 모델 (논문 Table I: -46.67 - 30·log10(D))"""
        dist = float(np.linalg.norm(tx_pos - rx_pos))
        dist = max(dist, 0.5)  # 수치 안정
        pl_db = -46.67 - 30.0 * math.log10(dist)
        rx_dbm = tx_power_dbm + pl_db
        return OBSSEnv._dbm_to_mw(rx_dbm)

    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_db(mw: float) -> float:
        return 10.0 * math.log10(max(mw, 1e-30))