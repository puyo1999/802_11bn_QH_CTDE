"""
src/envs/obss_env.py (Refactored for CTDE & Core 4-State)
IEEE 802.11bn OBSS 채널 접근 Gymnasium 환경
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.topology import Topology

# ── [Practical Setup] 실제 규격 기반 물리층 상수 ────────────────
SLOT_US = 9e-6          # 802.11 표준 Slot Time (9 us) 동일
DIFS_SLOTS = 2          # DIFS = 2 slots (18 us) 동일

# 1. 패킷 길이 (Packet Length)
PKT_LEN = 40            # ◀ 실제 1500 Byte 데이터 패킷 전송 시 점유 시간 계산
                        # 1500 Byte / MCS 5 (정상 전송 속도) 연산 시 약 30~50 slots가 매칭됩니다.
                        # 120 slots는 패킷이 너무 비대해져 비현실적인 충돌을 유발합니다.

# 2. 전송 출력 (Tx Power)
TX_POWER_DBM = 15.0     # ◀ 실내 엔터프라이즈 AP / 가정용 Wi-Fi의 표준 송신 출력 (15 ~ 20 dBm)

# 3. 에너지 감지 임계치 (CCA-ED threshold)
CCA_DEFAULT = -82.0     # 표준 규격상 기본 CCA-ED (위반 시 백오프 돌입 기준)

# 4. 수신 감도 및 패킷 디코딩 SINR 임계치
SINR_THRESH_DB = 10.0   # ◀ [핵심] 실제 MCS 4 ~ 5 대역에서 패킷 성공을 보장하는 SINR 임계치.
                        # 14dB나 18dB는 수신 환경이 지나치게 가혹하여 조금만 멀어져도 깨지며,
                        # 3dB는 간섭이 가득해도 다 성공해버리는 비현실적 디코딩입니다. 10.0dB가 황금 밸런스입니다.

NOISE_DBM = -95.0       # Thermal Noise Floor + Noise Figure 반영 값 동일

TX_POWER_DBM = 10.0  # dBm default
#TX_POWER_DBM = 18.0     # ◀ [기존 10.0 -> 18.0] 전송 출력을 높여 상호 간섭(OBSS) 반경을 대폭 확대
CCA_DEFAULT = -82.0  # dBm

SINR_THRESH_DB = 14.0  # MCS 5 threshold
# ◀ [기존 14.0 -> 18.0] MCS 임계치를 높여 패킷 성공 조건(SINR 요구량)을 매우 까다롭게 변경
#SINR_THRESH_DB = 18.0
#SINR_THRESH_DB = 5.0
CW_EXPS = list(range(7))  # [0,1,2,3,4,5,6]
CCA_DELTAS = [0, 5, 10, 15, 20]  #

@dataclass
class APState:
    """단일 AP의 내부 상태"""
    ap_id: int
    cw_exp: int = 0  # 현재 CW 지수 b
    backoff: int = 0  # 남은 backoff counter
    tx_power: float = TX_POWER_DBM
    cca_thresh: float = CCA_DEFAULT
    pkts_sent: int = 0  # 성공 패킷 수
    pkts_collided: int = 0  # 충돌 패킷 수
    delays: list = field(default_factory=list)
    pkts_in_queue: int = 0  # 큐 적체량


class OBSSEnv(gym.Env):
    """
    IEEE 802.11bn OBSS Gymnasium 환경 (QH-CTDE 논문 수식 정렬 버전)

    Observation space (per AP): 5차원 — 논문 식 (3)
        [ρ_i, β_i, r̄_i, p̄_i, τ_ratio]
        ρ_i     : channel busy ratio (PER 근사)
        β_i     : normalized buffer occupancy
        r̄_i     : mean neighbor RSSI, [0,1] 정규화
        p̄_i     : mean neighbor Tx power, [0,1] 정규화
        τ_ratio : backhaul delay ratio τ_sig/τ_coh (harsh=1.5 / normal=0.3)

    Reward: 논문 식 (5) R^t = R_thr + λ_f·R_fair + λ_l·R_lat
        R_thr  : 비례공정 로그합 (success-rate SINR proxy)
        R_fair : Jain's fairness index − 1
        R_lat  : 지수 UHR latency 패널티 (D_max = 5 ms)

    Action space:
        mode='cw'    -> MultiDiscrete([7] * n_aps)
        mode='cca'   -> MultiDiscrete([5] * n_aps)
        mode='joint' -> MultiDiscrete([7, 5] * n_aps)
    """

    def __init__(
            self,
            topology: Topology,
            mode: str = "joint",
            interact_slots: int = 55_556,
            max_episodes: int = 600,
            seed: int | None = None,
            traffic_type: str = "saturated",
            poisson_arrival_rate: float = 0.1,
            queue_max_size: int = 100,
    ):
        super().__init__()
        self.topo = topology
        self.mode = mode
        self.interact_slots = interact_slots
        self.max_episodes = max_episodes
        self.n_aps = topology.n_aps
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

        self.traffic_type = traffic_type
        self.poisson_arrival_rate = poisson_arrival_rate
        self.queue_max_size = queue_max_size

        # ── 논문 식 (3): 5차원 State 공간 ──
        # [ρ_i, β_i, r̄_i, p̄_i, τ_ratio]
        obs_dim = 5
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(self.n_aps, obs_dim), dtype=np.float32
        )

        if mode == "cw":
            self.action_space = spaces.MultiDiscrete([7] * self.n_aps)
        elif mode == "cca":
            self.action_space = spaces.MultiDiscrete([5] * self.n_aps)
        else:
            self.action_space = spaces.MultiDiscrete([7, 5] * self.n_aps)

        self._aps: list[APState] = []
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._aps = [APState(ap_id=i) for i in range(self.n_aps)]

        if self.traffic_type == "saturated":
            for ap in self._aps:
                ap.pkts_in_queue = self.queue_max_size
        else:
            for ap in self._aps:
                ap.pkts_in_queue = 0

        self._step = 0
        return self._get_obs(), {}

    def step_(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_actions(actions)
        self._simulate(self.interact_slots)

        obs = self._get_obs()
        rew = self._compute_reward()  # 연속 협력형 시스템 리워드 계산

        self._step += 1
        done = self._step >= self.max_episodes
        info = self._get_info()
        self._reset_stats()
        return obs, rew, done, False, info

    # src/envs/obss_env.py 내부의 step 메서드 수정 예시

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. 에이전트들의 자원 관리 행동(CW, OBSS-PD) 적용
        self._apply_actions(actions)

        # ── [추가] 동적 모드 스위칭: 에피소드 절반이 지나면 가혹 조건(고간섭) 활성화 ──
        # 물리 시뮬레이션(_simulate) 전에 AP들의 배경 간섭이나 출력을 동적으로 변동시킵니다.
        is_harsh_mode = self._step >= (self.max_episodes // 2)

        if is_harsh_mode:
            # [논문 검증용 가혹 조건 주입]
            # 모든 AP가 인지하는 기본 노이즈 및 주변 OBSS 버스트 트래픽 간섭을 강제로 10배(10dB) 증폭
            # 만약 기존 환경 코드에 전역 간섭 계수가 있다면 곱해주거나, 시뮬레이터 내부 로직이 참조하도록 설정합니다.
            # 예시: 시뮬레이터 내부에서 참조할 동적 플래그 설정
            self.is_harsh_interference = True
        else:
            self.is_harsh_interference = False

        # 2. 무선 채널 슬롯 시뮬레이션 수행
        # (이 내부에서 SINR을 계산할 때 self.is_harsh_interference가 True면 Interference *= 10.0 처리가 되도록 연동)
        self._simulate(self.interact_slots)

        # 3. 차기 4차원 관측치(State) 및 시스템 협력 보상 수집
        obs = self._get_obs()
        rew = self._compute_reward()  # 연속 협력형 시스템 리워드 계산

        # 4. 타임스텝 업데이트 및 종료 조건 판단
        self._step += 1
        done = self._step >= self.max_episodes
        info = self._get_info()

        # [추가] 정보 전달용 dict(info)에 현재 환경 모드 상태를 기록 (디버깅 및 시각화용)
        info["is_harsh_mode"] = is_harsh_mode

        # 5. 통계치 초기화 후 반환
        self._reset_stats()
        return obs, rew, done, False, info

    def _apply_actions(self, actions: np.ndarray) -> None:
        if self.mode == "cw":
            for i, ap in enumerate(self._aps):
                ap.cw_exp = int(actions[i])
                ap.backoff = self._draw_backoff(ap.cw_exp)
        elif self.mode == "cca":
            for i, ap in enumerate(self._aps):
                delta = CCA_DELTAS[int(actions[i])]
                ap.cca_thresh = CCA_DEFAULT + delta
                ap.tx_power = TX_POWER_DBM - delta
        else:
            for i, ap in enumerate(self._aps):
                ap.cw_exp = int(actions[2 * i])
                ap.backoff = self._draw_backoff(ap.cw_exp)
                delta = CCA_DELTAS[int(actions[2 * i + 1])]
                ap.cca_thresh = CCA_DEFAULT + delta
                ap.tx_power = TX_POWER_DBM - delta

    def _simulate(self, n_slots: int) -> None:
        aps = self._aps
        topo = self.topo
        tx_rem = [0] * self.n_aps
        is_collision = [False] * self.n_aps
        pkt_delay_slots = [0] * self.n_aps

        t = 0
        while t < n_slots:
            if self.traffic_type == "poisson":
                for ap in aps:
                    if self._rng.random() < self.poisson_arrival_rate:
                        if ap.pkts_in_queue < self.queue_max_size:
                            ap.pkts_in_queue += 1

            channel_busy = [False] * self.n_aps
            for i in range(self.n_aps):
                for neighbor_id in topo.neighbors[i]:
                    if tx_rem[neighbor_id] > 0:
                        channel_busy[i] = True
                        break

            new_tx_aps = []
            for i, ap in enumerate(aps):
                if tx_rem[i] > 0:
                    continue
                if ap.pkts_in_queue <= 0:
                    continue

                if channel_busy[i]:
                    pkt_delay_slots[i] += 1
                else:
                    if ap.backoff > 0:
                        ap.backoff -= 1
                        pkt_delay_slots[i] += 1
                    if ap.backoff == 0:
                        new_tx_aps.append(i)

            if new_tx_aps:
                for i in new_tx_aps:
                    tx_rem[i] = PKT_LEN + DIFS_SLOTS

                for i in new_tx_aps:
                    ap = aps[i]
                    sta_pos = topo.sta_positions[i][0]
                    rx_power = self._rx_power(topo.ap_positions[i], sta_pos, ap.tx_power)

                    # 1. 이웃 AP 간섭 합산
                    interference = 0.0
                    for j in range(self.n_aps):
                        if j != i and tx_rem[j] > 0:
                            interference += self._rx_power(topo.ap_positions[j], sta_pos, aps[j].tx_power)

                    # 2. 가혹 조건: 간섭만 10배 증폭 (노이즈 가산 전에 적용)
                    if getattr(self, 'is_harsh_interference', False):
                        interference *= 10.0

                    # 3. 열잡음 1회만 가산 (버그 수정: 기존 2회 → 1회)
                    interference += self._dbm_to_mw(NOISE_DBM)

                    # 3. 최종 SINR 계산 및 충돌 여부 판정
                    sinr_db = self._mw_to_db(rx_power / interference)
                    is_collision[i] = (sinr_db < SINR_THRESH_DB)

            for i, ap in enumerate(aps):
                if tx_rem[i] > 0:
                    tx_rem[i] -= 1
                    if tx_rem[i] == 0:
                        # poisson 모드만 성공 시 큐 차감
                        # saturated 모드: 큐는 항상 queue_max_size 유지 (재충전)
                        if self.traffic_type == "poisson":
                            ap.pkts_in_queue = max(0, ap.pkts_in_queue - 1)

                        if is_collision[i]:
                            ap.pkts_collided += 1
                            ap.cw_exp = min(ap.cw_exp + 1, 6)
                        else:
                            ap.pkts_sent += 1
                            total_delay = (pkt_delay_slots[i] + PKT_LEN + DIFS_SLOTS) * SLOT_US
                            ap.delays.append(total_delay)

                        ap.backoff = self._draw_backoff(ap.cw_exp)
                        pkt_delay_slots[i] = 0

            t += 1

    # ── 논문 식 (5): R^t = R_thr + λ_f·R_fair + λ_l·R_lat ─────────
    def _compute_reward(self) -> float:
        lambda_f = 0.3   # fairness 가중치
        lambda_l = 0.5   # latency 가중치
        lambda_exp = 200.0  # 지수 패널티 스케일 (식 8)
        D_max = 5e-3     # UHR 지연 기한 5 ms

        # ── R_thr: 논문 식 (6) — 비례공정 로그합 ──────────────────
        # log(1 + SINR) 근사: pkts_sent 기반 정규화 airtime을 SINR proxy로 사용
        # (실제 SINR은 _simulate 내부에서만 계산되므로 per-step 캐싱 불가)
        sinr_proxies = []
        for ap in self._aps:
            total = ap.pkts_sent + ap.pkts_collided
            success_rate = ap.pkts_sent / total if total > 0 else 0.0
            # success_rate ∈ [0,1] → SINR proxy: 성공률이 높을수록 높은 SINR
            sinr_proxy = success_rate / (1.0 - success_rate + 1e-6)
            sinr_proxies.append(sinr_proxy)

        R_thr = sum(math.log1p(s) for s in sinr_proxies)

        # ── R_fair: 논문 식 (7) — Jain's fairness index − 1 ────────
        eta = [math.log1p(s) for s in sinr_proxies]  # throughput proxy
        eta_sum = sum(eta)
        eta_sq_sum = sum(e**2 for e in eta)
        N = self.n_aps
        if eta_sq_sum > 0:
            R_fair = (eta_sum ** 2) / (N * eta_sq_sum) - 1.0
        else:
            R_fair = -1.0

        # ── R_lat: 논문 식 (8) — 지수 latency 패널티 ───────────────
        R_lat = 0.0
        for ap in self._aps:
            d95 = self._ap_d95(ap)
            if d95 > 0:
                R_lat -= math.exp(lambda_exp * (d95 - D_max))
            # d95 == 0 (패킷 없음): 패널티 없음

        R_lat = max(R_lat, -1e6)  # 수치 안정성 클리핑

        return float(R_thr + lambda_f * R_fair + lambda_l * R_lat)

    # ── 논문 식 (3): 5차원 State 벡터 ──────────────────────────────
    # o_i^t = [ρ_i, β_i, r̄_i, p̄_i, τ_sig/τ_coh]
    #   ρ_i  : channel busy ratio  (충돌률로 근사 — 실제 채널 점유 추적 불가)
    #   β_i  : normalized buffer occupancy
    #   r̄_i  : mean neighbor RSSI (dBm, 정규화)
    #   p̄_i  : mean neighbor Tx power (dBm, 정규화)
    #   τ_ratio: backhaul delay ratio (시뮬레이션에서는 harsh mode 플래그로 근사)
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_aps, 5), dtype=np.float32)
        for i, ap in enumerate(self._aps):
            # ρ_i: channel busy ratio — PER로 근사 (충돌 = 채널 경쟁 과부하)
            rho = self._ap_per(ap)

            # β_i: 정규화된 버퍼 점유율
            beta = float(ap.pkts_in_queue / self.queue_max_size)

            # r̄_i: 이웃 AP들의 평균 수신 RSSI (dBm), [-100, 0] → [0, 1] 정규화
            r_bar = self._ap_neighbor_rssi_norm(i)

            # p̄_i: 이웃 AP들의 평균 송신 전력 (dBm), [0, 20] → [0, 1] 정규화
            p_bar = self._ap_neighbor_txpower_norm(i)

            # τ_sig/τ_coh: harsh mode = 1.5 (임계 초과), normal = 0.3 (여유)
            tau_ratio = 1.5 if getattr(self, 'is_harsh_interference', False) else 0.3

            obs[i] = [rho, beta, r_bar, p_bar, tau_ratio]
        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "d95": [self._ap_d95(ap) for ap in self._aps],
            "thp": [ap.pkts_sent for ap in self._aps],  # 누적 성공 패킷 카운트 직접 제공
            "per": [self._ap_per(ap) for ap in self._aps],
        }

    def _reset_stats(self) -> None:
        for ap in self._aps:
            ap.pkts_sent = ap.pkts_collided = 0
            ap.delays.clear()

    def _ap_d95(self, ap: APState) -> float:
        if not ap.delays: return 0.0
        return float(np.percentile(ap.delays, 95))

    def _ap_per(self, ap: APState) -> float:
        total = ap.pkts_sent + ap.pkts_collided
        return ap.pkts_collided / total if total > 0 else 0.0

    def _ap_cs(self, ap: APState) -> float:
        topo = self.topo
        neighbors = topo.neighbors[ap.ap_id]
        if not neighbors: return float(CCA_DEFAULT)
        powers = [
            self._mw_to_db(self._rx_power(topo.ap_positions[nb], topo.ap_positions[ap.ap_id], self._aps[nb].tx_power))
            for nb in neighbors
        ]
        return float(np.mean(powers))

    def _ap_neighbor_rssi_norm(self, ap_idx: int) -> float:
        """
        논문 식 (3): r̄_i — 이웃 AP들의 평균 수신 RSSI, [0,1] 정규화
        정규화: (rssi_dbm - (-100)) / (0 - (-100))
        """
        topo = self.topo
        neighbors = topo.neighbors[ap_idx]
        if not neighbors:
            return 0.0
        rssi_vals = []
        for nb in neighbors:
            rx_mw = self._rx_power(
                topo.ap_positions[nb],
                topo.ap_positions[ap_idx],
                self._aps[nb].tx_power
            )
            rssi_dbm = 10.0 * math.log10(max(rx_mw, 1e-30))
            rssi_vals.append(rssi_dbm)
        mean_rssi = float(np.mean(rssi_vals))
        return float(np.clip((mean_rssi + 100.0) / 100.0, 0.0, 1.0))

    def _ap_neighbor_txpower_norm(self, ap_idx: int) -> float:
        """
        논문 식 (3): p̄_i — 이웃 AP들의 평균 송신 전력, [0,1] 정규화
        정규화: tx_power_dbm / 20.0  (최대 20 dBm 가정)
        """
        topo = self.topo
        neighbors = topo.neighbors[ap_idx]
        if not neighbors:
            return 0.0
        powers = [self._aps[nb].tx_power for nb in neighbors]
        return float(np.clip(np.mean(powers) / 20.0, 0.0, 1.0))

    @staticmethod
    def _draw_backoff(cw_exp: int) -> int:
        cw = 2 ** (4 + cw_exp)
        return random.randint(0, cw - 1)

    @staticmethod
    def _rx_power(tx_pos: np.ndarray, rx_pos: np.ndarray, tx_power_dbm: float) -> float:
        dist = float(np.linalg.norm(tx_pos - rx_pos))
        dist = max(dist, 0.5)
        pl_db = -46.67 - 30.0 * math.log10(dist)
        rx_dbm = tx_power_dbm + pl_db
        return OBSSEnv._dbm_to_mw(rx_dbm)

    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_db(mw: float) -> float:
        return 10.0 * math.log10(max(mw, 1e-30))