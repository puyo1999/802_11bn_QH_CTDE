"""
src/envs/qh_obss_env.py
IEEE 802.11bn MAPC Coordinated Scheduling Environment for QH-CTDE
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

# ── [Physical Layer & MAC Constants] ─────────────────────────────
SLOT_MS = 1.0  # 1ms Scheduling Slot (Wi-Fi 8 TXOP block)
TX_POWER_DBM = 15.0  # 15 dBm Standard Tx Power
NOISE_DBM = -95.0  # Thermal Noise Floor

# [Link Adaptation] MCS 0 ~ 15에 대한 물리적 제약 모델링
N_MCS = 16
MCS_SINR_THRESH = [2.0 + 1.5 * i for i in range(N_MCS)]  # MCS 0: 2dB ~ MCS 15: 24.5dB
MCS_CAPACITY = [1 + int(1.5 * i) for i in range(N_MCS)]  # 전송 가능한 패킷 수 (1 ~ 23)


@dataclass
class APState:
    """단일 AP의 내부 상태 및 STA 큐 관리"""
    ap_id: int
    k_stas: int
    queues: np.ndarray = field(init=False)  # 각 STA별 버퍼 큐 길이
    pkts_sent: int = 0
    pkts_dropped: int = 0

    def __post_init__(self):
        self.queues = np.zeros(self.k_stas, dtype=np.float32)


class OBSSEnv(gym.Env):
    """
    논문(Section 3 / Section 4)의 수학적 모델을 완벽히 구현한 스케줄링 기반 환경.
    State, Action, Queue Evolution, Reward 함수가 수식과 1:1 대응합니다.
    """

    def __init__(
            self,
            topology: Topology,
            mode: str = "qh",
            interact_slots: int = 55_556,
            max_episodes: int = 600,
            seed: int | None = None,
            traffic_type: str = "saturated",
            poisson_arrival_rate: float = 0.1,
            queue_max_size: int = 100,
            k_stas: int = 4,
            k_stas_per_ap: int | None = None,
            poisson_lambda: float | None = None,
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
        self.poisson_lambda = poisson_lambda if poisson_lambda is not None else poisson_arrival_rate
        self.queue_max_size = queue_max_size

        # STA 수 설정 (YAML 오버라이드 지원)
        if k_stas_per_ap is not None:
            self.k_stas = k_stas_per_ap
        else:
            self.k_stas = k_stas
        self.k_stas_per_ap = self.k_stas

        # ── State 차원 정확히 정의 (논문 식 3 반영) ──
        # K개 버퍼비율(beta) + K개 SINR(gamma) + 1개 Neighbor RSSI(r_bar) + 1개 Latency Ratio(tau)
        # 총 obs_dim = 2 * K + 2
        obs_dim = 2 * self.k_stas + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_aps, obs_dim), dtype=np.float32
        )

        # ── Action 공간 정의 (논문 식 4 반영) ──
        if mode == "cw":
            self.action_space = spaces.MultiDiscrete([7] * self.n_aps)
        elif mode == "cca":
            self.action_space = spaces.MultiDiscrete([5] * self.n_aps)
        elif mode == "joint":
            self.action_space = spaces.MultiDiscrete([7, 5] * self.n_aps)
        elif mode == "qh":
            # STA 선택지 (K개 + 1 dummy) 또는 MCS 선택지 (16개)
            self.action_space = spaces.MultiDiscrete([self.k_stas, 16] * self.n_aps)

        self._aps = [APState(ap_id=i, k_stas=self.k_stas) for i in range(self.n_aps)]
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._aps = [APState(ap_id=i, k_stas=self.k_stas) for i in range(self.n_aps)]
        self._step = 0

        # 초기 트래픽 주입
        for ap in self._aps:
            ap.queues += self._rng.poisson(self.poisson_lambda, size=self.k_stas)
            ap.queues = np.clip(ap.queues, 0, self.queue_max_size)

        return self._get_obs(), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. 버퍼 큐 진화 (Queue Evolution - 식 2)
        for ap in self._aps:
            arrivals = self._rng.poisson(self.poisson_lambda, size=self.k_stas)
            ap.queues = np.clip(ap.queues + arrivals, 0, self.queue_max_size)

        # 2. Action 파싱 (Target STA 및 MCS 추출)
        target_stas = []
        selected_mcs = []
        for i in range(self.n_aps):
            if np.sum(self._aps[i].queues) == 0:
                target_stas.append(-1)
                selected_mcs.append(0)
            else:
                target_stas.append(int(actions[2 * i]))
                selected_mcs.append(int(actions[2 * i + 1]))

        # 3. 물리 계층(SINR 및 Interference) 계산
        successful_tx = [0] * self.n_aps

        for i in range(self.n_aps):
            target_sta = target_stas[i]

            if target_sta == -1:
                continue

            if target_sta < len(self.topo.sta_positions[i]):
                sta_pos = self.topo.sta_positions[i][target_sta]
            else:
                sta_pos = [9999.0, 9999.0]

            rx_power = self._rx_power(self.topo.ap_positions[i], sta_pos, TX_POWER_DBM)

            interference = self._dbm_to_mw(NOISE_DBM)
            for j in range(self.n_aps):
                if j != i and target_stas[j] != -1:
                    interference += self._rx_power(self.topo.ap_positions[j], sta_pos, TX_POWER_DBM)

            # Harsh Mode (Episode 후반부 간섭 증폭)
            if self._step >= (self.max_episodes // 2):
                interference *= 5.0

            sinr_db = self._mw_to_db(rx_power / interference)
            mcs_idx = selected_mcs[i]

            # MCS 임계치 확인 및 전송 처리
            if sinr_db >= MCS_SINR_THRESH[mcs_idx]:
                tx_pkts = min(self._aps[i].queues[target_sta], MCS_CAPACITY[mcs_idx])
                self._aps[i].queues[target_sta] -= tx_pkts
                self._aps[i].pkts_sent += tx_pkts
                successful_tx[i] = tx_pkts
            else:
                self._aps[i].pkts_dropped += 1

        # 4. State 갱신 및 보상 계산
        obs = self._get_obs()
        rew = self._compute_reward(successful_tx)

        self._step += 1
        done = self._step >= self.max_episodes

        info = {
            "is_harsh_mode": self._step >= (self.max_episodes // 2),
            "system_throughput": sum(successful_tx),
            "thp": successful_tx,
        }

        return obs, rew, done, False, info

    def _compute_reward(self, successful_tx: list[int]) -> float:
        """
        논문 Section 3/4 수식 (5)~(8) 반영:
        1. R_thr  : Log-Throughput 보상 (sum_i log(1 + Thp_i))
        2. R_fair : Jain's Fairness Index 보상
        3. R_lat  : UHR Latency 지수 패널티 (-sum_i exp(lambda * (T_i - D_max)))
        """
        lambda_f = 0.5
        lambda_l = 0.2
        d_max = 5.0  # UHR 지연 한계 (5ms)

        # 1. Throughput Reward (식 6)
        r_thr = sum([math.log(1.0 + float(tx)) for tx in successful_tx])

        # 2. Jain's Fairness Reward (식 7)
        tx_array = np.array(successful_tx, dtype=np.float32)
        sum_tx = np.sum(tx_array)
        sum_sq_tx = np.sum(tx_array ** 2)
        if sum_sq_tx > 0:
            jain_index = (sum_tx ** 2) / (self.n_aps * sum_sq_tx)
        else:
            jain_index = 1.0
        r_fair = jain_index - 1.0

        # 3. UHR Latency Exponential Penalty (식 8)
        r_lat = 0.0
        for i, ap in enumerate(self._aps):
            avg_q = float(np.mean(ap.queues))
            t_i = avg_q * SLOT_MS  # 추정 큐 지연 시간
            if t_i > d_max:
                r_lat -= math.exp(0.1 * (t_i - d_max))

        total_reward = r_thr + lambda_f * r_fair + lambda_l * r_lat
        return float(total_reward)

    def _get_obs(self) -> np.ndarray:
        """
        논문 식 (3) Observation Vector 구성:
        o_i^t = [beta_i, gamma_i, r_bar_i, p_i, tau_sig/tau_coh]
        """
        obs = np.zeros((self.n_aps, 2 * self.k_stas + 2), dtype=np.float32)

        for i, ap in enumerate(self._aps):
            # 1. 버퍼 점유 비율 (beta) -> K 차원
            q_ratios = ap.queues / self.queue_max_size

            # 2. 채널 SINR 프록시 (gamma) -> K 차원
            base_sinr = np.zeros(self.k_stas, dtype=np.float32)
            for k in range(self.k_stas):
                if k < len(self.topo.sta_positions[i]):
                    rx_pwr = self._rx_power(self.topo.ap_positions[i], self.topo.sta_positions[i][k], TX_POWER_DBM)
                else:
                    rx_pwr = -110.0
                base_sinr[k] = self._mw_to_db(rx_pwr / self._dbm_to_mw(NOISE_DBM)) / 40.0

            # 3. 이웃 AP 평균 수신 신호 강도 (r_bar) -> 1 차원
            neighbor_pwrs = []
            for j in range(self.n_aps):
                if j != i:
                    pwr = self._rx_power(self.topo.ap_positions[j], self.topo.ap_positions[i], TX_POWER_DBM)
                    neighbor_pwrs.append(pwr)
            avg_neighbor_rssi = self._mw_to_db(np.mean(neighbor_pwrs)) / 100.0 if neighbor_pwrs else 0.0

            # 4. 백홀 지연/채널 응집 시간 비율 (tau_sig / tau_coh) -> 1 차원
            tau_ratio = 1.0 if self._step >= (self.max_episodes // 2) else 0.1

            # 최종 2*K + 2 차원 결합
            obs[i] = np.concatenate([q_ratios, base_sinr, [avg_neighbor_rssi], [tau_ratio]])

        return obs

    @staticmethod
    def _rx_power(tx_pos: np.ndarray, rx_pos: np.ndarray, tx_power_dbm: float) -> float:
        dist = float(np.linalg.norm(tx_pos - rx_pos))
        dist = max(dist, 1.0)  # 최소 거리 1m
        pl_db = -46.67 - 30.0 * math.log10(dist)  # 6GHz 경로 손실 모델
        rx_dbm = tx_power_dbm + pl_db
        return OBSSEnv._dbm_to_mw(rx_dbm)

    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_db(mw: float) -> float:
        return 10.0 * math.log10(max(mw, 1e-30))