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
# MCS가 높을수록 전송량(Capacity)은 크지만, 요구되는 SINR 임계치도 높아짐.
# (Implicit Power-Distance Coupling을 RL이 스스로 학습하도록 유도)
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
    논문(Section 4.1)의 수학적 모델을 완벽히 구현한 스케줄링 기반 환경.
    CW가 제거되고, Action은 (STA_idx, MCS_idx)를 선택하는 MultiDiscrete 공간.
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
            # ── [이름 맞춤] 기존 코드의 명칭인 k_stas를 기본으로 수용 ──
            k_stas: int = 4,
            k_stas_per_ap: int | None = None,  # yaml 설정에서 k_stas_per_ap로 들어올 경우를 대비한 예비 인자
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

        # ── [핵심 수정] 변수명 충돌 및 AttributeError 방지 예외 처리 ──
        # YAML에 설정된 값(k_stas_per_ap)이 있으면 그것을 쓰고, 없으면 기본 파라미터(k_stas)를 사용합니다.
        if k_stas_per_ap is not None:
            self.k_stas = k_stas_per_ap
        else:
            self.k_stas = k_stas

        # 내부 코드 호환성을 위해 두 이름 모두 멤버 변수로 저장해 둡니다.
        self.k_stas_per_ap = self.k_stas

        # ── 모드별 State 차원 자동 정렬 (self.k_stas 기준으로 변경) ──
        if mode == "qh":
            # 총 차원 = 4 + K (K=4 이면 딱 8차원 정의)
            obs_dim = 4 + self.k_stas
        else:
            obs_dim = 4  # 기존 베이스라인 용 공간

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_aps, obs_dim), dtype=np.float32
        )

        # ── 모드별 Action 공간 평탄화(Flat) 매핑 ──
        if mode == "cw":
            self.action_space = spaces.MultiDiscrete([7] * self.n_aps)
        elif mode == "cca":
            self.action_space = spaces.MultiDiscrete([5] * self.n_aps)
        elif mode == "joint":
            self.action_space = spaces.MultiDiscrete([7, 5] * self.n_aps)
        elif mode == "qh":
            # CW 선택지 9개, MCS 선택지 16개 구조를 플랫하게 반복
            self.action_space = spaces.MultiDiscrete([9, 16] * self.n_aps)

        # ── 이제 이 라인(125번 라인) 실행 시 self.k_stas를 정상적으로 참조하여 에러가 나지 않습니다 ──
        self._aps = [APState(ap_id=i, k_stas=self.k_stas) for i in range(self.n_aps)]
        self._step = 0

    def _generate_sta_positions(self):
        """AP 반경 내에 K개의 STA를 거리에 따른 다양성을 주어 배치"""
        self.sta_positions = []
        for i in range(self.n_aps):
            ap_pos = self.topo.ap_positions[i]
            stas = []
            for k in range(self.k_stas):
                # 거리를 다양하게 배치 (가까운 유저 ~ 셀 엣지 유저)
                radius = 1.0 + (k / self.k_stas) * 15.0
                angle = self._rng.uniform(0, 2 * math.pi)
                stas.append(ap_pos + np.array([radius * math.cos(angle), radius * math.sin(angle)]))
            self.sta_positions.append(stas)

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
        # 1. 큐 진화 (Queue Evolution - Equation for Arrivals)
        for ap in self._aps:
            arrivals = self._rng.poisson(self.poisson_lambda, size=self.k_stas)
            ap.queues = np.clip(ap.queues + arrivals, 0, self.queue_max_size)

        # 2. Action 파싱 (Target STA 및 MCS 추출)
        target_stas = []
        selected_mcs = []
        for i in range(self.n_aps):
            # 행동 마스킹 로직: 큐가 모두 비어있으면 전송 쉼(Dummy STA)
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

            # [수정] 이번 슬롯에서 해당 AP가 전송을 쉬기로 했다면 (-1), 계산을 건너뜀
            if target_sta == -1:
                continue

            # [수정] 에이전트가 선택한 target_sta의 실제 위치 가져오기 (토폴로지 범위 방어벽)
            if target_sta < len(self.topo.sta_positions[i]):
                sta_pos = self.topo.sta_positions[i][target_sta]
            else:
                # 존재하지 않는 더미 STA를 선택한 경우, 시스템에 간섭을 주지 않도록 격리 조치
                sta_pos = [9999.0, 9999.0]

            # 수신 전력 계산 (선택된 target_sta 기준)
            rx_power = self._rx_power(self.topo.ap_positions[i], sta_pos, TX_POWER_DBM)

            # 인접 OBSS 간섭 합산
            interference = self._dbm_to_mw(NOISE_DBM)
            for j in range(self.n_aps):
                # 다른 AP(j)가 활성화되어 있고, 전송 중일 때 발생하는 간섭 누적
                if j != i and target_stas[j] != -1:
                    interference += self._rx_power(self.topo.ap_positions[j], sta_pos, TX_POWER_DBM)

            # 가혹 조건 모드 (Episode 후반부에 간섭 증폭 - 논문 검증용)
            if self._step >= (self.max_episodes // 2):
                interference *= 5.0

            sinr_db = self._mw_to_db(rx_power / interference)
            mcs_idx = selected_mcs[i]

            # 성공 여부 판별 (MCS 요구 임계치 만족 확인)
            if sinr_db >= MCS_SINR_THRESH[mcs_idx]:
                # [수정] 유출된 k 대신 에이전트가 지정한 target_sta의 큐에서 차감
                tx_pkts = min(self._aps[i].queues[target_sta], MCS_CAPACITY[mcs_idx])
                self._aps[i].queues[target_sta] -= tx_pkts  # 패킷 전송 성공 (버퍼 감소)
                self._aps[i].pkts_sent += tx_pkts
                successful_tx[i] = tx_pkts
            else:
                self._aps[i].pkts_dropped += 1  # 충돌/디코딩 실패

        # 4. State 갱신 및 보상(Reward) 계산
        obs = self._get_obs()
        rew = self._compute_reward(successful_tx)

        self._step += 1
        done = self._step >= self.max_episodes

        # 5. 대조군 스크립트 연동을 위한 결과 패킹
        info = {
            "is_harsh_mode": self._step >= (self.max_episodes // 2),
            "system_throughput": sum(successful_tx),
            "thp": successful_tx,  # 👈 [수정] self.successful_tx 오타를 지역 변수 성공 스펙으로 변경
        }

        return obs, rew, done, False, info
    '''
    def _compute_reward(self, successful_tx: list[int]) -> float:
        """
        논문에 정의된 Reward Function
        1. R_thr: 성공적인 전송에 대한 보상 (Throughput)
        2. R_fair: 큐 적체(Saturation)에 대한 강력한 패널티 (Fairness/Latency 강제)
        """
        w_thp = 1.0
        w_queue_penalty = 2.0

        system_reward = 0.0
        for i, ap in enumerate(self._aps):
            # 처리량 보상
            system_reward += w_thp * successful_tx[i]

            # 큐 공평성 패널티 (특정 STA의 큐가 꽉 차면 지수적 패널티 부여)
            queue_ratios = ap.queues / self.queue_max_size
            queue_penalty = np.sum(queue_ratios ** 2)  # 버퍼가 찰수록 급격히 패널티 증가

            system_reward -= w_queue_penalty * queue_penalty

        return float(system_reward)
    '''
    def _compute_reward(self, successful_tx: list[int]) -> float:
        w_thp = 1.0

        # 💡 [핵심 튜닝] 기존 2.0 -> 0.2 로 하향 조정
        # 밀집 환경(Topo6)에서 큐가 쌓이더라도 전송을 포기하지 않고
        # 끝까지 Throughput 확보에 집중하도록 패널티의 공포를 줄여줍니다.
        w_queue_penalty = 0.2

        system_reward = 0.0
        for i, ap in enumerate(self._aps):
            # 1. 처리량 보상 (가장 중요한 목표)
            system_reward += w_thp * successful_tx[i]

            # 2. 큐 공평성 패널티 (특정 STA의 큐가 꽉 차면 지수적 패널티 부여)
            # 큐 비율은 0~1 사이이므로 제곱하면 작은 값은 더 작아지고 꽉 찰 때만 커집니다.
            queue_ratios = ap.queues / self.queue_max_size
            queue_penalty = np.sum(queue_ratios ** 2)

            # 패널티 차감
            system_reward -= w_queue_penalty * queue_penalty

        return system_reward
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_aps, 2 * self.k_stas + 1), dtype=np.float32)
        for i, ap in enumerate(self._aps):
            # 1. 버퍼 비율 (beta)
            q_ratios = ap.queues / self.queue_max_size

            # 2. 채널 퀄리티 프록시 (거리 기반 Base SINR 역산)
            base_sinr = np.zeros(self.k_stas)
            for k in range(self.k_stas):
                # 1. 토폴로지에 실제 존재하는 사용자인지 인덱스 검사 (IndexError 원천 차단)
                if k < len(self.topo.sta_positions[i]):
                    rx_pwr = self._rx_power(self.topo.ap_positions[i], self.topo.sta_positions[i][k], TX_POWER_DBM)
                else:
                    # 2. CW_Topo1처럼 사용자가 1명뿐이라 빈 슬롯(k=1,2,3)이 생기는 경우,
                    # 통신 성능에 영향이 없는 매우 낮은 무선 신호 세기(-110.0 dBm)로 안전하게 패딩 처리
                    rx_pwr = -110.0

                #rx_pwr = self._rx_power(self.topo.ap_positions[i], self.sta_positions[i][k], TX_POWER_DBM)
                #rx_pwr = self._rx_power(self.topo.ap_positions[i], self.topo.sta_positions[i][k], TX_POWER_DBM)

                base_sinr[k] = self._mw_to_db(rx_pwr / self._dbm_to_mw(NOISE_DBM)) / 40.0  # 정규화

            # 3. 딜레이 동역학 상태 (가상 백홀 상태)
            bh_ratio = 1.0 if self._step >= (self.max_episodes // 2) else 0.0

            obs[i] = np.concatenate([q_ratios, base_sinr, [bh_ratio]])

        return obs

    @staticmethod
    def _rx_power(tx_pos: np.ndarray, rx_pos: np.ndarray, tx_power_dbm: float) -> float:
        dist = float(np.linalg.norm(tx_pos - rx_pos))
        dist = max(dist, 1.0)  # 최소 거리 1m 보장
        pl_db = -46.67 - 30.0 * math.log10(dist)  # 6GHz 실내 경로 손실 모델
        rx_dbm = tx_power_dbm + pl_db
        return OBSSEnv._dbm_to_mw(rx_dbm)

    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_db(mw: float) -> float:
        return 10.0 * math.log10(max(mw, 1e-30))