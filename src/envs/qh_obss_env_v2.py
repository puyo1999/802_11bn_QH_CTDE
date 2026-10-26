"""
src/envs/qh_obss_env_v2.py
IEEE 802.11bn MAPC 협조 스케줄링 환경 (모듈 조립형).

qh_obss_env_mod1.py의 동작을 보존하되, 물리 계층·간섭·중첩 제약·관측·보상을
개별 모듈로 분리하고 모든 파라미터를 생성자 인자로 노출한다.

    action (mode='qh') : MultiDiscrete([K, n_mcs] * n_aps)
                         AP별로 (대상 STA 인덱스, MCS 인덱스)를 선택한다.
    observation        : (n_aps, 2K + overlap_dim + 2)

중첩 제약은 src/envs/overlap.py의 정책에 위임한다. hard 모드에서 중첩 상대 AP가
동일 슬롯에 전송하면 해당 전송은 SINR 값과 무관하게 실패한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.interference import InterferenceModel
from src.envs.link_adaptation import MCSTable
from src.envs.observation import ObservationBuilder
from src.envs.overlap import OverlapConfig, OverlapPolicy
from src.envs.physical import (
    DEFAULT_NOISE_DBM,
    DEFAULT_TX_POWER_DBM,
    PathLossModel,
)
from src.envs.reward import RewardConfig, compute_reward
from src.envs.traffic import TrafficConfig, TrafficModel
from src.envs.topology import Topology

SLOT_MS = 1.0


@dataclass
class APState:
    """단일 AP의 STA별 버퍼 큐 및 누적 통계."""
    ap_id: int
    k_stas: int
    queues: np.ndarray = field(init=False)
    pkts_sent: int = 0
    pkts_dropped: int = 0
    pkts_blocked: int = 0      # 중첩 제약으로 차단된 전송 횟수

    def __post_init__(self) -> None:
        self.queues = np.zeros(self.k_stas, dtype=np.float32)


class QHOBSSEnv(gym.Env):
    """QH-CTDE 스케줄링 환경."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: Topology,
        mode: str = "qh",
        interact_slots: int = 55_556,
        max_episodes: int = 600,
        seed: int | None = None,
        traffic: dict[str, Any] | TrafficConfig | None = None,
        queue_max_size: int = 100,
        k_stas: int | None = None,
        n_mcs: int = 16,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        noise_dbm: float = DEFAULT_NOISE_DBM,
        harsh_mode_enabled: bool = True,
        harsh_start_ratio: float = 0.5,
        harsh_interference_gain: float = 5.0,
        overlap: dict[str, Any] | OverlapConfig | None = None,
        reward: dict[str, Any] | RewardConfig | None = None,
        allow_mute: bool = False,
    ):
        super().__init__()
        self.topo = topology
        self.mode = mode
        self.interact_slots = interact_slots
        self.max_episodes = max_episodes
        self.n_aps = topology.n_aps
        self.queue_max_size = queue_max_size
        self.traffic_cfg = (
            traffic if isinstance(traffic, TrafficConfig) else TrafficConfig.from_dict(traffic)
        )
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # K는 토폴로지의 최대 STA 수 이상이어야 전 STA를 선택할 수 있다.
        topo_max = topology.max_stas_per_ap()
        self.k_stas = max(k_stas, topo_max) if k_stas is not None else topo_max

        # ── 가혹 조건 파라미터 ──
        self.harsh_mode_enabled = harsh_mode_enabled
        self.harsh_start_ratio = harsh_start_ratio
        self.harsh_interference_gain = harsh_interference_gain
        self.harsh_start_step = int(max_episodes * harsh_start_ratio)
        self.is_harsh_interference = False

        # ── 모듈 조립 ──
        self.overlap_policy = OverlapPolicy(
            topology,
            overlap if isinstance(overlap, OverlapConfig) else OverlapConfig.from_dict(overlap),
        )
        self.interference = InterferenceModel(
            topology=topology,
            overlap_policy=self.overlap_policy,
            path_loss=PathLossModel(),
            tx_power_dbm=tx_power_dbm,
            noise_dbm=noise_dbm,
            harsh_interference_gain=harsh_interference_gain,
        )
        self.mcs_table = MCSTable(n_mcs=n_mcs)
        self.obs_builder = ObservationBuilder(
            n_aps=self.n_aps,
            k_stas=self.k_stas,
            queue_max_size=queue_max_size,
            interference=self.interference,
            overlap=self.overlap_policy,
        )
        self.reward_cfg = (
            reward if isinstance(reward, RewardConfig) else RewardConfig(**(reward or {}))
        )
        self.traffic = TrafficModel(self.traffic_cfg, self._rng)

        # ── 공간 정의 ──
        # allow_mute=True 이면 STA 분기에 인덱스 K(= mute, 이번 슬롯 전송 안 함)를 추가한다.
        # 미존재 STA 인덱스도 전송하지 않지만, 학습 정책은 그 인덱스를 마스킹하므로
        # 명시적 mute 없이는 Co-TDMA 같은 "쉬기" 정책을 표현할 수 없다.
        self.allow_mute = allow_mute
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_aps, self.obs_builder.obs_dim), dtype=np.float32,
        )
        n_sta_actions = self.k_stas + 1 if allow_mute else self.k_stas
        self.action_space = spaces.MultiDiscrete(
            [n_sta_actions, self.mcs_table.n_mcs] * self.n_aps
        )

        # 토폴로지에 실제로 존재하는 STA만 트래픽을 받는다. 패딩 STA 큐는 서비스될 수 없어
        # 상한까지 차면서 지연 보상과 관측(buffer_ratio)에 정책과 무관한 상수를 더했다.
        self.valid_sta_counts = [len(stas) for stas in topology.sta_positions]
        self._arrival_mask = np.zeros((self.n_aps, self.k_stas), dtype=np.float32)
        for i, n_real in enumerate(self.valid_sta_counts):
            self._arrival_mask[i, :n_real] = 1.0

        self._aps = [APState(ap_id=i, k_stas=self.k_stas) for i in range(self.n_aps)]
        self._step = 0

    # ── 프로퍼티 ────────────────────────────────────────────────
    @property
    def obs_dim(self) -> int:
        return self.obs_builder.obs_dim

    def feature_layout(self) -> list[tuple[str, int, int]]:
        return self.obs_builder.feature_layout()

    @property
    def mute_index(self) -> int | None:
        """명시적 mute 행동의 STA 인덱스. allow_mute=False 이면 None."""
        return self.k_stas if self.allow_mute else None

    def queues(self) -> np.ndarray:
        return np.stack([ap.queues for ap in self._aps])

    def peek_arrivals(self) -> np.ndarray:
        """다음 step 의 도착 패킷 수를 RNG 를 소비하지 않고 미리 본다 (genie oracle 전용)."""
        state = self._rng.bit_generator.state
        arrivals = np.stack([self.traffic.arrivals(self.k_stas) * self._arrival_mask[i]
                             for i in range(self.n_aps)])
        self._rng.bit_generator.state = state
        return arrivals

    # ── Gymnasium API ───────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
            self.traffic.set_rng(self._rng)

        self._aps = [APState(ap_id=i, k_stas=self.k_stas) for i in range(self.n_aps)]
        for i, ap in enumerate(self._aps):
            ap.queues = self.traffic.initial_queue(self.k_stas, self.queue_max_size) * self._arrival_mask[i]
        self._step = 0
        self.is_harsh_interference = False
        self.overlap_policy.reset_stats()

        return self.obs_builder.build([ap.queues for ap in self._aps], False), {}

    def step(self, actions: np.ndarray):
        # 1. 큐 진화 — ratio와 무관하게 항상 도착만 주입한다. 차감은 _transmit에서
        #    실제 전송 성공 여부에 따라 이뤄진다 (saturated 강제 고정 없음).
        for i, ap in enumerate(self._aps):
            ap.queues = self.traffic.step_queue(ap.queues, self.queue_max_size, self._arrival_mask[i])

        # 2. 가혹 조건 판정
        self.is_harsh_interference = (
            self.harsh_mode_enabled and self._step >= self.harsh_start_step
        )

        # 3. 액션 파싱 및 전송 처리
        target_stas, selected_mcs = self._parse_actions(actions)
        transmitting = [i for i, t in enumerate(target_stas) if t >= 0]
        successful_tx, blocked_this_step = self._transmit(
            target_stas, selected_mcs, transmitting
        )

        # 4. 관측 및 보상
        obs = self.obs_builder.build([ap.queues for ap in self._aps], self.is_harsh_interference)
        breakdown = compute_reward(
            successful_tx=successful_tx,
            queue_lengths=[ap.queues for ap in self._aps],
            cfg=self.reward_cfg,
            n_overlap_blocked=blocked_this_step,
        )

        self._step += 1
        done = self._step >= self.max_episodes

        info = {
            "is_harsh_mode": self.is_harsh_interference,
            "system_throughput": float(sum(successful_tx)),
            "thp": successful_tx,
            "overlap_blocked_step": blocked_this_step,
            "queue_mean": [float(np.mean(ap.queues)) for ap in self._aps],
            **breakdown.as_dict(),
            **self.overlap_policy.stats(),
        }
        return obs, breakdown.total, done, False, info

    # ── 내부 처리 ────────────────────────────────────────────────

    def _parse_actions(self, actions: np.ndarray) -> tuple[list[int], list[int]]:
        """
        평탄화된 액션을 (대상 STA, MCS) 쌍으로 분해한다.
        큐가 비어 있거나 미존재 STA(또는 mute 인덱스 K)를 지정한 AP는 전송을 수행하지 않는다(-1).
        """
        actions = np.asarray(actions).reshape(-1)
        target_stas: list[int] = []
        selected_mcs: list[int] = []

        for i in range(self.n_aps):
            sta_idx = int(actions[2 * i])
            mcs_idx = self.mcs_table.clamp(int(actions[2 * i + 1]))
            n_real = len(self.topo.sta_positions[i])

            if sta_idx >= n_real or self._aps[i].queues[sta_idx] <= 0:
                target_stas.append(-1)
                selected_mcs.append(0)
            else:
                target_stas.append(sta_idx)
                selected_mcs.append(mcs_idx)

        return target_stas, selected_mcs

    def _transmit(
        self,
        target_stas: list[int],
        selected_mcs: list[int],
        transmitting: list[int],
    ) -> tuple[list[float], int]:
        """전 AP의 전송을 처리하고 (AP별 성공 패킷 수, 중첩 차단 횟수)를 반환한다."""
        successful_tx = [0.0] * self.n_aps
        blocked_this_step = 0

        for i in transmitting:
            sta_idx = target_stas[i]
            mcs_idx = selected_mcs[i]

            result = self.interference.compute(
                ap_idx=i,
                sta_idx=sta_idx,
                transmitting_aps=transmitting,
                is_harsh=self.is_harsh_interference,
            )

            # 중첩 제약이 SINR 판정보다 우선한다.
            if result.blocked_by_overlap:
                self._aps[i].pkts_blocked += 1
                self._aps[i].pkts_dropped += 1
                blocked_this_step += 1
                continue

            if self.mcs_table.decodable(result.sinr_db, mcs_idx):
                capacity = self.mcs_table.capacity(mcs_idx)
                tx_pkts = float(min(self._aps[i].queues[sta_idx], capacity))
                self._aps[i].queues[sta_idx] -= tx_pkts   # 부하 비율과 무관하게 항상 차감
                self._aps[i].pkts_sent += int(tx_pkts)
                successful_tx[i] = tx_pkts
            else:
                self._aps[i].pkts_dropped += 1

        return successful_tx, blocked_this_step


# 기존 임포트 관례(from src.envs.xxx import OBSSEnv)와의 호환을 위한 별칭
OBSSEnv = QHOBSSEnv
