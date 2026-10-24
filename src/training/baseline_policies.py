"""
src/training/baseline_policies.py
학습 없이 동작하는 비교 기준 정책 (QH 환경 qh_v2 전용).

모든 정책은 env.step()에 그대로 넘길 평탄화 액션 [sta_0, mcs_0, sta_1, mcs_1, ...]을 돌려준다.
측정값을 사후 보정하지 않는다: 처리량은 env.step()이 돌려준 info 값만 쓴다.

정책
  random      : AP마다 실제 STA 중 하나와 MCS를 균등 무작위로 고른다.
  greedy      : AP마다 큐가 가장 긴 STA를 고르고, MCS는 ARF로 정한다.
  round_robin : AP마다 실제 STA를 스텝 순서대로 돌아가며 고르고, MCS는 ARF로 정한다.
  oracle      : 모든 AP가 동시에 전송한다고 가정한 즉시 SINR에서 복호 가능한 최고 MCS를 고르고,
                overlap STA를 피한다 (src/training/oracle_policy.py). 쉬기를 하지 않으므로 상한이
                아니다 ("all-transmit SINR heuristic").
  exact_oracle: 모든 (STA | mute) 조합을 전수 탐색한 슬롯 단위 정확 최적해 (다음 도착까지 아는 genie).
                같은 슬롯에서 어떤 스케줄러보다 처리량이 높거나 같은 상한이다.
  co_tdma     : 한 슬롯에 한 AP만 라운드 로빈으로 전송하고 나머지는 쉰다. MCS 는 간섭 없는 SINR 기준.
  exact_oracle / co_tdma 는 mute 가 필요하다 (env.allow_mute 또는 미존재 STA 인덱스).

ARF (Auto Rate Fallback, 802.11의 고전적 링크 적응)
  AP별 MCS 상태를 두고, 전송 시도(대상 STA 큐 > 0)마다 결과를 본다.
  - 시작 MCS: 다른 AP 간섭이 없다고 가정한 SINR에서의 최고 MCS (에피소드 첫 스텝에 한 번).
  - 연속 실패 ARF_DOWN회 → MCS 1단계 하향. 상향 직후 첫 시도가 실패하면 즉시 하향.
  - 연속 성공 ARF_UP회 → MCS 1단계 상향.
  실패 원인(간섭, overlap 차단)을 구분하지 않는다. 실제 ARF와 같다.
"""

from __future__ import annotations

import numpy as np

from src.training.oracle_policy import co_tdma_actions, exact_oracle_actions, oracle_actions, prospective_harsh

POLICIES = ("random", "greedy", "round_robin", "oracle", "exact_oracle", "co_tdma")
ARF_UP = 10
ARF_DOWN = 2


def _n_real(env, ap: int) -> int:
    return len(env.topo.sta_positions[ap])


def _interference_free_mcs(env, ap: int, sta: int, harsh: bool) -> int:
    """다른 AP가 전송하지 않는다고 가정한 SINR에서 복호 가능한 최고 MCS."""
    saved = (env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count)
    try:
        result = env.interference.compute(ap, sta, {ap}, is_harsh=harsh)
    finally:
        # 계획 단계 계산이 overlap 통계 카운터를 올리지 않도록 복원한다.
        env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count = saved
    return env.mcs_table.best_mcs_for(result.sinr_db)


class BaselinePolicy:
    def __init__(self, name: str, seed: int = 0):
        if name not in POLICIES:
            raise ValueError(f"unknown baseline policy: {name}")
        self.name = name
        # 무작위 정책은 환경 RNG와 분리된 자체 RNG를 쓴다 (트래픽 도착 과정을 바꾸지 않기 위해).
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.turn = 0            # co_tdma: 다음 차례 AP
        self.mcs = None          # ARF 상태: AP별 현재 MCS
        self.n_ok = None         # 연속 성공 횟수
        self.n_fail = None       # 연속 실패 횟수
        self.probing = None      # 직전에 상향했는지
        self.attempted = None    # 이번 스텝에 전송을 시도했는지 (대상 큐 > 0)

    def _uses_arf(self) -> bool:
        return self.name in ("greedy", "round_robin")

    def act(self, env) -> np.ndarray:
        if self.name == "oracle":
            self.t += 1
            return oracle_actions(env, policy="oracle")
        if self.name == "exact_oracle":
            self.t += 1
            return exact_oracle_actions(env)
        if self.name == "co_tdma":
            self.t += 1
            actions, self.turn = co_tdma_actions(env, self.turn)
            return actions

        n = env.n_aps
        harsh = prospective_harsh(env)
        if self._uses_arf() and self.mcs is None:
            self._max_mcs = env.mcs_table.n_mcs - 1
            self.mcs = [0] * n
            self.n_ok = [0] * n
            self.n_fail = [0] * n
            self.probing = [False] * n
        self.attempted = [False] * n

        actions: list[int] = []
        for i in range(n):
            n_real = _n_real(env, i)
            queues = env._aps[i].queues
            if self.name == "random":
                sta = int(self.rng.integers(n_real))
                mcs = int(self.rng.integers(env.mcs_table.n_mcs))
            else:
                if self.name == "greedy":
                    sta = max(range(n_real), key=lambda k: (float(queues[k]), -k))
                else:  # round_robin
                    sta = self.t % n_real
                if self.t == 0:
                    self.mcs[i] = _interference_free_mcs(env, i, sta, harsh)
                mcs = self.mcs[i]
            self.attempted[i] = bool(queues[sta] > 0)
            actions.extend([sta, mcs])
        self.t += 1
        return np.asarray(actions, dtype=np.int32)

    def observe(self, info: dict) -> None:
        """env.step() 결과로 ARF 상태를 갱신한다. ARF를 쓰지 않는 정책은 무시한다."""
        if not self._uses_arf() or self.attempted is None:
            return
        thp = info.get("thp", [])
        for i, tried in enumerate(self.attempted):
            if not tried:
                continue
            ok = i < len(thp) and float(thp[i]) > 0
            if ok:
                self.n_ok[i] += 1
                self.n_fail[i] = 0
                self.probing[i] = False
                if self.n_ok[i] >= ARF_UP:
                    self.mcs[i] = min(self.mcs[i] + 1, self._max_mcs)
                    self.n_ok[i] = 0
                    self.probing[i] = True
            else:
                self.n_fail[i] += 1
                self.n_ok[i] = 0
                if self.probing[i] or self.n_fail[i] >= ARF_DOWN:
                    self.mcs[i] = max(self.mcs[i] - 1, 0)
                    self.n_fail[i] = 0
                self.probing[i] = False
