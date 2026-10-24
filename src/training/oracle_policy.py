"""
src/training/oracle_policy.py
학습 없이 환경 상태만으로 행동을 정하는 기준 정책.

용도: "throughput 0"의 원인이 환경/평가 경로인지 학습 부족인지를 가른다.
오라클이 0이 아닌 throughput을 내면 환경·집계·평가 경로는 정상이다.

주의: 아래 oracle 은 상한이 아니다. 모든 AP가 항상 전송한다는 전제에서 MCS 만 최적화하므로
쉬기(mute)로 간섭을 줄이는 스케줄(Co-TDMA 등)보다 낮을 수 있다. 처리량 상한은
파일 하단의 exact_oracle_actions 를 쓴다.

정책
  oracle        : 모든 AP가 동시에 전송한다고 가정(보수적)하고, 그 SINR에서 복호
                  가능한 최고 MCS를 고른다. 일부 AP만 전송해도 SINR은 더 좋아지므로
                  선택한 MCS는 항상 복호 가능하다.
  oracle_queued : 큐가 있는 AP만 전송한다고 가정한다. 다음 스텝 도착으로 가정보다
                  많은 AP가 전송하면 일부 MCS가 실패할 수 있다.
  fixed_mcs     : STA 선택은 오라클과 같고 MCS만 고정한다 (진단용).

STA 선택은 overlap이 아닌 STA 중 큐가 가장 긴 것이다. overlap STA는 hard-block
영향을 분리하기 위해 피한다 (비중첩 STA가 없는 AP만 예외).
"""

from __future__ import annotations

import itertools

import numpy as np


def choose_targets(env) -> list[int]:
    """AP별 대상 STA 인덱스. 비중첩 STA 중 큐가 가장 긴 것(동률이면 앞 인덱스)."""
    targets = []
    for i in range(env.n_aps):
        n_real = len(env.topo.sta_positions[i])
        candidates = [k for k in range(n_real) if not env.topo.is_overlap_sta(i, k)]
        if not candidates:
            candidates = list(range(n_real))
        queues = env._aps[i].queues
        targets.append(max(candidates, key=lambda k: (float(queues[k]), -k)))
    return targets


def prospective_harsh(env) -> bool:
    """env.step()가 이번 스텝에 적용할 harsh 여부 (step 내부와 동일한 식)."""
    return bool(env.harsh_mode_enabled and env._step >= env.harsh_start_step)


def oracle_actions(env, policy: str = "oracle", fixed_mcs: int = 0, compute_fn=None) -> np.ndarray:
    """
    env.step()에 그대로 넘길 평탄화 액션 [sta_0, mcs_0, sta_1, mcs_1, ...].

    compute_fn: SINR 계산 함수. 디버그 스크립트가 env.interference.compute를 계측용으로
                덮어쓴 경우, 덮어쓰기 전 원본을 넘겨야 계획 단계 호출이 계측 기록에 섞이지 않는다.
    """
    if policy not in ("oracle", "oracle_queued", "fixed_mcs"):
        raise ValueError(f"지원하지 않는 policy: {policy}")

    compute = compute_fn or env.interference.compute
    targets = choose_targets(env)
    harsh = prospective_harsh(env)

    if policy == "oracle_queued":
        transmitting = [i for i in range(env.n_aps) if env._aps[i].queues[targets[i]] > 0]
    else:
        transmitting = list(range(env.n_aps))

    # 계획 단계의 SINR 계산이 overlap 통계 카운터를 올리지 않도록 저장 후 복원한다.
    saved = (env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count)
    actions: list[int] = []
    try:
        for i in range(env.n_aps):
            if policy == "fixed_mcs":
                mcs = env.mcs_table.clamp(fixed_mcs)
            else:
                active = set(transmitting) | {i}
                result = compute(i, targets[i], active, is_harsh=harsh)
                mcs = env.mcs_table.best_mcs_for(result.sinr_db)
            actions.extend([targets[i], mcs])
    finally:
        env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count = saved

    return np.asarray(actions, dtype=np.int32)


# ── 정확 per-slot oracle 과 Co-TDMA ─────────────────────────────────────────
# 위의 oracle 은 "모든 AP 동시 전송"을 전제로 MCS 만 고르는 휴리스틱이며 상한이 아니다.
# 아래 exact_oracle_actions 는 모든 (STA | mute) 조합을 전수 탐색해 슬롯 처리량을 최대화하는
# 슬롯 단위 정확 최적해로, 다음 슬롯 도착(peek_arrivals)까지 아는 genie 이므로 같은 슬롯의
# 어떤 스케줄러보다도 처리량이 높거나 같다.

def mute_action(env, ap_idx: int) -> int:
    """AP ap_idx 가 이번 슬롯을 쉬도록 하는 STA 인덱스 (명시적 mute 또는 미존재 STA)."""
    if env.mute_index is not None:
        return env.mute_index
    n_real = len(env.topo.sta_positions[ap_idx])
    if n_real < env.k_stas:
        return n_real
    raise ValueError(f"AP{ap_idx} 는 mute 할 수 없습니다: env.allow_mute=true 로 설정하십시오")


def _flat(env, stas: list[int], mcs: list[int]) -> np.ndarray:
    actions: list[int] = []
    for i, (sta, m) in enumerate(zip(stas, mcs)):
        actions.extend((mute_action(env, i), 0) if sta < 0 else (sta, m))
    return np.asarray(actions, dtype=np.int32)


def schedule_value(env, stas: list[int], queues: np.ndarray, harsh: bool, compute=None):
    """전송 집합 stas(-1=쉼)의 슬롯 처리량과 링크별 최고 복호 가능 MCS."""
    compute = compute or env.interference.compute
    transmitting = [i for i, s in enumerate(stas) if s >= 0]
    total, mcs = 0.0, [0] * env.n_aps
    for i in transmitting:
        result = compute(i, stas[i], transmitting, is_harsh=harsh)
        if result.blocked_by_overlap:
            continue
        m = env.mcs_table.best_mcs_for(result.sinr_db)
        if env.mcs_table.decodable(result.sinr_db, m):
            mcs[i] = m
            total += min(float(queues[i, stas[i]]), env.mcs_table.capacity(m))
    return total, mcs


def exact_oracle_actions(env) -> np.ndarray:
    harsh = prospective_harsh(env)
    queues = np.clip(env.queues() + env.peek_arrivals(), 0, env.queue_max_size)
    options = [[-1] + [k for k in range(len(env.topo.sta_positions[i])) if queues[i, k] > 0]
               for i in range(env.n_aps)]
    saved = (env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count)
    best = (-1.0, [-1] * env.n_aps, [0] * env.n_aps)
    try:
        for stas in itertools.product(*options):
            value, mcs = schedule_value(env, list(stas), queues, harsh)
            if value > best[0]:
                best = (value, list(stas), mcs)
    finally:
        env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count = saved
    return _flat(env, best[1], best[2])


def co_tdma_actions(env, turn: int) -> tuple[np.ndarray, int]:
    """한 슬롯에 한 AP만 전송(라운드 로빈, 큐가 있는 AP만). MCS 는 간섭 없는 SINR 기준 최고값."""
    harsh = prospective_harsh(env)
    queues = env.queues()
    stas = [-1] * env.n_aps
    for offset in range(env.n_aps):
        i = (turn + offset) % env.n_aps
        n_real = len(env.topo.sta_positions[i])
        if queues[i, :n_real].sum() > 0:
            stas[i] = int(np.argmax(queues[i, :n_real]))
            turn = i + 1
            break
    saved = (env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count)
    try:
        _, mcs = schedule_value(env, stas, np.full_like(queues, np.inf), harsh)
    finally:
        env.overlap_policy.blocked_count, env.overlap_policy.soft_penalty_count = saved
    return _flat(env, stas, mcs), turn
