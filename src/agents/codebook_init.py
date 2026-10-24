"""
src/agents/codebook_init.py
VQ 코드북 초기화용 관측 수집과 k-means 중심 계산.

기존 초기화는 env.reset()을 여러 번 호출해 관측을 모았는데, 리셋 직후 상태는 시드와
무관하게 같다(큐 0, normal 모드). 그래서 샘플 32개가 서로 다른 벡터 2개였고, 코드북이
그 2개로만 채워져 큐 상태와 harsh 전환(tau_ratio)이 양자화에서 사라졌다.

여기서는 무작위 정책으로 에피소드를 끝까지(harsh 구간 포함) 돌려 관측을 모으고,
k-means 중심 K개를 돌려준다. 중심을 정확히 K개로 넘기면 코덱이 앞쪽 K개를 쓰든
전체를 쓰든 서로 다른 코드가 보장된다.

주의: 이 함수는 AP 단위 관측 벡터(행=AP, 열=특징)를 모은다. 코덱이 AP별로 같은
코드북을 쓰는 구조(로그의 shape=(1, 4, 14))와 맞춘 것이다.
"""

from __future__ import annotations

import numpy as np


def collect_observations(env, n_episodes: int = 4, base_seed: int = 0,
                         extra_features: int = 0) -> np.ndarray:
    """
    무작위 정책으로 n_episodes개 에피소드를 끝까지 실행하며 AP별 관측을 모은다.
    반환: (샘플 수, obs_dim). extra_features>0이면 0으로 채운 열을 뒤에 붙인다
    (통신 지연 age 특징이 켜진 경우 기존 코드의 zero_age와 같은 처리).
    """
    env.action_space.seed(base_seed)   # 무작위 행동도 재현 가능하게
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        rows.append(obs)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(env.action_space.sample())
            rows.append(obs)
    data = np.concatenate(rows, axis=0).astype(np.float32)
    if extra_features > 0:
        pad = np.zeros((data.shape[0], extra_features), dtype=np.float32)
        data = np.concatenate([data, pad], axis=-1)
    return data


def kmeans_centers(data: np.ndarray, k: int, n_iter: int = 30, seed: int = 0) -> np.ndarray:
    """
    k-means++ 시드 + Lloyd 반복. 서로 다른 벡터가 k개보다 적으면 있는 만큼만 반환한다
    (그 경우 호출자가 경고해야 한다).
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(data.round(6), axis=0)
    if len(unique) <= k:
        return unique.astype(np.float32)

    centers = [unique[rng.integers(len(unique))]]
    for _ in range(1, k):
        d2 = np.min(((unique[:, None, :] - np.stack(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum()
        centers.append(unique[rng.choice(len(unique), p=probs)])
    centers = np.stack(centers)

    for _ in range(n_iter):
        d2 = ((data[:, None, :] - centers[None]) ** 2).sum(-1)
        assign = d2.argmin(1)
        new_centers = centers.copy()
        for j in range(k):
            members = data[assign == j]
            if len(members) > 0:
                new_centers[j] = members.mean(0)
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return centers.astype(np.float32)


def quantization_report(data: np.ndarray, centers: np.ndarray) -> dict:
    """data를 centers로 양자화했을 때의 코드 사용 수와 재구성 오차."""
    d2 = ((data[:, None, :] - centers[None]) ** 2).sum(-1)
    idx = d2.argmin(1)
    recon = centers[idx]
    return {
        "centers": int(len(centers)),
        "codes_used": int(len(np.unique(idx))),
        "mse": float(np.mean((data - recon) ** 2)),
        "max_abs_err": float(np.max(np.abs(data - recon))),
    }
