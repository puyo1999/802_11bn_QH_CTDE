"""
src/agents/transition_memory.py
QH-CTDE 에이전트의 (state, action, reward, next_state, done) 트랜지션 버퍼.

기존 QHCTDEAgent에 내장돼 있던 deque 로직을 분리하여 독립적으로
용량/샘플링 정책을 바꿀 수 있게 한다.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any


class TransitionMemory:
    """단순 균등 샘플링 리플레이 버퍼."""

    def __init__(self, capacity: int):
        self.buffer: deque[tuple[Any, Any, float, Any, bool]] = deque(maxlen=capacity)

    def push(self, state, action, reward: float, next_state, done: bool) -> None:
        self.buffer.append((state, action, float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> list[tuple[Any, Any, float, Any, bool]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
