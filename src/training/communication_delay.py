"""Reproducible bounded delay for AP-to-server observation messages.

Delay is sampled in integer decision cycles. A shared two-state congestion
process adds temporal correlation, while per-AP integer jitter models link
variation. This is an abstract sensitivity model, not a calibrated PHY/MAC
backhaul simulator.
"""
from __future__ import annotations

import numpy as np

from src.config.schema import CommunicationDelayConfig


class ObservationDelayChannel:
    """Per-AP packet queues that expose the newest observation delivered so far."""

    def __init__(self, config: CommunicationDelayConfig, n_agents: int):
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.config = config
        self.n_agents = n_agents
        self._rng = np.random.default_rng()
        self._queues: list[list[tuple[int, int, np.ndarray]]] = []
        self._latest: np.ndarray | None = None
        self._latest_source_steps: np.ndarray | None = None
        self._congested = False
        self._last_delay_steps = np.zeros(n_agents, dtype=np.int32)

    def reset(self, initial_observation: np.ndarray, seed: int) -> None:
        initial = np.asarray(initial_observation, dtype=np.float32)
        if initial.ndim != 2 or initial.shape[0] != self.n_agents:
            raise ValueError("initial_observation must have shape [n_agents, observation_dim]")
        self._rng = np.random.default_rng(seed)
        self._queues = [[] for _ in range(self.n_agents)]
        # The reset observation is a synchronized initial condition at step zero.
        self._latest = initial.copy()
        self._latest_source_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._congested = bool(self._rng.random() < self.config.initial_congested_prob)
        self._last_delay_steps.fill(0)

    def _sample_delays(self) -> np.ndarray:
        if self._congested:
            if self._rng.random() < self.config.congested_to_normal_prob:
                self._congested = False
        elif self._rng.random() < self.config.normal_to_congested_prob:
            self._congested = True
        base = (self.config.congested_delay_steps if self._congested
                else self.config.normal_delay_steps)
        jitter = self._rng.integers(0, self.config.jitter_max_steps + 1,
                                    size=self.n_agents)
        return np.minimum(base + jitter, self.config.max_delay_steps).astype(np.int32)

    def transmit(self, observations: np.ndarray, source_step: int) -> np.ndarray:
        """Queue one timestamped local observation from each AP; return delays."""
        if self._latest is None:
            raise RuntimeError("reset the delay channel before transmit")
        packet = np.asarray(observations, dtype=np.float32)
        if packet.shape != self._latest.shape:
            raise ValueError(f"observation shape changed: expected {self._latest.shape}, got {packet.shape}")
        delays = self._sample_delays()
        for ap in range(self.n_agents):
            arrival_step = int(source_step + delays[ap])
            queue = self._queues[ap]
            queue.append((arrival_step, int(source_step), packet[ap].copy()))
            queue.sort(key=lambda item: (item[0], item[1]))
        self._last_delay_steps = delays
        return delays.copy()

    def receive(self, current_step: int) -> tuple[np.ndarray, np.ndarray]:
        """Deliver due packets and return latest per-AP observations and their ages."""
        if self._latest is None or self._latest_source_steps is None:
            raise RuntimeError("reset the delay channel before receive")
        for ap, queue in enumerate(self._queues):
            while queue and queue[0][0] <= current_step:
                _, source_step, packet = queue.pop(0)
                # Do not let a late older packet overwrite a newer delivered sample.
                if source_step >= self._latest_source_steps[ap]:
                    self._latest[ap] = packet
                    self._latest_source_steps[ap] = source_step
        ages = np.maximum(0, current_step - self._latest_source_steps).astype(np.float32)
        return self._latest.copy(), ages

    def latest_delays(self) -> np.ndarray:
        """Integer-cycle delays sampled for the most recent transmitted batch."""
        return self._last_delay_steps.copy()
