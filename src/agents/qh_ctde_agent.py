"""
src/agents/qh_ctde_agent.py
QH-CTDE 에이전트 — 하위 모듈(ActorFeatureExtractor, SchedulingHeads,
VectorQuantizer, DSMGate, TransitionMemory)을 조립한다.

state_dim은 항상 생성자 인자로만 받는다 (src.envs.space_validator.SpaceSpec에서
산출된 값을 agent_factory가 전달). 코드 내 하드코딩 금지.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.actor_network import ActorFeatureExtractor, SchedulingHeads
from src.agents.dsm_gate import DSMGate, DSMMode
from src.agents.transition_memory import TransitionMemory
from src.agents.vector_quantizer import VectorQuantizer


class QHCTDEAgent:
    """단일 AP를 제어하는 QH-CTDE 에이전트 (VQ 코디네이션 + CA-DSM 게이팅)."""

    def __init__(
        self,
        ap_id: int,
        state_dim: int,
        n_stas: int,
        n_mcs: int = 16,
        hidden_dim: int = 128,
        use_quantization: bool = True,
        vq_bits: int = 4,
        use_dsm: bool = True,
        gamma_margin: float = 0.8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        batch_size: int = 32,
        mem_size: int = 5000,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.ap_id = ap_id
        self.state_dim = state_dim
        self.n_stas = n_stas
        self.n_mcs = n_mcs
        self.gamma = gamma
        self.tau = tau
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.use_quantization = use_quantization
        self.use_dsm = use_dsm
        self.dsm = DSMGate(use_dsm=use_dsm, gamma_margin=gamma_margin)
        self.last_gate_used: bool | None = None

        self.feature_net = ActorFeatureExtractor(state_dim, hidden_dim).to(self.device)
        self.heads = SchedulingHeads(hidden_dim, n_stas, n_mcs).to(self.device)

        param_groups = [
            {"params": self.feature_net.parameters()},
            {"params": self.heads.parameters()},
        ]
        if self.use_quantization:
            self.vq = VectorQuantizer(codebook_size=2 ** vq_bits, feature_dim=hidden_dim).to(self.device)
            param_groups.append({"params": self.vq.parameters()})
        else:
            self.vq = None

        self.optimizer = optim.Adam(param_groups, lr=lr)
        self.memory = TransitionMemory(mem_size)

    # ── 특징 경로 (양자화 게이팅 포함) ─────────────────────────
    def _gated_feature(
        self, raw_feat: torch.Tensor, state_or_states, dsm_mode: DSMMode, batched: bool
    ) -> torch.Tensor:
        if self.vq is None:
            return raw_feat

        quantized = self.vq.quantize(raw_feat, straight_through=batched)
        if batched:
            gate = self.dsm.decide_batch(state_or_states, dsm_mode).float().unsqueeze(-1)
            return gate * quantized + (1.0 - gate) * raw_feat

        use_quant = self.dsm.decide(state_or_states, dsm_mode)
        self.last_gate_used = use_quant
        return quantized if use_quant else raw_feat

    # ── 행동 선택 ────────────────────────────────────────────
    def select_action(
        self, state: np.ndarray, valid_sta_cnt: int, dsm_mode: DSMMode = "auto"
    ) -> np.ndarray:
        if random.random() < self.eps:
            self.last_gate_used = None
            return np.array(
                [random.randrange(valid_sta_cnt), random.randrange(self.n_mcs)], dtype=np.int32
            )

        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_feat = self.feature_net(s_t)
            feat = self._gated_feature(raw_feat, state, dsm_mode, batched=False)

            sta_logits, mcs_logits = self.heads(feat)
            if valid_sta_cnt < self.n_stas:
                sta_logits[0, valid_sta_cnt:] = -1e9

            sta_act = int(sta_logits.argmax(dim=-1).item())
            mcs_act = int(mcs_logits.argmax(dim=-1).item())

        return np.array([sta_act, mcs_act], dtype=np.int32)

    def store_transition(self, state, action, reward: float, next_state, done: bool) -> None:
        self.memory.push(state, action, reward, next_state, done)

    # ── 학습 ────────────────────────────────────────────────
    def update(self, dsm_mode: DSMMode = "auto") -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = np.array([b[1] for b in batch])
        sta_a = torch.LongTensor(actions[:, 0]).to(self.device)
        mcs_a = torch.LongTensor(actions[:, 1]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)

        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        raw_feat = self.feature_net(states)
        feat = self._gated_feature(raw_feat, states, dsm_mode, batched=True)
        sta_logits, mcs_logits = self.heads(feat)

        sta_dist = torch.distributions.Categorical(logits=sta_logits)
        mcs_dist = torch.distributions.Categorical(logits=mcs_logits)
        log_probs = sta_dist.log_prob(sta_a) + mcs_dist.log_prob(mcs_a)

        loss = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.feature_net.parameters()) + list(self.heads.parameters()),
            max_norm=self.grad_clip,
        )
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def set_eval(self) -> None:
        self.feature_net.eval()
        self.heads.eval()

    def set_train(self) -> None:
        self.feature_net.train()
        self.heads.train()

    # ── 저장/복원 ────────────────────────────────────────────
    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "feature_net": self.feature_net.state_dict(),
            "heads": self.heads.state_dict(),
            "eps": self.eps,
        }
        if self.vq is not None:
            payload["vq"] = self.vq.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(ckpt["feature_net"])
        self.heads.load_state_dict(ckpt["heads"])
        if self.vq is not None and "vq" in ckpt:
            self.vq.load_state_dict(ckpt["vq"])
        self.eps = ckpt.get("eps", self.eps_end)
