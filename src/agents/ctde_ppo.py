"""Local and topology-aware joint actors with a centralized value critic."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.agents.observation_codec import ObservationCodec
from src.debug_c4 import log


class CTDEActorCritic(nn.Module):
    """Per-AP policies plus an attention actor that conditions on joint AP context."""

    def __init__(self, observation_dim: int, n_agents: int, n_stas: int, n_mcs: int,
                 hidden_dim: int, vq_bits: int, adjacency: torch.Tensor):
        super().__init__()
        self.observation_dim = observation_dim
        self.n_agents = n_agents
        self.n_stas = n_stas
        self.n_mcs = n_mcs
        if tuple(adjacency.shape) != (n_agents, n_agents):
            raise ValueError("adjacency must have shape [n_agents, n_agents]")
        adjacency = adjacency.to(dtype=torch.bool).clone()
        adjacency.fill_diagonal_(True)
        self.register_buffer("attention_mask", ~adjacency)

        self.codec = ObservationCodec(observation_dim, vq_bits)
        self.local_encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        n_heads = 4 if hidden_dim % 4 == 0 else 1
        self.joint_attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.local_sta_head = nn.Linear(hidden_dim, n_stas)
        self.local_mcs_head = nn.Linear(hidden_dim, n_mcs)
        self.joint_actor = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.joint_sta_head = nn.Linear(hidden_dim, n_stas)
        self.joint_mcs_head = nn.Linear(hidden_dim, n_mcs)
        self.critic = nn.Sequential(nn.Linear(n_agents * observation_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1))

    def encode_observations(self, observations: torch.Tensor):
        return self.codec.encode(observations)

    def actor_logits(self, local_observations: torch.Tensor,
                     valid_sta_counts: torch.Tensor | None = None,
                     centralized: bool | torch.Tensor = False):
        """Central mode attends over AP features; fallback uses only local features."""
        features = self.local_encoder(local_observations)
        context, _ = self.joint_attention(
            features, features, features, attn_mask=self.attention_mask,
            need_weights=False,
        )
        joint_features = self.joint_actor(torch.cat((features, context), dim=-1))
        local_sta = self.local_sta_head(features)
        local_mcs = self.local_mcs_head(features)
        joint_sta = self.joint_sta_head(joint_features)
        joint_mcs = self.joint_mcs_head(joint_features)

        if isinstance(centralized, torch.Tensor):
            mode_mask = centralized.to(device=features.device, dtype=torch.bool).view(-1, 1, 1)
            sta_logits = torch.where(mode_mask, joint_sta, local_sta)
            mcs_logits = torch.where(mode_mask, joint_mcs, local_mcs)
        elif centralized:
            sta_logits, mcs_logits = joint_sta, joint_mcs
        else:
            sta_logits, mcs_logits = local_sta, local_mcs

        if valid_sta_counts is not None:
            sta_ids = torch.arange(self.n_stas, device=sta_logits.device)
            mask = sta_ids.view(1, 1, -1) >= valid_sta_counts.view(-1, self.n_agents, 1)
            sta_logits = sta_logits.masked_fill(mask, -1.0e9)
        return sta_logits, mcs_logits

    def sample_actions(self, local_observations: torch.Tensor,
                       valid_sta_counts: torch.Tensor, deterministic: bool = False,
                       centralized: bool | torch.Tensor = False):
        sta_logits, mcs_logits = self.actor_logits(
            local_observations, valid_sta_counts, centralized=centralized)
        sta_dist, mcs_dist = Categorical(logits=sta_logits), Categorical(logits=mcs_logits)
        sta_actions = sta_logits.argmax(-1) if deterministic else sta_dist.sample()
        mcs_actions = mcs_logits.argmax(-1) if deterministic else mcs_dist.sample()
        actions = torch.stack((sta_actions, mcs_actions), dim=-1)
        log("ACTOR centralized=%s deterministic=%s sta_logits_shape=%s mcs_logits_shape=%s sta_actions=%s mcs_actions=%s valid_sta_counts=%s mcs_argmax=%s",
            centralized if isinstance(centralized, bool) else centralized.detach().cpu().tolist(),
            deterministic, tuple(sta_logits.shape), tuple(mcs_logits.shape),
            sta_actions.detach().cpu().tolist(), mcs_actions.detach().cpu().tolist(),
            valid_sta_counts.detach().cpu().tolist() if valid_sta_counts is not None else None,
            mcs_logits.argmax(-1).detach().cpu().tolist())
        log_probs = sta_dist.log_prob(sta_actions) + mcs_dist.log_prob(mcs_actions)
        entropies = sta_dist.entropy() + mcs_dist.entropy()
        return actions, log_probs, entropies

    def evaluate_actions(self, local_observations: torch.Tensor, actions: torch.Tensor,
                         valid_sta_counts: torch.Tensor,
                         centralized: bool | torch.Tensor = False):
        sta_logits, mcs_logits = self.actor_logits(
            local_observations, valid_sta_counts, centralized=centralized)
        sta_dist, mcs_dist = Categorical(logits=sta_logits), Categorical(logits=mcs_logits)
        log_probs = sta_dist.log_prob(actions[..., 0]) + mcs_dist.log_prob(actions[..., 1])
        entropy = sta_dist.entropy() + mcs_dist.entropy()
        return log_probs, entropy

    def value(self, joint_observations: torch.Tensor) -> torch.Tensor:
        """Central critic consumes the uncompressed joint observation [B, N, D]."""
        global_state = joint_observations.flatten(start_dim=-2)
        return self.critic(global_state).squeeze(-1)
