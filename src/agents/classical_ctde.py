"""Small, stable classical CTDE baseline used before a quantum encoder is added."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


class LocalPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_stas: int, n_mcs: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.sta = nn.Linear(hidden_dim, n_stas)
        self.mcs = nn.Linear(hidden_dim, n_mcs)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(state)
        return self.sta(features), self.mcs(features)


class CentralCritic(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


@dataclass
class UpdateResult:
    actor_loss: float
    critic_loss: float


class ClassicalCTDE:
    """Decentralized local actors plus a centralized value critic (A2C style)."""
    def __init__(self, local_dim: int, n_aps: int, n_stas: int, *, hidden_dim: int = 128,
                 n_mcs: int = 16, lr: float = 3e-4, gamma: float = .99, device: str = "cpu"):
        self.n_aps, self.n_stas, self.n_mcs, self.gamma = n_aps, n_stas, n_mcs, gamma
        self.device = torch.device(device)
        self.actors = nn.ModuleList([LocalPolicy(local_dim, hidden_dim, n_stas, n_mcs) for _ in range(n_aps)]).to(self.device)
        self.critic = CentralCritic(local_dim * n_aps, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actors.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_actions(self, obs: np.ndarray, valid_sta_counts: list[int], *, deterministic: bool = False):
        actions, log_probs = [], []
        for index, actor in enumerate(self.actors):
            state = torch.as_tensor(obs[index], dtype=torch.float32, device=self.device).unsqueeze(0)
            sta_logits, mcs_logits = actor(state)
            sta_logits[:, valid_sta_counts[index]:] = -torch.inf
            sta_dist, mcs_dist = torch.distributions.Categorical(logits=sta_logits), torch.distributions.Categorical(logits=mcs_logits)
            sta = sta_logits.argmax(-1) if deterministic else sta_dist.sample()
            mcs = mcs_logits.argmax(-1) if deterministic else mcs_dist.sample()
            actions.extend((int(sta.item()), int(mcs.item())))
            log_probs.append(sta_dist.log_prob(sta) + mcs_dist.log_prob(mcs))
        return np.asarray(actions, dtype=np.int32), torch.stack(log_probs).sum()

    def update(self, obs: np.ndarray, log_prob: torch.Tensor, reward: float, next_obs: np.ndarray, done: bool) -> UpdateResult:
        state = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_obs.reshape(1, -1), dtype=torch.float32, device=self.device)
        value = self.critic(state)
        with torch.no_grad():
            target = torch.tensor([reward], dtype=torch.float32, device=self.device) + self.gamma * (1.0 - float(done)) * self.critic(next_state)
        advantage = target - value
        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.square().mean()
        self.optimizer.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        nn.utils.clip_grad_norm_(list(self.actors.parameters()) + list(self.critic.parameters()), 1.0)
        self.optimizer.step()
        return UpdateResult(float(actor_loss.detach()), float(critic_loss.detach()))
