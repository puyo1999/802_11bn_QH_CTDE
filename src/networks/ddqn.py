"""
src/networks/ddqn.py
LSTM + Double DQN Decision Network (논문 Fig.4 오른쪽 블록)

구조: LSTM64 × 2 → FC64 → FC(n_actions)
      Dueling 구조 옵션 추가 (개선 실험용)
"""

import torch
import torch.nn as nn


class LSTMDecisionNet(nn.Module):
    """
    논문 Decision Net: 2-layer LSTM + 2-layer FC.
    입력: fused time-series  (B, T, fused_dim)
    출력: Q-values           (B, n_actions)
    """

    def __init__(
        self,
        fused_dim:  int = 8,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        n_actions:  int = 7,
        dueling:    bool = False,
    ):
        super().__init__()
        self.dueling = dueling

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.act  = nn.ReLU()

        if dueling:
            # Dueling: value stream + advantage stream
            self.value_head = nn.Linear(64, 1)
            self.adv_head   = nn.Linear(64, n_actions)
        else:
            self.q_head = nn.Linear(64, n_actions)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, fused_dim)
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Returns:
            q_values : (B, n_actions)
            hidden   : LSTM hidden state (for sequential inference)
        """
        lstm_out, hidden = self.lstm(x, hidden)   # (B, T, lstm_hidden)
        last = lstm_out[:, -1, :]                 # (B, lstm_hidden)
        h = self.act(self.fc1(last))              # (B, 64)

        if self.dueling:
            v = self.value_head(h)                # (B, 1)
            a = self.adv_head(h)                  # (B, n_actions)
            q = v + a - a.mean(dim=-1, keepdim=True)
        else:
            q = self.q_head(h)                    # (B, n_actions)

        return q, hidden


class DDQNNet(nn.Module):
    """
    Online + Target 네트워크를 하나의 모듈로 래핑.
    soft update 메서드 포함.
    """

    def __init__(self, fused_dim: int, n_actions: int, dueling: bool = False):
        super().__init__()
        self.online = LSTMDecisionNet(fused_dim, n_actions=n_actions,
                                      dueling=dueling)
        self.target = LSTMDecisionNet(fused_dim, n_actions=n_actions,
                                      dueling=dueling)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def soft_update(self, tau: float = 0.004) -> None:
        """논문 Table II: τ = 0.004"""
        for p_o, p_t in zip(self.online.parameters(),
                             self.target.parameters()):
            p_t.data.copy_(tau * p_o.data + (1.0 - tau) * p_t.data)

    def forward(self, x: torch.Tensor,
                hidden: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        return self.online(x, hidden)

    def target_forward(self, x: torch.Tensor,
                       hidden: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        with torch.no_grad():
            return self.target(x, hidden)
