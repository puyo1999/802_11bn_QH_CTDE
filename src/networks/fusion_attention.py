"""
src/networks/fusion_attention.py
제안 방법: Self-Attention Fusion Network (Attention-DRLCA)

핵심 아이디어:
  - AP 자신 → Query
  - 각 이웃 AP (직접 + relay hidden) → Key / Value
  - Attention score α_j = softmax(QK^T / sqrt(d_k)) 로
    이웃 중요도를 동적 학습
  - 가변 이웃 수 지원 (zero-padding 불필요, padding_mask 사용)

입력:
  self_state       : (B, state_dim)
  neighbor_states  : (B, N_nbr, state_dim)   — 패딩 없음
  neighbor_mask    : (B, N_nbr) bool — True면 유효 이웃

출력:
  fused            : (B, out_dim)
  attn_weights     : (B, N_nbr+1) — 해석 가능성용
"""
# numpy 1.26.0 다운그레이드에 따른 후속 조치
# Python 3.9 이하 버전에서는 torch.Tensor | None 같은 | (Union) 문법을 지원하지 않지만,
# 파일 맨 위에 from __future__ import annotations 를 넣어주면
# Python 3.10+ 스타일의 최신 타입 힌트 문법을 미리 빌려와 사용할 수 있게 됩니다.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionFusionNet(nn.Module):
    def __init__(
        self,
        state_dim:   int = 6,
        hidden_dim:  int = 32,
        n_heads:     int = 2,
        out_dim:     int = 8,
        dropout:     float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads
        self.out_dim    = out_dim

        # 상태를 hidden_dim으로 임베딩
        self.embed = nn.Linear(state_dim, hidden_dim)

        # Multi-head QKV projection
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj      = nn.Linear(hidden_dim, hidden_dim)

        # 최종 압축: hidden_dim → out_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        self_state:      torch.Tensor,          # (B, state_dim)
        neighbor_states: torch.Tensor,          # (B, N, state_dim)
        neighbor_mask:   torch.Tensor | None = None,  # (B, N) bool True=유효
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused        : (B, out_dim)
            attn_weights : (B, n_heads, 1, N)  — 시각화용
        """
        B, N, _ = neighbor_states.shape

        # ── 임베딩 ─────────────────────────────────────────────
        self_emb = self.embed(self_state)                    # (B, H)
        nbr_emb  = self.embed(neighbor_states)               # (B, N, H)

        # Query: 자신, Key/Value: 이웃
        q = self.W_q(self_emb).unsqueeze(1)                  # (B, 1, H)
        k = self.W_k(nbr_emb)                                # (B, N, H)
        v = self.W_v(nbr_emb)                                # (B, N, H)

        # ── Multi-head reshape ─────────────────────────────────
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, S, H) → (B, n_heads, S, head_dim)
            B_, S, H = t.shape
            return t.view(B_, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)   # (B, h, 1, d)
        k = split_heads(k)   # (B, h, N, d)
        v = split_heads(v)   # (B, h, N, d)

        # ── Scaled dot-product attention ───────────────────────
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, h, 1, N)

        if neighbor_mask is not None:
            # mask: True = 유효 → False 위치에 -inf
            inv_mask = ~neighbor_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            scores = scores.masked_fill(inv_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)   # (B, h, 1, N)
        attn_weights = self.attn_drop(attn_weights)

        # ── Context vector ─────────────────────────────────────
        context = torch.matmul(attn_weights, v)    # (B, h, 1, d)
        context = context.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)
        context = context.squeeze(1)               # (B, H)
        context = self.proj(context)               # (B, H)

        # ── 자신 임베딩과 concat → 최종 압축 ────────────────────
        fused = self.output_mlp(
            torch.cat([self_emb, context], dim=-1)
        )                                          # (B, out_dim)

        return fused, attn_weights                 # attn: (B, h, 1, N)

    def get_attention_scores(
        self,
        self_state:      torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_mask:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        헤드 평균 attention score 반환 — 논문 해석 가능성 실험용
        Returns: (B, N)
        """
        with torch.no_grad():
            _, attn = self.forward(self_state, neighbor_states, neighbor_mask)
        # (B, h, 1, N) → 헤드 평균 → (B, N)
        return attn.mean(dim=1).squeeze(-2)
