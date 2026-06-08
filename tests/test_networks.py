"""tests/test_networks.py — Fusion / DDQN 네트워크 shape 검증"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from src.networks.fusion_fc import FCFusionNet
from src.networks.fusion_attention import AttentionFusionNet
from src.networks.ddqn import DDQNNet


def test_fc_fusion_shape():
    net = FCFusionNet(state_dim=6, max_neighbors=3, out_dim=8)
    s   = torch.randn(4, 6)           # batch=4
    n   = torch.randn(4, 3, 6)
    out = net(s, n)
    assert out.shape == (4, 8)


def test_attn_fusion_shape():
    net = AttentionFusionNet(state_dim=6, hidden_dim=32, n_heads=2, out_dim=8)
    s   = torch.randn(4, 6)
    n   = torch.randn(4, 3, 6)
    mask = torch.ones(4, 3, dtype=torch.bool)
    fused, attn = net(s, n, mask)
    assert fused.shape == (4, 8)
    assert attn.shape  == (4, 2, 1, 3)   # (B, heads, 1, N)


def test_attn_fusion_variable_neighbors():
    """가변 이웃 수 처리 — padding 없이 동작해야 함"""
    net = AttentionFusionNet(state_dim=6, hidden_dim=32, n_heads=2, out_dim=8)
    # 이웃이 1명인 경우
    s   = torch.randn(2, 6)
    n   = torch.randn(2, 1, 6)
    fused, _ = net(s, n)
    assert fused.shape == (2, 8)


def test_attn_fusion_padding_mask():
    """mask=False 이웃의 attention score가 0에 가까워야 함"""
    net  = AttentionFusionNet(state_dim=6, hidden_dim=32, n_heads=2, out_dim=8)
    s    = torch.randn(1, 6)
    n    = torch.randn(1, 3, 6)
    mask = torch.tensor([[True, True, False]])  # 3번째 이웃 무효
    _, attn = net(s, n, mask)
    # (1, heads, 1, 3) — 마지막 이웃 score ≈ 0
    score_invalid = attn[0, :, 0, 2].mean().item()
    assert abs(score_invalid) < 1e-5, f"Invalid neighbor attn should be ~0, got {score_invalid}"


def test_ddqn_shape():
    net = DDQNNet(fused_dim=8, n_actions=7)
    x   = torch.randn(4, 10, 8)   # (B, T, F)
    q, hidden = net(x)
    assert q.shape == (4, 7)


def test_ddqn_soft_update():
    net = DDQNNet(fused_dim=8, n_actions=7)
    p0  = list(net.target.parameters())[0].data.clone()
    # online을 랜덤으로 바꾼 뒤 soft update
    for p in net.online.parameters():
        p.data.fill_(1.0)
    net.soft_update(tau=0.5)
    p1 = list(net.target.parameters())[0].data
    # target이 업데이트되었는지 확인
    assert not torch.allclose(p0, p1)
