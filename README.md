# Attention-DRLCA: Attention-based Fusion for IEEE 802.11bn Channel Access Optimization

> **논문 확장 실험 코드**  
> Yan et al. (2025) *"Multi-Agent Reinforcement Learning-Based Channel Access Optimization for IEEE 802.11bn"* (TGCN)에서  
> FC Fusion Network를 **Self-Attention Fusion**으로 교체한 Attention-DRLCA 구현

---

## 프로젝트 구조

```
attention_drlca/
├── src/
│   ├── envs/
│   │   ├── obss_env.py          # IEEE 802.11bn OBSS Gymnasium 환경
│   │   └── topology.py          # 토폴로지 생성 (fixed 6종 + random)
│   ├── networks/
│   │   ├── fusion_fc.py         # 기존 FC Fusion (baseline)
│   │   ├── fusion_attention.py  # Self-Attention Fusion (제안)
│   │   └── ddqn.py              # LSTM + DDQN Decision Network
│   ├── agents/
│   │   ├── drlca_agent.py       # 기존 DRLCA agent (FC fusion)
│   │   └── attn_drlca_agent.py  # Attention-DRLCA agent (제안)
│   └── utils/
│       ├── replay_buffer.py     # Experience Memory
│       ├── metrics.py           # d95, THP, PER 계산
│       └── logger.py            # WandB / TensorBoard 로거
├── configs/
│   ├── base.yaml                # 공통 하이퍼파라미터
│   ├── drlca_fc.yaml            # Baseline 실험 설정
│   └── drlca_attn.yaml          # Attention 실험 설정
├── experiments/
│   ├── train.py                 # 학습 진입점
│   ├── evaluate.py              # 평가 / cross-topology test
│   └── ablation.py              # Ablation study
├── scripts/
│   ├── run_baseline.sh
│   └── run_attention.sh
├── tests/
│   ├── test_env.py
│   └── test_networks.py
└── docs/
    └── experiment_design.md     # 실험 설계 문서
```

---

## 빠른 시작

```bash
# 1. 환경 설치
pip install -r requirements.txt

# 2. Baseline (FC Fusion) 학습
python experiments/train.py --config configs/drlca_fc.yaml

# 3. Attention-DRLCA 학습
python experiments/train.py --config configs/drlca_attn.yaml

# 4. 평가 (cross-topology)
python experiments/evaluate.py --config configs/drlca_attn.yaml --checkpoint runs/attn/best.pt

# 5. Ablation
python experiments/ablation.py --config configs/drlca_attn.yaml
```

---

## 핵심 지표 (Wi-Fi 8 PAR)

| 지표 | 정의 | 목표 |
|------|------|------|
| `d95` | 95th-percentile latency | ↓ 25% vs CSMA/CA |
| `THP` | 정규화 처리량 | ≥ CSMA/CA |
| `PER` | MPDU 손실률 | ≤ CSMA/CA |

---

## 재현 환경

- Python 3.10+
- PyTorch 2.2+
- Gymnasium 0.29+
- CUDA 11.8+ (선택)
