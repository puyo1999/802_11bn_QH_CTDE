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

# ① 제안 기법 (Quantized-Hybrid CTDE & Q-MAPPO) 가치 믹서 학습
python experiments/train_ctde.py --config configs/qh_ctde.yaml

# ② 대조 베이스라인 1 (Attention-DRLCA) 학습
python experiments/train_ctde.py --config configs/drlca_attn.yaml

# 4. 평가 (cross-topology)
python experiments/evaluate.py --config configs/drlca_attn.yaml --checkpoint runs/attn/best.pt

# 5. Ablation
python experiments/ablation.py --config configs/drlca_attn.yaml
```

### Configuration-first CTDE 실행 경로

고정 토폴로지와 환경 파라미터는 이제 `configs/topologies.yaml` 및 YAML 설정으로
분리되어 있습니다. `T3`, `T6`의 `shared_sta_groups`는 중첩 커버리지 STA를 명시하며,
Classical CTDE는 환경이 반환한 관측값에서 local/global 차원을 검증합니다.

```bash
# 양자 계층 없이 classical CTDE 계약/학습 루프 검증
python experiments/train_classical_ctde.py --config configs/classical_ctde.yaml --topology T3
```

`qml.qnode`는 이 classical 경로가 재현 가능하게 검증된 후 encoder 계층에만 추가합니다.
---

## 🚀 다중 사용자 스케줄링 실험 가이드 ($K=4$)

Wi-Fi 8(802.11bn) 표준 사양에 맞춘 AP당 4인용 다중 사용자 환경($K=4$)에서 제안 기법(QH-CTDE) 및 베이스라인 모델들을 공정하게 비교 학습하기 위한 설정 및 구동 가이드입니다.

### 1. 핵심 환경 및 네트워크 차원 정의
다중 사용자 스케줄링 활성화에 따라 환경의 상태 공간(Observation Space)과 행동 공간(Action Space) 규격이 다음과 같이 확장되었습니다.
* **State Dimension (`state_dim`):** `9` (기존 싱글 STA 전용 4차원에서 다중 사용자 징후 및 DSM 스위칭 지표가 결합되어 **9차원**으로 확장)
* **Action Shape (`action_shape`):** `[9, 16]` (Contention Window 제어 후보 9개 $\times$ MCS 레벨 인덱스 16개)

### 2. 실험 실행 명령어

제안 알고리즘 및 베이스라인을 동일한 $K=4$ 환경(`mode: "qh"`)에서 실행하려면 아래 명령어를 순차적으로 수행합니다.

```bash


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
