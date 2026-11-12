# 선행 연구 조사: 통신 효율적 MARL, 양자화, 분산 연합학습, 동적 설정

- 작성일: 2026-11-12
- 목적: `docs/paper3_dynamic_quantization.md`의 조합안이 이미 연구됐는지 확인
- 검색 키워드: communication-efficient, quantized, decentralized federated learning, dynamic configuration,
  MARL 통신 게이팅, 가치 기반 통신(VoI), 링크 적응 + MARL, oracle 모방
- 확인 방법: 웹 검색으로 후보를 찾고, **arXiv API 기록 또는 Crossref DOI 기록으로 제목·저자·날짜·게재처를 확인**했다.
  본문까지 읽은 논문은 없다. 성능 수치는 초록과 2차 요약에서 가져온 것이므로 인용 전에 본문으로 재확인한다.
  `references.bib`에는 아직 넣지 않았다.

## 1. 결론 요약

| 아이디어 | 선행 연구 | 판단 |
|---|---|---|
| MARL 메시지를 VQ로 압축하고, 게이트로 보낼지 정하며, 대역폭 예산 벌점을 둠 | Farooq & Iqbal (ICRA 2026) 이 거의 그대로 함 | **단독으로는 새롭지 않음** |
| 대역폭 제약 하의 통신 여부 결정(이벤트 트리거, 라그랑주) | ETCNet (2020), SchedNet (ICLR 2019), IC3Net 계열 | 새롭지 않음 |
| 메시지의 정보 가치(VoI)로 통신 결정 | VIL2C (2025), KL 기반 게이팅 (2026), AAMAS 2026 비용-편익 | 개념은 있음 |
| 양자화 수준을 시간·클라이언트별로 바꿈 (연합학습) | DAdaQuant (2021), 이중 적응 양자화 (2024), LM-DFL (2023) 등 다수 | 연합학습에서는 성숙. 단, **학습 중 모델 갱신**의 양자화이고 실행 중 협조 메시지가 아님 |
| oracle 해를 모방해 분산 정책 학습 | Chang et al. (ICML 2026 워크숍, ILP oracle 모방, 엣지 스케줄링) | 행동 모방은 있음. **oracle로 통신 가치를 계산해 비트 배분을 가르치는 것**은 찾지 못함 |
| 셀 간 협조 메시지를 **링크 적응(MCS/CQI 선택)**에 쓰는 MARL | 찾지 못함. 단일 셀 DRL 링크 적응(You et al. 2026)과 비학습 협조 링크 적응(특허)만 있음 | **열려 있는 것으로 보임** |

→ "동적 VQ 비트 배분" 자체는 이미 있다. 이 연구의 차별점은 **정확 oracle**과 **링크 적응 중심의 메시지**에서 나와야 한다.

## 2. 분야별 정리

### 2.1 통신 효율적 MARL (실행 중 메시지)

| 논문 | 핵심 | 확인 |
|---|---|---|
| A. Farooq, K. Iqbal, "Bandwidth-Efficient Multi-Agent Communication through Information Bottleneck and Vector Quantization" | 정보 병목 + VQ로 메시지를 이산화, 게이트로 통신 시점 결정, 예산 벌점. 무통신 대비 성능 향상과 대역폭 절감 보고 | arXiv 2602.02035 (2026-02), ICRA 2026 채택(arXiv 주석) |
| G. Hu et al., "Event-Triggered Multi-agent Reinforcement Learning with Communication under Limited-bandwidth Constraint" (ETCNet) | 전송 여부를 제약 MDP로, 라그랑주 승수로 대역폭 제약 | arXiv 2010.04978 (2020) |
| D. Kim et al., "Learning to Schedule Communication in Multi-agent Reinforcement Learning" (SchedNet) | 공유 매체에서 누가 보낼지 학습, 메시지 인코딩 함께 학습 | arXiv 1902.01554, ICLR 2019 |
| Q. Zhang et al., "VIL2C: Value-of-Information Aware Low-Latency Communication for Multi-Agent Reinforcement Learning" | 지연된 메시지의 정보 가치를 중요도와 지연으로 정의 | arXiv 2511.19146 (2025-11) |
| T. Kaman, "When to Communicate: Belief Distributions and KL Divergence for Principled Gating in Multi-Agent RL" | 믿음 분포의 KL 발산으로 통신 게이트 | arXiv 2608.14559 (2026-05) |
| Lin, Zhang, Zeng, "Why and Whom to Communicate? A Dual-Objective, Cost-Benefit Framework for Multi-Agent Communication" | 성능-비용 파레토 전선을 학습 | AAMAS 2026, DOI 10.65109/tayo7266 (Crossref) |
| G. Yang et al., "Multi-Agent Reinforcement Learning with Communication-Constrained Priors" | 손실 메시지와 무손실 메시지의 영향을 분리, 통신 가치를 전역 보상에 반영 | arXiv 2512.03528, NeurIPS 2025 |

### 2.2 양자화와 통신 효율 연합학습 (학습 중 모델 갱신)

| 논문 | 핵심 | 확인 |
|---|---|---|
| R. Hönig, Y. Zhao, R. Mullins, "DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning" | 양자화 수준을 시간(라운드)과 클라이언트 두 축으로 조정 | arXiv 2111.00465 (2021) |
| L. Chen, W. Liu, Y. Chen, W. Wang, "Communication-Efficient Design for Quantized Decentralized Federated Learning" | **분산 연합학습**에서 Lloyd-Max로 양자화 왜곡 최소화(LM-DFL), 이중 적응 변형 | arXiv 2303.08423 (2023). 저널 게재 정보는 미확인 |
| X. Han et al., "Energy-Efficient Wireless Federated Learning via Doubly Adaptive Quantization" | 무선 FL에서 양자화 수준과 자원 할당·스케줄링 결합 | arXiv 2402.12957 (2024) |
| A. Mahmoudi, M. Xiao, E. Björnson, "Accelerating Energy-Efficient Federated Learning in Cell-Free Networks with Adaptive Quantization" | 셀프리 망에서 그래디언트 비트 할당을 동적으로 조정 | arXiv 2412.20785 (2024-12) |
| H. Li et al., "FedFQ: Federated Learning with Fine-Grained Quantization" | 파라미터 단위로 비트 폭 할당 | arXiv 2408.08977 (2024) |
| E. Ardıç, Y. Genç, "Enhanced Privacy and Communication Efficiency in Non-IID Federated Learning with Adaptive Quantization and Differential Privacy" | 라운드별 비트 길이 스케줄 + 클라이언트별 스케줄러 | IEEE Access 13, 54322–54337 (2025), DOI 10.1109/ACCESS.2025.3554138 |
| A. Tariq et al., "Enhancing Communication Efficiency in FL with Adaptive Gradient Quantization and Communication Frequency Optimization" | 그래디언트 혁신도와 오차 민감도로 양자화 수준 조정 + 통신 빈도 최적화 | arXiv 2509.23419 (2025-09) |

### 2.3 무선 스케줄링·링크 적응과 학습

| 논문 | 핵심 | 확인 |
|---|---|---|
| J. Chang, A. Zhang, J. Chen, A. Akella, "From Offline Global Information to Online Decentralized Policies in Edge Network Scheduling" | ILP oracle 스케줄을 지도 모방, 그리고 CTDE MARL. 지역 정책이 휴리스틱보다 최대 37% 우수 | ICML 2026 워크숍 포스터 (학회 페이지) |
| R. Dahal, M. Vaezi, "Selective Experience Sharing in Reinforcement Learning Enhances Interference Management" | 다중 셀 간섭 관리 MARL에서 경험 공유를 선택적으로 줄여 시그널링 감소 | IEEE Commun. Lett. 2025, DOI 10.1109/LCOMM.2025.3535898 |
| L. You et al., "From Simulation to Reality: Practical Deep Reinforcement Learning-based Link Adaptation for Cellular Networks" | 주기적·낡은 CQI 하의 DRL 링크 적응, 단일 셀 | arXiv 2603.00689 (2026-02) |

## 3. 조합안별 새로움 판단

| 조합안 (`paper3_dynamic_quantization.md` 기준) | 가장 가까운 연구 | 차이 | 판단 |
|---|---|---|---|
| 동적 VQ 비트 + 게이트 + 예산 벌점 | Farooq & Iqbal 2026 | 거의 없음 | 단독 기여 불가. 비교 대상(baseline)으로 써야 함 |
| ① oracle로 계산한 정보 가치로 비트 배분을 지도학습 | Chang et al. 2026 (oracle 행동 모방), VIL2C (VoI 정의) | oracle을 **행동**이 아니라 **통신 가치의 정답**으로 씀. 정확 oracle이 있어야 가능 | 새로움 있음 |
| ② 유체 모델(협조 이득 법칙)로 비트 수를 정함 | 없음 (paper1 결과에 기반) | 이론 모델이 통신 정책을 결정 | 새로움 있음, 다만 paper1 결과 의존 |
| ③ 메시지를 링크 적응(CQI 선택)에 씀 + OLLA와 하나의 외부 루프 | 찾지 못함 | 손실의 가장 큰 몫(Δ_LA)을 통신으로 줄임 | **가장 새로움** |
| ⑥ (신규) 학습 중 통신(분산 연합학습 양자화)과 실행 중 통신(협조 메시지 양자화)을 **하나의 백홀 예산**으로 함께 배분 | FL 양자화 연구들은 학습 통신만, MARL 통신 연구들은 실행 통신만 다룸 | 두 통신을 같은 예산 안에서 맞바꿈 | 새로움 있음. 다만 O-RAN에서 RIC가 중앙 학습을 하므로 분산 학습의 동기를 따로 세워야 함 |

## 4. 시사점

1. **Farooq & Iqbal (2026)은 필수 비교 대상이다.** 게이트 + VQ + 예산 벌점을 그대로 구현해 baseline으로 둔다.
2. **차별점은 oracle과 링크 적응에서 나온다.** 추천 조합(③ + ①)은 이번 검색 범위에서 직접적인 선행 연구가 없었다.
3. **연합학습의 "이중 적응" 개념(시간 축 + 개체 축)**은 셀 × 슬롯 단위 비트 배분의 설명 틀로 빌려 쓸 수 있다.
   다만 대상(학습 중 모델 갱신 대 실행 중 메시지)이 다르다는 점을 관련 연구 절에서 분명히 한다.
4. **⑥은 별도 축의 아이디어다.** O-RAN에서 near-RT RIC 대신 기지국끼리 학습하는 이유(데이터 주권, RIC 부하)를
   세울 수 있을 때만 진행한다.

## 5. 한계와 다음 확인

- 표준 모드 웹 검색 8회와 1차 기록 확인으로, 체계적 문헌 조사가 아니다. 특히 ③은 "찾지 못함"이지 "없음"이 아니다.
- 다음 확인할 검색어: "interference-aware link adaptation multi-agent", "transmission intention exchange MCS",
  "Xn CSI sharing reinforcement learning", "coordinated link adaptation CoMP learning".
- 인용할 논문은 본문을 읽고 수치와 방법을 확인한 뒤 `paper3/references.bib`에 넣는다.

## 6. 출처

- [arXiv 2602.02035](https://arxiv.org/abs/2602.02035), [arXiv 2010.04978](https://arxiv.org/abs/2010.04978),
  [arXiv 1902.01554](https://arxiv.org/abs/1902.01554), [arXiv 2511.19146](https://arxiv.org/abs/2511.19146),
  [arXiv 2608.14559](https://arxiv.org/abs/2608.14559), [AAMAS 2026 PDF](https://ifaamas.org/Proceedings/aamas2026/pdfs/TAYO7266.pdf),
  [arXiv 2512.03528](https://arxiv.org/abs/2512.03528)
- [arXiv 2111.00465](https://arxiv.org/abs/2111.00465), [arXiv 2303.08423](https://arxiv.org/abs/2303.08423),
  [arXiv 2402.12957](https://arxiv.org/abs/2402.12957), [arXiv 2412.20785](https://arxiv.org/abs/2412.20785),
  [arXiv 2408.08977](https://arxiv.org/abs/2408.08977), [arXiv 2604.23426](https://arxiv.org/abs/2604.23426),
  [arXiv 2509.23419](https://arxiv.org/abs/2509.23419)
- [ICML 2026 포스터 페이지](https://icml.cc/virtual/2026/83735), [arXiv 2603.00689](https://arxiv.org/abs/2603.00689),
  Dahal & Vaezi: DOI 10.1109/LCOMM.2025.3535898
