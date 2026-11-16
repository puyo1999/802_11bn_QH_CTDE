# 노화 고려 링크 적응 (Aging-Aware Link Adaptation, AALA) 설계

- 상태: **설계 초안** (구현 전 검토용). 8절 선행 연구 조사 결과, 새 알고리즘이 아니라 **기존 처방의 회복률 평가**로 방향을 조정하는 것을 추천
- 작성일: 2026-11-16
- 근거: `docs/paper3_design_review.md`, LA0 정상 상태 결과(`experiments/cellular_results/la0_D20_S1*`), 아래 2절의 단일 링크 타당성 계산
- 관련 코드: `src/cellular/channel.py`(Gauss–Markov 페이딩), `src/cellular/linkadapt.py`(OLLA, 추정 채널), `src/cellular/cqi.py`

## 1. 문제

LA0(배치 20개, oracle 대상, $d = 4$)에서 지연 보고 + OLLA는 genie CQI 대비 기하평균 전송률이 3 km/h 82.7%, 30 km/h 49.7%,
60 km/h 43.1%였다. 같은 OLLA에 낡지 않은 보고(fresh)를 주면 82–85%로 속도와 무관했다. 즉 30 km/h 이상에서 손실의 대부분
(18–48점)이 **보고 노화**에서 나온다.

OLLA는 두 가지를 가정한다.
1. 보고된 SINR이 현재 SINR의 **편향만 있는 추정치**이고, 그 편향은 천천히 변한다.
2. 실패율을 **고정 목표(10%)**에 맞추는 것이 좋다.

노화가 크면 둘 다 틀린다. 보고는 현재 채널과 거의 무관한 값이 되고(60 km/h, 보고 나이 5에서 전력 상관 0.016),
10%를 지키려면 CQI를 극단적으로 낮춰야 한다. 실제로 최적 실패율은 불확실성이 클수록 높아진다.

## 2. 타당성: 단일 링크 이론 계산

스크립트: 저장소 밖 스크래치(`aging_feasibility.py`). 시뮬레이터와 같은 AR(1) 페이딩, 간섭 없음, 신호 페이딩만.
과거 전력 $g = |h(t-a)|^2$가 주어지면 현재 전력의 분포는 비중심 카이제곱이다.

$$
\frac{2|h(t)|^2}{1-c} \sim \chi'^2_2\!\left(\frac{2cg}{1-c}\right), \qquad c = \rho^{2a}.
$$

genie 대비 기대 스펙트럼 효율(%)과, 기대 처리량 규칙이 고른 결과의 실패율:

| 속도 | 보고 나이 $a$ | 전력 상관 $c$ | 평균 SNR | 고정 10% OLLA | 조건부 평균 + 10% | **기대 처리량 규칙 (ET)** | ET의 실패율 |
|---|---|---|---|---|---|---|---|
| 3 km/h | 5 | 0.991 | 10 dB | 90.5 | 91.0 | **91.3** | 8.4% |
| 30 km/h | 1 | 0.826 | 10 dB | 61.9 | 58.7 | **71.5** | 26.0% |
| 30 km/h | 5 | 0.385 | 10 dB | 38.6 | 41.7 | **61.1** | 36.0% |
| 30 km/h | 5 | 0.385 | 20 dB | 57.5 | 62.6 | **68.9** | 23.0% |
| 60 km/h | 5 | 0.016 | 0 dB | 0.0 | 0.0 | **52.2** | 63.6% |
| 60 km/h | 5 | 0.016 | 10 dB | 27.0 | 36.9 | **60.4** | 36.3% |
| 60 km/h | 5 | 0.016 | 20 dB | 48.7 | 58.0 | **67.2** | 19.9% |

읽는 법:
- **저속에서는 이득이 작다**(1–5점). 보고가 정확하므로 OLLA로 충분하다.
- **중·고속에서는 이득이 크다**(10–50점). 핵심은 예측 정확도보다 **"불확실성에 맞는 실패율을 고르는 것"**이다.
  조건부 평균(MMSE) 예측만으로는 거의 개선되지 않는다(0–10점).
- 60 km/h처럼 보고가 쓸모없을 때도 ET는 **통계적 채널 정보**(Rayleigh 분포)만으로 50–67%를 얻는다. 고정 10% OLLA는
  저 SNR에서 아예 전송하지 못한다(0%).
- 최적 실패율은 조건마다 5–64%로 달라진다. **어떤 고정 목표도 모든 UE와 조건에 맞을 수 없다.**

한계: 간섭의 페이딩 불확실성, 셀 침묵 여부, HARQ 재전송이 빠져 있다. 실제 시스템 이득은 이보다 작을 것이다(5절).

## 3. 방법 (AALA)

슬롯마다 셀은 대상 UE $u$, 부대역 $b$에 대해 다음 순서로 CQI를 고른다. 모두 셀이 가진 정보(지연 보고, 대규모 이득,
ACK/NACK)만 쓴다.

### 3.1 노화 추정기 (C1)

UE마다 보고 이력으로 전력 상관 $c_u$를 추정한다. 페이딩 보정 계수 $\kappa_{u,b}(t)$(보고 SINR ÷ 페이딩 없는 SINR, 이미
`linkadapt.fading_correction`에 있음)는 과거 페이딩 전력의 추정치이므로, 연속 보고 사이의 상관으로

$$
\hat{c}^{(1)}_u = \widehat{\mathrm{corr}}\bigl(\kappa_{u,b}(t), \kappa_{u,b}(t-1)\bigr), \qquad
\hat{c}_u = \bigl(\hat{c}^{(1)}_u\bigr)^{a},
$$

를 지수 이동 평균으로 추정한다(복소 AR(1)에서 전력의 1 슬롯 상관은 $\rho^2$, $a$ 슬롯 상관은 $\rho^{2a}$).
부대역들을 표본으로 함께 쓴다. 비교용으로 $c$를 정확히 아는 변형(C1\*)도 둔다.

### 3.2 조건부 SINR 분포 (C2)

신호 전력은 2절의 비중심 카이제곱, 과거 전력 $g = \kappa_{u,b}$, 상관 $\hat{c}_u$. 간섭은 대규모 이득으로 계산한
평균(침묵 정보가 있으면 반영, 없으면 모든 셀 전송 가정)에 지수 분포 페이딩을 준다. 비율 분포는 닫힌형이 없으므로
**표본 근사**(슬롯마다 $M = 64$개 공통 난수로 SINR 표본을 만들고 각 CQI의 성공 확률 $\hat{P}_q$를 계산)를 쓴다.
계산량은 UE × 부대역 × 64 × 15로 작다.

### 3.3 기대 처리량 CQI 선택 (C3)

$$
q^{\text{AALA}}_{u,b}(t) = \arg\max_{q} \; e_q \, \hat{P}_q\bigl(\gamma_{u,b}(t) \ge \theta_q \bigm| \text{보고}, \hat{c}_u, \Delta_u\bigr).
$$

실패율 목표를 정하지 않는다. 불확실성이 크면 규칙이 스스로 높은 실패율을 고른다.

### 3.4 보정 루프 (C4): OLLA를 대체

OLLA는 실패율을 고정 목표에 맞추는 루프다. AALA는 **예측 확률이 실제 성공률과 맞도록** 오프셋 $\Delta_u$(dB, SINR 표본 전체에
더함)를 조정한다.

$$
\Delta_u \leftarrow \Delta_u + \mu \bigl(\mathbb{1}[\text{ACK}] - \hat{P}_{q}\bigr),
$$

기대값이 0이 되는 지점은 $\mathbb{E}[\text{ACK}] = \mathbb{E}[\hat{P}_q]$, 즉 **확률 보정**이 맞는 지점이다. 실패율 자체는
자유롭게 최적값으로 간다. 대규모 이득 오차, 침묵 가정 오차, 간섭 분포 근사 오차처럼 천천히 변하는 편향을 이 루프가 흡수한다.

### 3.5 노화 고려 스케줄링 지표 (C5, 인과적 스케줄러용)

PF 지표와 대상 선택에 낡은 순간 전송률 대신 **기대 전송률** $\hat{R}_{u,b} = \max_q e_q \hat{P}_q$를 쓴다. 낡은 페이딩
피크를 쫓아 대상을 고르는 일을 줄인다. 인과적 기준 정책(`baselines.*(sim, rx=estimated_rx(sim))`)과 학습자의 관측에 함께 넣는다.

## 4. 평가 설계 (LA1)

LA0와 같은 틀(전체 버퍼, $C = 3$, $U = 4$, $B = 4$, PF 목적, 워밍업 300 + 측정 1,000 슬롯, 배치 20개)에서 CQI 선택만 바꾼다.

### 4.1 비교 대상

| 이름 | 내용 | 역할 |
|---|---|---|
| genie | 현재 SINR로 최고 CQI | 상한 |
| fresh | 낡지 않은 보고 + OLLA 10% | 노화가 없을 때 |
| OLLA-10 | 지연 보고 + OLLA 10% (LA0의 local) | 표준 기준 |
| **OLLA-tuned** | 조건마다 목표 BLER을 {0.1, 0.2, 0.3, 0.5} 중 최선으로 고른 OLLA | **강한 기준**: 이득이 "목표 조정"만으로 나오는지 가림 |
| MMSE+OLLA | 조건부 평균 예측 + OLLA 10% | 예측만의 효과 |
| AALA | C1–C4 | 제안 |
| AALA\* | C1 대신 정확한 $c$ | 추정 오차의 비용 |
| ET-genie | 과거 페이딩을 정확히 알고 정확한 조건부 분포로 ET | 같은 보고로 가능한 인과적 상한 |

대상 선택은 두 가지로 돌린다. (i) LA0와 같은 genie 대상(손실 분해 비교용), (ii) **인과적 대상**(추정 채널 + C5). 설계 검토 4번 항목과
LA0의 확정 변형 해석 문제를 해결하는 것이 (ii)다.

### 4.2 지표

기하평균 전송률(주), 처리량, 5퍼센타일, 실패율. **노화 비용 회복률** = (AALA − OLLA-10) / (fresh − OLLA-10).

### 4.3 가설

| 번호 | 가설 | 근거 |
|---|---|---|
| H1 | 30 km/h에서 AALA가 노화 비용의 절반 이상을 회복한다 | 2절: 10 dB에서 38.6 → 61.1 |
| H2 | 60 km/h 이득의 대부분은 통계적 채널 정보에서 온다(AALA\*와 AALA의 차이 작음) | 2절: $c \approx 0$에서도 ET 50–67% |
| H3 | OLLA-tuned도 일부를 회복하지만, UE·SNR마다 최적 실패율이 달라 AALA보다 낮다 | 2절: 최적 실패율 5–64% |
| H4 | 3 km/h에서는 AALA와 OLLA-10의 차이가 2점 이내다(해를 끼치지 않음) | 2절: 저속 이득 1–5점 |

### 4.4 요인과 계산량

$d \in \{0, 4, 8\}$, $v \in \{3, 30, 60\}$, 대상 2종, 변형 8개. LA0 기준(변형 5 × 대상 3 = 15 에피소드가 작업당 약 2.5분)으로
변형 8 × 대상 2 = 16 에피소드라 거의 같은 계산량: **약 1.7시간**(4코어). AALA는 표본 근사 때문에 에피소드당 비용이 1.5–2배일 수
있어 **2–3시간**으로 잡는다. 작업별 저장과 이어서 실행은 LA0 코드를 그대로 쓴다.

## 5. 위험과 대응

| 위험 | 대응 |
|---|---|
| **HARQ가 없음.** 실제 NR은 실패 시 재전송·결합하므로 10% BLER 목표가 합리적이고, 실패의 비용이 작다 | (a) 리뷰어 지적에 대비해 OLLA-tuned를 강한 기준으로 둔다. (b) 단순 HARQ 모델(Chase 결합, 최대 재전송 1–3회)을 확장으로 추가. HARQ가 있으면 실패 비용이 줄어 최적 실패율이 더 높아지므로, AALA 쪽에 불리하지 않을 것으로 본다. 단, 확인 필요 |
| **AR(1) 페이딩은 실제 채널이 아님.** AALA의 분포 모델이 시뮬레이터와 같아서 유리할 수 있음 | 실제 도플러 스펙트럼(Jakes, 합-정현파)을 따르는 페이딩 옵션을 추가해 모델 불일치 상황에서 재평가. C4 보정 루프가 불일치를 얼마나 흡수하는지가 핵심 |
| 간섭 페이딩·침묵 불확실성으로 단일 링크 이득이 줄어듦 | 2절은 상한. LA1에서 실제 이득을 잼. ET-genie로 남은 여지를 함께 보고 |
| 노화 추정기(C1)의 수렴 시간 | 워밍업 300 슬롯 안에서 수렴하는지 테스트. 안 되면 초기값을 셀 평균으로 |
| 기여의 새로움: 불확실성 고려 링크 적응·BLER 목표 적응은 선행 연구가 있을 수 있음 | 구현 전에 선행 연구 검색(키워드: outdated CSI rate adaptation, adaptive BLER target, expected-throughput MCS, channel aging link adaptation). 차별점은 **oracle 손실 분해 안에서 "노화 비용"을 분리해 그 회복률로 평가**하는 것 |

## 6. 구현 계획

1. `src/cellular/aging.py`: 노화 추정기(C1), 조건부 SINR 표본(C2), ET 선택(C3), 보정 루프(C4)
2. `linkadapt.LinkAdapter`에 모드 추가: `olla_tuned`(목표 인자), `mmse`, `aala`, `aala_known_c`, `et_genie`
3. `baselines`: C5 기대 전송률 지표를 쓰는 인과적 대상 선택
4. `experiments/cellular_gap.py la1`: LA0 구조 재사용(작업별 저장, 정상 상태 지표)
5. 테스트
   - $c = 1$, 간섭 없음에서 ET가 genie와 같음
   - $c = 0$에서 ET가 Rayleigh 분포의 최적 CQI(닫힌형 $\arg\max_q e_q e^{-\theta_q/\bar\gamma}$)와 같음
   - 보정 루프가 의도적으로 넣은 3 dB 편향을 흡수함
   - 노화 추정기가 알려진 $\rho$에서 $c$를 오차 10% 이내로 추정
6. 단계: 단일 링크 검증(2절 수치 재현) → LA1 소규모(배치 4개) → 본 실행(배치 20개)

## 7. 논문 배치

| 선택지 | 내용 |
|---|---|
| A. paper3에 포함 | 분석(oracle 손실 분해) → 발견(손실의 대부분은 노화) → 처방(AALA)의 흐름. 기여 목록에 "노화 비용을 겨냥한 링크 적응" 추가 |
| B. paper4로 분리 | paper3는 분석 논문으로 두고, AALA를 HARQ·Jakes 채널까지 포함한 독립 논문으로 |

LA1 결과가 H1을 만족하면 A를 추천한다. paper3의 결론이 "문제 발견"에서 "문제 해결"로 강해진다.

## 8. 선행 연구와 새로움 (2026-11-16 검색)

검색 키워드: outdated CSI rate adaptation, adaptive BLER target, expected-throughput MCS selection, channel aging link adaptation,
Thompson sampling link adaptation, RL link adaptation. 서지 정보는 arXiv API와 Crossref 기록으로 확인했고, 본문은 읽지 않았다.

### 8.1 찾은 연구

| 연구 | 내용 (초록·요약 기준) | AALA 구성 요소와의 관계 | 확인 |
|---|---|---|---|
| Kang, Jang, Kim, Kim, "Outage Analysis of Multi-Antenna Rate Adaptive Systems With Outdated Feedback" | 낡은 피드백 아래 전송률 선택의 outage 분석, outage 제약 하 평균 처리량 최대화 | **C3(기대 처리량 선택)과 같은 원리** | IEEE TWC 13(10), 5453–5466, 2014, DOI 10.1109/TWC.2014.2339833 |
| Guharoy, Mehta, "Joint evaluation of reduced feedback scheme, scheduling, and rate adaptation in OFDMA systems with feedback delays" | 피드백 지연 하 OFDMA 전송률 적응, 지연에 따른 문턱 조정 | C3 | GLOBECOM 2012, DOI 10.1109/GLOCOM.2012.6503838 |
| Pellaco, Saxena, Bengtsson, Jaldén, "Wireless link adaptation with outdated CSI — a hybrid data-driven and model-based approach" | 낡은 CSI 하 모델 기반 + 데이터 기반 링크 적응 | **C1–C3과 가장 가까움** | SPAWC 2020, DOI 10.1109/SPAWC48557.2020.9154263 |
| Saxena, Tullberg, Jaldén, "Reinforcement Learning for Efficient and Tuning-Free Link Adaptation" (LTSLA) | SINR 확률분포를 두고 전송 결과로 베이즈 갱신, OLLA 대비 최대 50% 이득 보고 | **C2 + C4(확률 모델을 ACK/NACK로 보정)와 같은 발상** | arXiv 2010.08651 (2020) |
| Park, Daniels, Heath, "Optimizing the Target Error Rate for Link Adaptation" | HARQ 처리량을 최대화하는 목표 오류율을 닫힌형으로, 최적 목표는 SINR에 반비례 | "고정 10% 목표는 최적이 아니다"라는 발견 자체 | GLOBECOM 2015, DOI 10.1109/GLOCOM.2015.7417770 |
| Wiesmayr et al., "SALAD: Self-Adaptive Link Adaptation" | 순간 BLER 목표를 피드백 제어로 조정, 5G 시험망에서 OLLA 변형들 대비 최대 15% | C4와 목적이 겹침(OLLA 대체) | arXiv 2510.05784 (2025) |
| Díaz-Ruiz et al., 두 편 (채널 노화 완화용 CSI 예측, 링크 적응용 CSI 예측의 성능-복잡도) | GRU·LSTM·Wiener 예측기로 노화 완화 | C1(노화 추정·예측) | arXiv 2510.24400, 2511.20160 (2025) |
| Demirel et al., "Generalization in Reinforcement Learning for Radio Access Networks" | RL 링크 적응, 휴리스틱 대비 약 10% | RL 대안 | arXiv 2507.06602 (2025) |
| Mota et al., "Adaptive Modulation and Coding based on Reinforcement Learning for 5G Networks" | Q-learning MCS 선택 | RL 대안 | arXiv 1912.04030, GLOBECOM 2019 |
| Vinjam et al., "Link Adaptation Using Joint-Thompson Sampling" | MCS 간 단조성을 쓰는 Thompson sampling | 베이지안 대안 | SPCOM 2026, DOI 10.1109/SPCOM71105.2026.11622985 |
| Aghayev et al., "Improving 5G AI-RAN MCS Selection by Predicting Retransmissions" | 재전송 예측으로 MCS 선택, OLLA의 보수성 지적 | 문제 인식 공유 | arXiv 2609.09324 (2026) |

### 8.2 판단

- **AALA를 "새로운 링크 적응 알고리즘"으로 내세우기는 어렵다.**
  - 낡은 CSI 아래 기대 처리량을 최대화하는 전송률 선택(C3)은 2012–2014년부터 분석돼 있다.
  - 모델 기반 낡은 CSI 링크 적응(C1–C3)은 Pellaco et al.(2020)이 가깝다.
  - 확률 모델 + ACK/NACK 베이즈 갱신(C2 + C4)은 LTSLA(2020)가 같은 발상이다.
  - "고정 목표 BLER은 최적이 아니다"는 Park et al.(2015)과 SALAD(2025)가 이미 보였다.
- **남는 새로움은 "알고리즘"이 아니라 "협조 스케줄링과의 결합"과 "정확 oracle 안에서의 분해"다.**
  1. **분해:** 정확 oracle 대비 손실을 협조 · OLLA 운용 비용 · 노화 비용으로 나누고, 기존 처방들이 노화 비용의 몇 %를 회복하는지 같은 기준으로 비교하는 연구는 찾지 못했다.
  2. **협조와의 결합:** LA0에서 OLLA 운용 비용이 셀 침묵이 많을수록 커졌다(전체 재사용 9점 → oracle 대상 15–18점). 보고가 "모든 셀 전송"을 가정하기 때문이다. 협조 스케줄러가 간섭 집합을 슬롯마다 바꾸는 상황의 링크 적응, 그리고 노화된 정보 아래에서의 침묵 결정(기대 전송률 기반)은 이번 검색에서 다룬 연구가 없었다. 다만 검색 범위가 좁아 "찾지 못함"이지 "없음"은 아니다.

### 8.3 방향 조정 제안

| 선택지 | 내용 | 평가 |
|---|---|---|
| **A. paper3에 "기존 처방의 회복률" 실험으로 넣기 (추천)** | AALA를 제안 방법이 아니라 **기존 처방들의 대표 구현**으로 둔다. LA1에서 고정 10% OLLA, 목표 조정 OLLA(Park/SALAD 계열), 기대 처리량 선택(Kang/Pellaco 계열), 확률 모델 보정(LTSLA 계열)을 같은 oracle 분해 안에서 비교하고, 노화 비용 회복률을 보고한다 | 새로움 주장 없이 논문을 강하게 함. 인용으로 위치가 분명함 |
| B. paper4로 "협조 인식 링크 적응" | 8.2의 2번(간섭 집합이 바뀌는 협조 스케줄링 + 노화)을 겨냥한 방법. 침묵 정보 반영 보고 보정, 기대 전송률 기반 침묵 결정 | 새로움 가능성은 있으나 추가 선행 연구 검색과 설계가 필요 |
| C. AALA를 새 알고리즘으로 주장 | — | 선행 연구와 겹쳐서 권하지 않음 |

A를 택하면 4.1절의 비교 대상 이름을 선행 연구 계열로 바꾸고(예: "ET (Kang et al. 계열)"), 기여 문장은 "노화 비용과 그 회복률의 정량화"로 쓴다.

## 9. 확정이 필요한 항목

- [ ] HARQ 모델을 LA1에 포함할지, 확장으로 미룰지
- [ ] Jakes 페이딩 옵션을 LA1에 포함할지
- [ ] 표본 수 $M$(64)과 보정 루프 단계 $\mu$
- [ ] 논문 배치(A 또는 B)
