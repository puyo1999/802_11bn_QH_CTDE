# 이동성, 연결(association) 동역학, AP 간 플래핑 모델과 논문 배치안

- 상태: 초안 (모델 정리와 논문 배치 검토)
- 작성일: 2026-10-28
- 관련 문서: `paper/main.tex`, `docs/random_topology_experiment_plan.md`

## 1. 문제 정리

현재 모델은 STA 위치와 연결(서빙 AP)이 에피소드 안에서 고정되어 있고, 정책은 지연 없는 관측을 받는다.
실제 환경에서 "실시간 RSSI 미반영"은 원인이 셋이다.

| 원인 | 시간 규모 | 현재 모델 |
|---|---|---|
| 채널 시간 변화 (도플러, 소규모 페이딩) | 수–수십 ms | 결정적 / 슬롯별 독립 Rayleigh (양 극단만) |
| 관측 지연 (RSSI·ACK 피드백 지연) | 1–수 슬롯 | 없음 |
| 이동성과 연결 변경 (로밍, 플래핑) | 수백 ms–초 | 없음 (연결 고정) |

STA가 AP2 근처로 이동해도 바로 AP2에 붙지 않으므로(sticky client), 연결은 위치와 별개의 상태로 모델링해야 한다.

## 2. 모델

### 2.1 이동 (가우스-마르코프)

$$
\mathbf{v}_k(t) = \beta\,\mathbf{v}_k(t-1) + (1-\beta)\,\bar{\mathbf{v}} + \sqrt{1-\beta^2}\,\sigma_v\,\mathbf{w}(t),\qquad
\mathbf{x}_k(t+1) = \mathbf{x}_k(t) + \mathbf{v}_k(t)\,T_{\text{slot}}
$$

### 2.2 측정 RSSI와 필터

$$
R_{j,k}(t) = P_{\text{rx}}\bigl(d_{j,k}(t)\bigr) + X_{j,k}(t),\qquad
\bar{R}_{j,k}(t) = (1-\alpha)\,\bar{R}_{j,k}(t-1) + \alpha\,R_{j,k}(t)
$$

- $X$: 섀도잉(3–6 dB)과 페이딩(Rayleigh 한 표본의 dB 표준편차 약 5.6 dB) 측정 잡음.

### 2.3 로밍 규칙 (히스테리시스, 유지 시간, 지연)

연결 상태 $a_k(t) \in \mathcal{N}$. 다음 두 조건이 $T_{\text{TTT}}$ 슬롯 동안 계속 성립하면 AP $j$로 전환한다.

$$
\bar{R}_{j,k}(t) > \bar{R}_{a_k(t),k}(t) + \Delta_{\text{hys}},\qquad \bar{R}_{a_k(t),k}(t) < R_{\text{roam}}
$$

- 전환 후 $T_{\text{HO}}$ 슬롯 동안 서비스 불가 (레거시 로밍 수백 ms, 802.11r 수십 ms).
- 옛 AP의 큐: 새 AP로 전달(지연 $T_{\text{fwd}}$) 또는 폐기 중 하나로 둔다.

### 2.4 잘못 연결된 구간의 시간 규모

AP 간격 $L$인 두 AP 사이 직선을 따라 이동하는 STA의 RSSI 차이는 $D(x) = 30\log_{10}\bigl(x/(L-x)\bigr)$ dB
(두 AP 쪽 변화가 동시에 반영됨). 경계($x = L/2$)에서 히스테리시스 $\Delta_{\text{hys}}$를 넘는 지점까지의 거리는

$$
x_\Delta - \frac{L}{2} = \frac{L}{2}\tanh\!\left(\frac{\Delta_{\text{hys}}\ln 10}{60}\right)
$$

이다. $L = 20$ m, $\Delta_{\text{hys}} = 3$ dB에서 약 1.15 m, 1 m/s 보행으로 약 1.2초(+ $T_{\text{TTT}}$ + $T_{\text{HO}}$)
→ 현재 에피소드(200 ms)보다 6배 이상 길다. 로밍 임계값 $R_{\text{roam}}$ 조건이 더 늦게 성립하면 그 지점까지 늘어난다.
따라서 에피소드를 수 초로 늘리거나, 연결을 에피소드 단위 상태로 다룬다.

### 2.5 AP 간 플래핑

경계 근처에서 필터링된 RSSI 차이

$$
D_k(t) = \bar{R}_{j,k}(t) - \bar{R}_{a,k}(t) = \mu_k(\mathbf{x}_k(t)) + \varepsilon_k(t),\qquad
\sigma_\varepsilon^2 \approx 2\sigma^2\frac{\alpha}{2-\alpha}
$$

- 슬롯당 전환 확률(경계, 유지 시간 $n$ 슬롯): $p_{\text{HO}} \le Q(\Delta_{\text{hys}}/\sigma_\varepsilon)$.
  필터로 표본 간 상관이 있어 정확한 값은 시뮬레이션으로 구한다.
- 플래핑: 전환 후 $T_{\text{pp}}$(1–5초) 안에 원래 AP로 복귀. 두 상태 마르코프 사슬에서 평균 체류 시간 $1/p_{\text{HO}}$.

### 2.6 비용과 트레이드오프

$$
L(\Delta_{\text{hys}}, T_{\text{TTT}}, \alpha) =
\mathbb{E}[\text{잘못 연결된 시간}]\cdot(\text{처리량 손실률})
+ \lambda_{\text{HO}}\,\bigl(T_{\text{HO}}\cdot\text{서비스율} + \text{큐 이전 손실}\bigr)
$$

- 느린 반응(큰 $\Delta$, 큰 $T_{\text{TTT}}$, 작은 $\alpha$): 플래핑 감소, 잘못 연결된 구간 증가.
- MAPC 협조는 잘못 연결된 구간의 손실(첫째 항)을 줄인다 → 최적 히스테리시스가 커지고 플래핑이 줄어든다.

### 2.7 재연결 없는 서빙 AP 전환 (802.11bn에서 논의되는 끊김 없는 로밍)

서빙 AP를 스케줄러가 바꿀 수 있으면, 전환 비용을 목적함수에 넣는다.

$$
\max \sum_t \Bigl[R(t) - c_{\text{sw}}\sum_k \mathbb{1}\bigl[a_k(t)\neq a_k(t-1)\bigr]\Bigr]
$$

이 항 때문에 oracle은 슬롯 단위가 아니라 여러 슬롯을 함께 보는 문제(동적 계획법, 유한 기간 탐색)가 된다.

### 2.8 시간 상관 채널과 관측 지연 (연결 고정 상태에서도 가능)

- 가우스-마르코프 페이딩: 복소 이득 $h(t) = \rho\,h(t-1) + \sqrt{1-\rho^2}\,n(t)$, $\rho = J_0(2\pi f_D T_{\text{slot}})$.
  6 GHz, 1 m/s에서 $f_D \approx 20$ Hz, $\rho \approx 0.99$.
- 관측 지연: 정책은 $o_i(t-d)$를 받는다 ($d \in \{0, 1, 5, 20\}$).
- 비교: (i) 지연된 관측만, (ii) 두 시간 규모 구조(학습자는 전송 집합, MCS는 최신 ACK 기반 ARF),
  (iii) 최근 $k$ 슬롯의 SINR·ACK 이력을 관측에 추가.

## 3. 논문 배치안

### 3.1 이번 논문 (`paper/main.tex`, exact-oracle gap analysis)

연결 고정을 유지한다. 넣는 것은 시간 상관 채널·관측 지연(선택)과 범위 밖 항목의 명시뿐이다.

| 위치 | 내용 | 분량 | 조건 |
|---|---|---|---|
| III. System Model, *Interference and link outcome* | 가우스-마르코프 페이딩을 페이딩 변형의 세 번째 옵션으로 한 문장 + 식 1개 | 3–4줄 | 2.8 실험을 할 때 |
| III. System Model, *Observations* | 관측 지연 $d$ 정의 한 문장 | 1–2줄 | 2.8 실험을 할 때 |
| VI. Experimental Setup | 지연 값, 비교 변형 (i)–(iii), 토폴로지와 시드 | 1문단 | 2.8 실험을 할 때 |
| VII. Results, 새 소절 *Robustness to channel aging and feedback delay* | 표 1개: 지연별 $\Delta_{\text{LA}}$, $\Delta_{\text{I}}$, 두 시간 규모 구조의 회복분 | 1문단 + 표 | 2.8 실험을 할 때 |
| VII. Results 또는 VIII. Discussion | 이동성 평가(STA를 1–5 m 옮긴 배치에서 학습 정책 성능 저하) | 2–3문장 | 평가만, 비용 적음 |
| VIII. Discussion, 개선 방향 문단 | 두 시간 규모 구조가 지연 관측의 약점을 메운다는 설계 지침 | 1–2문장 | 2.8 결과에 따라 |
| VIII. Discussion, 한계 문단 | "연결은 고정되어 있으며, 히스테리시스 기반 로밍 지연과 AP 간 플래핑은 후속 연구로 남긴다." | 1문장 | 항상 |
| IX. Conclusion, 마지막 문장 | 후속 연구 방향으로 이동성과 연결 동역학 한 구절 | 1구절 | 항상 |

- 2.1–2.7(이동, 로밍, 플래핑, 서빙 AP 전환)은 이번 논문에 넣지 않는다. 환경 구조(관측 차원, AP별 STA 슬롯,
  에피소드 길이)가 바뀌어 기존 결과와 비교 기준이 달라지기 때문이다.
- 2.8을 하지 않으면 이번 논문은 한계 문단 한 문장과 결론 한 구절만 바뀐다.

### 3.2 후속 논문 (가제: *MAPC-Assisted Roaming in IEEE 802.11bn: Coordination Gains in the Mis-Association Window*)

| 절 | 내용 | 이 문서의 해당 부분 |
|---|---|---|
| I. Introduction | sticky client와 플래핑 문제, MAPC가 트레이드오프를 옮긴다는 가설, 기여 목록 | 1, 2.6 |
| II. Related Work | 802.11k/v/r 로밍, 셀룰러 핸드오버(히스테리시스, TTT, 핑퐁), MAPC, 이동성 하의 학습 스케줄링 | — |
| III. System Model | 이동(2.1), 측정과 필터(2.2), 로밍 규칙(2.3), 핸드오버 지연과 큐 이전 | 2.1–2.3 |
| IV. Analysis | 잘못 연결된 구간 길이(2.4), 플래핑 확률 근사와 마르코프 사슬(2.5), 비용 식(2.6) | 2.4–2.6 |
| V. Problem Formulation | 전환 비용을 포함한 목적함수, 재연결 없는 서빙 AP 전환(2.7) | 2.7 |
| VI. Method | 연결 고정 per-slot oracle(이번 논문 재사용), 이상적 연결 상한, 유한 기간 oracle, 학습자 | 2.7, 이번 논문 IV |
| VII. Experimental Setup | 경로(AP1→AP2 횡단), 속도, $\Delta_{\text{hys}}$·$T_{\text{TTT}}$·$\alpha$ 범위, 무작위 토폴로지 재사용 | 2.1–2.3 |
| VIII. Results | (a) 연결 지연 손실, (b) MAPC가 회복하는 비율, (c) 히스테리시스–플래핑 트레이드오프 곡선과 MAPC의 최적점 이동 | 2.6 |
| IX. Discussion / X. Conclusion | 로밍 파라미터 설계 지침, 802.11bn 끊김 없는 로밍과의 관계 | 2.7 |

## 4. 다음 단계

1. 이번 논문: 무작위 토폴로지 2단계 반영 후, 한계 문단과 결론에 범위 밖 항목을 명시한다.
2. 2.8(시간 상관 채널·관측 지연)을 이번 논문에 넣을지 결정한다. 넣으면 실험 약 6–8시간.
3. 후속 논문: 이 문서의 2.1–2.7을 바탕으로 별도 계획 문서와 환경 확장을 시작한다.
