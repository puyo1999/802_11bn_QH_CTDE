# 개념도(Fig. 1) 구성안

- 상태: 초안 (PPT로 직접 작성 예정)
- 작성일: 2026-10-27
- 관련 문서: `paper/main.tex` (System Model, Method 절)

## 1. 목적과 위치

- **위치**: 서론 마지막 문단 직전 또는 System Model 절 첫머리. 논문의 첫 그림(Fig. 1)이 된다.
- **목적**: 독자가 본문을 읽기 전에 두 가지를 한눈에 잡게 한다.
  1. 무엇을 스케줄링하는가: AP마다 슬롯마다 (대상 STA 또는 침묵, MCS)를 정한다.
  2. 무엇을 측정하는가: 정확 oracle을 기준으로 학습 스케줄러의 손실을 세 조각으로 나눈다.
- 크기: 2단 폭 그림(`figure*`, 폭 약 18 cm, 높이 6–7 cm). (a)와 (b)를 좌우로 둔다.

## 2. 패널 (a): MAPC 시스템 모델

**배치** (T3와 비슷한 2×2 AP 배치, 실제 좌표와 무관한 개념 그림)

| 요소 | 표현 | 아이콘/도형 |
|---|---|---|
| AP 4개 | AP1–AP4, 정사각형 모서리 | 라우터 또는 Wi-Fi AP 아이콘 |
| 일반 STA | AP마다 1–2개 | 노트북 또는 스마트폰 아이콘 |
| 중첩 STA | AP1–AP2 사이 1개, 강조색 | 같은 아이콘 + 주황 테두리 |
| 커버리지 | AP마다 옅은 원 | 반투명 원, 중첩 영역은 두 원이 겹치게 |
| 서빙 링크 | AP → STA 실선 화살표 | 검정 |
| 간섭 | 다른 AP → STA 점선 화살표 1–2개만 | 회색 점선 |
| AP 간 협조 | AP 사이 얇은 연결선 + "MAPC coordination" | 파선, 백홀 느낌 |

**슬롯 결정 표시** (패널 오른쪽 또는 아래의 작은 표)

```
Slot t   AP1    AP2    AP3    AP4
target   STA2   mute   STA1   STA1
MCS      7      –      11     9
```

- "mute"를 꼭 넣는다: 논문의 핵심 설계(침묵 행동이 있어야 Co-TDMA를 표현할 수 있음)를 보여 준다.
- 표 아래 한 줄: `Co-TDMA: one AP per slot / Co-SR: several APs per slot`

**주석**: 중첩 STA 옆에 "fails if AP2 transmits" (hard overlap 제약).

## 3. 패널 (b): 평가 틀과 손실 분해

**왼쪽: 세 종류의 스케줄러** (위에서 아래로 상자 3개)

| 상자 | 내용 | 아이콘 |
|---|---|---|
| Exact per-slot oracle | 모든 (STA \| mute) 조합 전수 탐색 + 최고 복호 가능 MCS, genie | 돋보기 또는 별 |
| Learned schedulers (CTDE) | MAPPO: AP마다 로컬 actor (0-bit) / Central PPO: 전 AP 정보 actor (∞-bit) | 신경망 아이콘 |
| Reference | Co-TDMA (시간 분할), All-transmit SINR | 시계 |

- MAPPO와 Central PPO는 같은 상자 안에 두 줄로: 학습 시 중앙 critic, 실행 시 actor의 정보량 차이를 화살표로 표시.
- MCS 제어 세 가지(learned / ARF / ideal)는 학습자 상자 아래에 작은 스위치 모양으로.

**오른쪽: 손실 분해 막대** (세로 누적 막대 1개, 0–100%)

```
100% ┬ Exact oracle
     │  Δopt  (최적화 손실)
     ├ Central PPO + ideal MCS
     │  ΔI    (정보 + 결합 행동 선택)
     ├ MAPPO + ideal MCS
     │  ΔLA   (링크 적응 손실)
     ├ MAPPO + learned MCS
     │
  0% ┴
- - - - Co-TDMA (점선, 헤드룸 H = 1 − Co-TDMA/oracle)
```

- 막대 높이는 개념용이다. 실제 수치를 넣지 않는다(결과 절과 혼동 방지).
- 기호는 본문과 같게: `Δ_LA`, `Δ_I`, `Δ_opt`, `H`.

## 4. 스타일 규칙

- **글꼴**: 본문과 같은 계열(Times 또는 Helvetica), 그림 안 최소 8 pt.
- **색**: 3–4색 이내. 강조(중첩 STA, 손실 조각)는 결과 그림과 같은 색을 쓴다.
  MAPPO `#2a78d6`(파랑), Central PPO `#eb6834`(주황), 회색 `#52514e`.
- **흑백 인쇄**: 색만으로 구분하지 않는다. 실선/점선/파선과 라벨을 함께 쓴다.
- **아이콘**: 한 종류의 세트(PPT 삽입 → 아이콘)만 쓰고, 선 굵기와 크기를 통일한다.
- **내보내기**: PPT에서 그림 영역만 선택 → "그림으로 저장" → **PDF**(벡터).
  `paper/figures/fig_concept.pdf`로 저장하면 본문에 바로 넣는다.
- **캡션**(문장형, 마침표로 끝냄) 초안:
  > System model and evaluation framework. (a) In every slot, each AP selects a target STA or silence
  > and an MCS; an overlap STA fails when an overlapping AP transmits. (b) The exact per-slot oracle
  > serves as the reference, and the loss of a learned scheduler splits into the link-adaptation gap
  > Δ_LA, the information gap Δ_I, and the residual optimization gap Δ_opt.

## 5. 작업 순서

1. PPT 슬라이드 크기를 18 cm × 7 cm로 설정한다.
2. (a) 배치 → 링크 → 슬롯 표 순서로 그린다.
3. (b) 상자 3개 → 누적 막대 → 기호 라벨 순서로 그린다.
4. PDF로 내보내 `paper/figures/fig_concept.pdf`에 넣으면, 본문 참조와 캡션은 Claude가 반영한다.
