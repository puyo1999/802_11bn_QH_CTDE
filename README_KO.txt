C4 디버깅 로그 패치 (기존 파일/체크포인트 덮어쓰기 방지)

포함 파일은 기존 프로젝트의 동일 경로에 적용할 수 있는 계측 버전입니다.
이 폴더의 파일을 프로젝트에 바로 덮어쓰지 말고, 먼저 원본을 백업하세요.

계측 대상:
- src/debug_c4.py: QH_DEBUG_C4=1일 때만 로그 활성화
- src/agents/observation_codec.py: VQ 인덱스/코드 사용/복원 오차
- src/agents/ctde_ppo.py: 선택된 STA/MCS와 logits 형태
- src/training/centralized_execution.py: 원본-복원 관측 차이, 중앙 액션
- src/training/ppo.py: PPO log-prob 차이, ratio, VQ loss, gradient
- experiments/train_ablation.py: 학습/평가 step의 액션과 env info 요약

Windows PowerShell에서 적용 후 로그 활성화:
  $env:QH_DEBUG_C4 = "1"
  $env:QH_DEBUG_C4_LOG = ".\debug_logs\c4_debug.log"

그 다음 기존 결과 폴더를 재사용하지 말고 새로운 output_dir로 실행하세요. 예:
  python .\experiments\train_ablation.py --seed 1 --condition C4 --topologies T3 --steps 200 --output_dir .\results_csv\C4_T3_seed1_debug

주의:
1. train_ablation.py는 지정 output_dir에 모델/CSV를 저장하므로 새 디렉터리를 사용합니다.
2. 환경 내부의 per-AP queue/SINR/실제 decode 성공은 env info에 해당 필드가 있어야 로그됩니다. 현재 업로드된 env 파일은 없으므로 환경 내부 계측은 아직 추가되지 않았습니다.
3. PPO actor_step 로그의 gradient norm은 clipping 호출 뒤의 반환값을 기록하도록 추가되어 있어, gradient norm 원값이 필요하면 별도 수정이 필요합니다.
