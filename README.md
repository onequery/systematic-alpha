# systematic-alpha (WSL 전용, Agent 자동운용)

이 레포는 KR/US 신호를 바탕으로 3개 에이전트가 자동으로 매매/리뷰/토의를 수행하는 프로젝트입니다.

## 1. 핵심 동작

- 신호 생성
  - KR/US 장 시작 구간에서 신호 JSON 생성
  - 예: `out/kr/YYYYMMDD/results/*.json`, `out/us/YYYYMMDD/results/*.json`
- Agent Lab
  - `agent_a`: 모멘텀/수급 강화형
  - `agent_b`: 리스크 우선형
  - `agent_c`: 반대가설/다양성형
- 데몬
  - `telegram-chat`: 텔레그램 명령 수신/응답
  - `auto-strategy-daemon`: 장중 반복 모니터링/제안/자동실행

## 2. 설치

```bash
conda create -n systematic-alpha python=3.12 -y
conda activate systematic-alpha
pip install -r requirements.txt
```

## 3. 환경변수

`.env.example`를 `.env`로 복사 후 값 입력:

- 필수
  - `KIS_APP_KEY`
  - `KIS_APP_SECRET`
  - `KIS_ACC_NO`
- 텔레그램
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
- OpenAI(권장)
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
  - `OPENAI_MAX_DAILY_COST`

알림 필터(중요):

- `AGENT_LAB_NOTIFY_EVENTS`
  - 기본값: `trade_executed,preopen_plan,session_close_report,weekly_council`
  - 이 목록에 없는 이벤트는 텔레그램 푸시를 보내지 않습니다.

## 4. 작업 등록/해제

```bash
chmod +x scripts/*.sh
./scripts/register_all_tasks_wsl.sh
```

전체 제거:

```bash
./scripts/remove_all_tasks_wsl.sh
```

상태 보존 리셋(추천):

```bash
./scripts/reset_tasks_preserve_state_wsl.sh
```

## 5. 자동 스케줄

### KR

- 07:30 KST: 유니버스 프리패치
- 08:50 KST: 개장 10분 전 플랜 보고
- 09:00 KST: 신호 생성(run_daily)
- 15:40 KST: 장 종료 10분 후 결과/토의 보고

### US (미국 동부시간 기준)

- 08:30 ET: 유니버스 프리패치
- 09:20 ET: 개장 10분 전 플랜 보고
- 09:30 ET: 신호 생성(run_daily)
- 16:10 ET: 장 종료 10분 후 결과/토의 보고

### 공통

- 매주 일요일 08:00 KST: 주간 토의 보고
- `@reboot`: `telegram-chat`, `auto-strategy-daemon` 자동 시작

참고:
- `auto-strategy-daemon`은 기본값에서 주간회의를 임의 호출하지 않습니다.
- 주간회의/주간보고는 위 일요일 스케줄 태스크가 담당합니다.

## 6. 텔레그램으로 받는 보고(요구사항 반영)

아래 5가지만 자동 보고합니다.

1. 에이전트 매매(체결) 발생 시
2. KR 개장 10분 전 플랜 보고
3. KR 마감 10분 후 일일 결과 + 에이전트 토의 요약
4. US 개장/마감도 2~3과 동일
5. 주간 에이전트 토의 보고

그 외 이벤트(하트비트, 데몬 시작, 장중 모니터링 등)는 기본적으로 텔레그램 푸시를 보내지 않습니다.
설명이 필요한 보고(개장 전 플랜, 장 종료 후 결과/토의, 주간 토의 요약)는 OpenAI LLM이 활성화된 경우 LLM 설명문을 함께 생성합니다.

## 7. 텔레그램 명령어

- `/help`
- `/agents`
- `/status`
- `/status KR`
- `/status US`
- `/status agent_a KR`
- `/queue`
- `/queue agent_b US`
- `/plan agent_a`
- `/ask agent_b 지금 계획 요약해줘`
- `/memory agent_c`
- `/directive agent_a 장중 변동성 급증 시 노출 축소`
- `/setparam agent_b min_strength 115 보수적으로 상향`
- `/directives`
- `/approve <id>`
- `/reject <id>`

## 8. 수동 실행 명령

```bash
# 초기화
./scripts/run_agent_lab_wsl.sh --action init --capital-krw 10000000 --agents 3

# 개장 전 플랜 보고
./scripts/run_agent_lab_wsl.sh --action preopen-plan --market KR --date YYYYMMDD
./scripts/run_agent_lab_wsl.sh --action preopen-plan --market US --date YYYYMMDD

# 장 마감 후 결과/토의 보고
./scripts/run_agent_lab_wsl.sh --action close-report --market KR --date YYYYMMDD
./scripts/run_agent_lab_wsl.sh --action close-report --market US --date YYYYMMDD

# 주간 토의
./scripts/run_agent_lab_wsl.sh --action weekly-council --week YYYY-Www

# 데몬
./scripts/run_agent_lab_wsl.sh --action telegram-chat
./scripts/run_agent_lab_wsl.sh --action auto-strategy-daemon
```

## 9. 경로

- DB: `state/agent_lab/agent_lab.sqlite`
- 에이전트 정체성: `state/agent_lab/agents/<agent_id>/identity.md`
- 에이전트 메모리: `state/agent_lab/agents/<agent_id>/memory.jsonl`
- 산출물: `out/agent_lab/YYYYMMDD/`
- 주간 산출물: `out/agent_lab/YYYYMMDD_weekly/`
- 로그: `logs/agent_lab/YYYYMMDD/`, `logs/cron/`

## 10. 트러블슈팅

- 텔레그램 무응답 시
  1. `ps -ef | grep run_agent_lab_wsl.sh | grep -v grep`
  2. `.env`의 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 확인
  3. `logs/cron/agent_telegram_chat.log` 확인
  4. `./scripts/reset_tasks_preserve_state_wsl.sh` 실행

- 레거시 상태 정리

```bash
/home/heesu/anaconda3/envs/systematic-alpha/bin/python -m systematic_alpha.agent_lab.cli --project-root . sanitize-state --retain-days 30 --keep-agent-memories 300
```
