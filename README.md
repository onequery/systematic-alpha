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
  - 각 에이전트는 KR/US를 동시에 운용하며, 시장 간 신호 강도/현재 노출을 함께 반영해 자동으로 시장별 예산을 배분합니다.
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

정규장 시간 강제(거래 안전장치):
- `AGENT_LAB_ENFORCE_MARKET_HOURS=1`이면 KR(09:00~15:35 KST), US(09:30~16:05 ET) 외 시간에는 주문 제안/실행을 차단합니다.

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
- WSL(Ubuntu) `cron`은 작업별 타임존 실행을 지원하지 않으므로, US 작업은 KST에서 DST/비DST 후보 시각 2개에 등록하고 ET 시각 일치 조건으로 실제 실행을 게이트합니다.

전략 승격 규칙(주간회의):
- 일반 승격: `점수>=0.60` 2주 연속 + `리스크 위반<=3회` + `위반율<=15%` + `주간 제안수>=10`
- 강제 보수형 승격: `3주 연속` 고위험(`위반>=4회` 또는 `제안>=10 && 위반율>=25%`)이면 즉시 보수형 파라미터로 활성화
- 보수형 강제 승격 시 `risk_budget_ratio`, `exposure_cap_ratio`, `position_cap_ratio`, `max_daily_picks`, 손실한도 등이 자동 보수화됩니다.

교차시장(KR/US) 동시운용 규칙:
- 기본은 KR/US `50:50` 목표 비중이며, 각 시장의 최근 신호 강도 차이에 따라 동적으로 기울기를 적용합니다.
- 각 시장 주문 제안 시 해당 시장의 목표 노출까지로 매수 예산을 cap 하므로, 한 시장이 전체 현금을 선점하지 않도록 제한합니다.
- 아래 전략 파라미터로 조정할 수 있습니다.
  - `market_split_kr` (KR 기본 비중)
  - `market_min_weight`, `market_max_weight` (시장별 최소/최대 비중)
  - `market_tilt_scale` (신호 강도 기반 비중 이동 폭)
  - `market_signal_lookback_days` (교차시장 신호 참조 기간)

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
