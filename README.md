# systematic-alpha (Agent-First)

This project is operated in **Agent mode**.
This repository is now **WSL-first (Linux shell)**.

- KR/US market signals are generated internally.
- `Agent Lab` (3 agents) consumes those signals and proposes BUY/SELL actions.
- Telegram notifications are focused on agent proposals and agent workflow events.

Manual "top-3 picks for discretionary trading" is no longer the primary workflow in this repository.

## Core Architecture

- `Signal Engine` (internal dependency)
  - Produces KR/US session JSON signals.
  - Output examples:
    - `out/kr/YYYYMMDD/results/*.json`
    - `out/us/YYYYMMDD/results/*.json`
- `Agent Lab` (primary operation layer)
  - `agent_a`: momentum/flow bias
  - `agent_b`: risk-first conservative bias
  - `agent_c`: counter-hypothesis/diversification bias
  - Runs multi-round weekly council debate and strategy updates.
  - Generates order proposals, review reports, and weekly council decisions.

## Install

```bash
conda create -n systematic-alpha python=3.12 -y
conda activate systematic-alpha
pip install -r requirements.txt
# openai is included in requirements.txt; install explicitly only if needed:
# pip install openai
```

## Environment

Copy `.env.example` to `.env`, then fill secrets.

Required:

- `KIS_APP_KEY`
- `KIS_APP_SECRET`
- `KIS_ACC_NO`

Recommended for notifications:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Optional for LLM-assisted agent updates:

- `AGENT_LAB_ENABLED=1`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini`
- `OPENAI_MAX_DAILY_COST=5.0`
- `AGENT_LAB_EXECUTION_MODE=mojito_mock` (mock-account API execution)
- `AGENT_LAB_AUTO_APPROVE=1` (auto execute without manual approval)
- `AGENT_LAB_USE_LIVE_FX=1` (try live USD/KRW rate first)
- `AGENT_LAB_FX_TIMEOUT_SECONDS=8` (timeout per FX provider call)
- `AGENT_LAB_TELEGRAM_USE_ENV_PROXY=0` (default; set `1` only if your network requires env proxy)
- `SYSTEMATIC_ALPHA_PROXY_MODE=auto` (`auto|off|clear_all`, network guard for broken proxy env)
- `AGENT_LAB_USDKRW_DEFAULT=1300` (fallback if live FX/cache unavailable)

## WSL One Command: Activate Everything

```bash
chmod +x scripts/*.sh
./scripts/register_all_tasks_wsl.sh
```

WSL scripts automatically resolve Python to `~/anaconda3/envs/systematic-alpha/bin/python`.
If `.env` contains a Windows-style `PYTHON_BIN`, it is ignored in WSL mode.

This registers cron-based automation in WSL:

- KR prefetch + open-time signal run + Agent ingest/propose/auto-execution
- US prefetch + open-time signal run + Agent ingest/propose/auto-execution
- Daily review, weekly council
- Telegram chat worker (`@reboot`)
- Auto strategy daemon (`@reboot`)
- Agent Lab initialization (`10,000,000 KRW`, `3 agents`) by default
- Plus immediate bootstrap: `telegram-chat` and `auto-strategy-daemon` are restarted right away (even if already running).

Default behavior:

- Agent proposals are auto-executed in paper account mode when `AGENT_LAB_AUTO_APPROVE=1`.
- Agent workflow notifications are sent from python orchestrator (if Telegram is configured).
- Manual order approval step is disabled in this WSL setup. Order proposal status is expected to be `EXECUTED` or `BLOCKED` (not `PENDING_APPROVAL`).

Quick check after registration:

```bash
crontab -l
ps -ef | grep run_agent_lab_wsl.sh | grep -v grep
```

## WSL One Command: Remove Everything

```bash
./scripts/remove_all_tasks_wsl.sh
```

## WSL One Command: Reset Tasks (Down + Up)

Use this after code/config updates when you want to fully restart scheduler + daemons in one line.

```bash
./scripts/reset_all_tasks_wsl.sh
```

## WSL One Command: Reset Tasks (Preserve State)

Use this during live operation to restart scheduler + daemons **without** re-running Agent Lab init.
This preserves current capital/accounting state, strategy versions, and agent memories.

```bash
./scripts/reset_tasks_preserve_state_wsl.sh
```

Equivalent one-liner:

```bash
INIT_AGENT_LAB=0 ./scripts/reset_all_tasks_wsl.sh
```

## Check Registered Cron Jobs

```bash
crontab -l
```

## Agent Notifications

When Telegram is configured, Agent Lab sends:

- Session ingest status
- Agent proposal summary (`BUY/SELL symbol x qty`)
- Auto trade execution results (`FILLED/REJECTED`, proposal id, symbol/qty)
- Failure events (ingest/propose/review/execution)
- Daily review completion
- Weekly council debate summary (champion, promoted versions, moderator summary)
- Weekly debate excerpts (opening/rebuttal, short form)
- OpenAI token/quota alerts during council (daily budget, quota, token/context limits)
- Interactive chat replies from agents (`/ask`, `/plan`, `/status`, `/queue`)
- Automatic data-reception self-heal events (critical failure detection + safe patch attempts)
- Automatic strategy-update events from auto strategy daemon
- OpenAI token/quota/budget alerts from auto strategy daemon

## Important Paths

- Agent DB: `state/agent_lab/agent_lab.sqlite`
- Agent identity: `state/agent_lab/agents/<agent_id>/identity.md`
- Agent memory: `state/agent_lab/agents/<agent_id>/memory.jsonl`
- Agent artifacts: `out/agent_lab/YYYYMMDD/`
- Agent action timeline: `out/agent_lab/YYYYMMDD/activity_log.jsonl`
- Agent logs: `logs/agent_lab/YYYYMMDD/`

## Manual Agent Commands (WSL)

```bash
# initialize
./scripts/run_agent_lab_wsl.sh --action init --capital-krw 10000000 --agents 3

# KR/US ingest + propose + auto-execute
./scripts/run_agent_lab_wsl.sh --action ingest-propose --market KR
./scripts/run_agent_lab_wsl.sh --action ingest-propose --market US

# review / council
./scripts/run_agent_lab_wsl.sh --action daily-review --date YYYYMMDD
./scripts/run_agent_lab_wsl.sh --action weekly-council --week YYYY-Www

# self-heal (manual trigger)
python -m systematic_alpha.agent_lab.cli --project-root . self-heal --market KR --date YYYYMMDD --log-path <log_file> --output-json <result_json>

# Telegram interactive worker
./scripts/run_agent_lab_wsl.sh --action telegram-chat

# Auto strategy daemon
./scripts/run_agent_lab_wsl.sh --action auto-strategy-daemon

# One-shot checks
./scripts/run_agent_lab_wsl.sh --action telegram-chat --chat-once
./scripts/run_agent_lab_wsl.sh --action auto-strategy-daemon --auto-strategy-once
```

## Telegram 사용 가이드

텔레그램에서 아래 명령어로 에이전트와 직접 소통할 수 있습니다.

사전 확인:

1. `.env`에 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`가 설정되어 있어야 합니다.
2. `telegram-chat` 데몬이 실행 중이어야 합니다.
3. 아래 명령으로 실행 상태를 확인할 수 있습니다.

```bash
ps -ef | grep run_agent_lab_wsl.sh | grep -v grep
```

주요 명령어:

| 명령어 | 용도 | 예시 입력 | 예상 응답(요약) |
|---|---|---|---|
| `/help` | 사용 가능한 명령 확인 | `/help` | 명령어 목록과 사용법 |
| `/agents` | 활성 에이전트 목록 확인 | `/agents` | `agent_a`, `agent_b`, `agent_c` 역할 요약 |
| `/status` | 전체 상태 요약 | `/status` | 최근 KR/US 실행 상태, 제안/주문/에러 요약 |
| `/status <agent_id>` | 특정 에이전트 상태 | `/status agent_a` | 해당 에이전트의 최신 포지션/최근 의사결정 |
| `/status KR\|US` | 시장별 상태 요약 | `/status KR` | KR 파이프라인 상태와 다음 실행 시각/남은 초 |
| `/status <agent_id> KR\|US` | 에이전트+시장 상태 | `/status agent_a US` | 해당 에이전트의 US 최신 제안/주문/다음 실행까지 남은 초 |
| `/queue` | 큐/스케줄 점검 | `/queue` | KR/US 파이프라인 마지막 실행 + 다음 실행 시각/남은 초 + 데몬 상태 |
| `/queue <agent_id> KR\|US` | 에이전트별 큐 점검 | `/queue agent_b KR` | 해당 에이전트의 KR 최신 proposal 상태 + 다음 실행 남은 초 |
| `/plan <agent_id>` | 현재 운용 계획 확인 | `/plan agent_b` | 금일 계획, 진입/회피 기준, 리스크 관점 |
| `/ask <agent_id> <질문>` | 자유 질의응답 | `/ask agent_c 오늘 상위 종목 대신 5위를 고른 이유?` | 해당 판단 근거와 대안 설명 |
| `/memory <agent_id>` | 최근 기억(학습 로그) 확인 | `/memory agent_a` | 최근 의사결정/교훈/다음 액션 |
| `/directive <agent_id> <지시문>` | 자연어 지시 생성(PENDING) | `/directive agent_a 거래량 급증 종목을 더 우선 고려해` | 지시 ID 생성, 상태 `PENDING` |
| `/setparam <agent_id> <key> <value> <메모>` | 파라미터 수정 지시(PENDING) | `/setparam agent_b min_strength 115 체결강도 기준 상향` | 파라미터 지시 ID 생성 |
| `/directives` | 지시 대기열 조회 | `/directives` | 승인/거절 대기 ID 목록 |
| `/approve <id> <메모>` | 지시 승인 적용 | `/approve 42 apply now` | 적용 완료, 전략/메모리 반영 결과 |
| `/reject <id> <사유>` | 지시 거절 | `/reject 42 risk too high` | 거절 처리, 사유 기록 |

실전 대화 예시:

1. 시장 분리 상태 점검
입력: `/status KR`
응답 예시: `KR pipeline + next_prefetch/next_signal-scan/next_agent-exec (in N seconds)`
2. 큐/남은 시간 점검
입력: `/queue agent_a US`
응답 예시: `US latest=BLOCKED, next_agent_exec=... (in N seconds), daemon status`
3. 전략 이유 질의
입력: `/ask agent_a 오늘 1순위 종목을 선택한 핵심 이유 3가지만 알려줘`
응답 예시: `체결강도 유지, VWAP 상단 유지, 호가비율 우위`
4. 전략 수정 요청
입력: `/setparam agent_b min_bid_ask_ratio 1.3 보수적으로 상향`
응답 예시: `directive_id=57, status=PENDING`
입력: `/approve 57 apply now`
응답 예시: `directive 57 applied, strategy version promoted`

지시(Directive) 처리 순서:

1. `/directive` 또는 `/setparam`으로 지시 생성 (`PENDING`)
2. `/directives`로 대기열 확인
3. `/approve` 또는 `/reject`로 확정
4. 적용된 지시는 `state/agent_lab/agents/<agent_id>/memory.jsonl`에 기록
5. `setparam` 적용 시 전략 버전이 갱신되고, 주간 회의 맥락에도 반영

운영 팁:

1. 실행 직후에는 `/status KR`, `/status US`, `/queue`로 먼저 상태/남은 시간을 확인한 뒤 질의하세요.
2. 전략 변경은 반드시 `PENDING -> approve` 순서로 적용하세요.
3. 응답이 없으면 `telegram-chat` 데몬 프로세스와 `logs/cron/agent_telegram_chat.log`를 확인하세요.

Identity and memory are persisted in `state/agent_lab/agents/<agent_id>/`.
If you stop and restart tasks for code updates, agents warm-start from identity + recent memory + latest checkpoint.

## Note on Legacy Signal Code

The legacy KR/US selector code is still required because Agent Lab currently ingests its signal outputs.
So selector code remains in the repository as an internal dependency.

## Disclaimer

This repository is for research and automation purposes only, not investment advice.
