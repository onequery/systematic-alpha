# systematic-alpha (Agent-First)

This project is operated in **Agent mode**.

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
- `AGENT_LAB_TELEGRAM_USE_ENV_PROXY=0` (default; set `1` only if your network requires env proxy)
- `SYSTEMATIC_ALPHA_PROXY_MODE=auto` (`auto|off|clear_all`, network guard for broken proxy env)

## One Command: Activate Everything

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_all_tasks.ps1
```

This command registers all required tasks and initializes Agent Lab:

- KR signal generation tasks
- US signal generation tasks
- Agent Lab post-session/review/weekly tasks
- Agent Lab Telegram chat worker task (logon trigger)
- Agent Lab auto strategy daemon task (logon trigger, autonomous strategy update timing)
- Agent initialization (`agent_a`, `agent_b`, `agent_c`)

Default behavior:

- Base KR/US selector Telegram notifications are disabled.
- Agent Lab notifications remain enabled (if Telegram is configured).

## One Command: Remove Everything

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\remove_all_tasks.ps1
```

## Check Registered Tasks

```powershell
Get-ScheduledTask -TaskName "SystematicAlpha*" | Format-Table TaskName, State -AutoSize
```

## Agent Notifications

When Telegram is configured, Agent Lab sends:

- Agent proposal summary (`BUY/SELL symbol x qty`)
- Failure events (ingest/propose/review)
- Daily review completion
- Weekly council debate summary (champion, promoted versions, moderator summary)
- Weekly debate excerpts (opening/rebuttal, short form)
- OpenAI token/quota alerts during council (daily budget, quota, token/context limits)
- Interactive chat replies from agents (`/ask`, `/plan`, `/status`)
- Automatic data-reception self-heal events (critical failure detection + safe patch attempts)
- Automatic strategy-update events from auto strategy daemon
- OpenAI token/quota/budget alerts from auto strategy daemon

## Important Paths

- Agent DB: `state/agent_lab/agent_lab.sqlite`
- Agent identity: `state/agent_lab/agents/<agent_id>/identity.md`
- Agent memory: `state/agent_lab/agents/<agent_id>/memory.jsonl`
- Agent artifacts: `out/agent_lab/YYYYMMDD/`
- Agent logs: `logs/agent_lab/YYYYMMDD/`

## Manual Agent Commands (Optional)

```powershell
# initialize
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action init -CapitalKrw 10000000 -Agents 3

# ingest + propose
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action ingest-propose -Market KR -Date YYYYMMDD
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action ingest-propose -Market US -Date YYYYMMDD

# review / council
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action daily-review -Date YYYYMMDD
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action weekly-council -Week YYYY-Www

# self-heal (manual trigger)
python -m systematic_alpha.agent_lab.cli --project-root . self-heal --market KR --date YYYYMMDD --log-path <log_file> --output-json <result_json>

# Telegram interactive worker
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action telegram-chat

# Auto strategy daemon (agents decide strategy-update timing)
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action auto-strategy-daemon

# One-shot check
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_lab.ps1 -Action auto-strategy-daemon -AutoStrategyOnce
```

## Telegram Agent Chat Commands

Send these commands in the configured Telegram chat:

- `/help`
- `/agents`
- `/status`
- `/status agent_a`
- `/plan agent_b`
- `/ask agent_c Why did you avoid the top symbol today?`
- `/memory agent_a`
- `/directive agent_a Consider sector leadership in your weekly review.`
- `/setparam agent_b min_strength 115 tighten entry quality`
- `/directives`
- `/approve 42 apply now`
- `/reject 42 not aligned with risk rules`

Identity and memory are persisted in `state/agent_lab/agents/<agent_id>/`.
If you stop and restart tasks for code updates, agents warm-start from identity + recent memory + latest checkpoint.

Directive workflow:

1. Create directive with `/directive` or `/setparam` (status=`PENDING`)
2. Review queue with `/directives`
3. Apply with `/approve <id>` or reject with `/reject <id>`
4. Applied directives are saved to agent memory, and `/setparam` creates a new promoted strategy version automatically
5. Applied freeform directives are injected into weekly council debate context so agents can reflect them in next strategy updates

## Note on Legacy Signal Code

The legacy KR/US selector code is still required because Agent Lab currently ingests its signal outputs.
So selector code remains in the repository as an internal dependency.

## Disclaimer

This repository is for research and automation purposes only, not investment advice.
