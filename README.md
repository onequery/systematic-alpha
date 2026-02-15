# systematic-alpha

Intraday stock candidate selector for Korea Investment & Securities (KIS) using:
- REST data (price, daily data, symbols)
- WebSocket data (execution, orderbook)
- rule-based filtering/scoring for short-term trading candidates

This project is a CLI app (no UI). It reads secrets from `.env` and prints top picks.

## Features

- Two-stage filtering
  - Stage 1: daily/price snapshot filters
  - Stage 2: realtime metrics (strength, VWAP, bid/ask ratio, volume ratio)
- Fallback fill rule: if stage1 candidates are fewer than `--final-picks`, thresholds are relaxed step-by-step to fill missing slots.
- Configurable thresholds via CLI arguments or environment variables
- JSON output export for downstream automation
- Telegram notifications for retry/failure/success status (optional)
- Live progress/heartbeat logs during scan and realtime collection
- Local `.env` loader (no external dotenv dependency)

## Strategy Summary

The default rules are aligned with a numeric intraday workflow:
- daily change threshold
- opening gap threshold
- previous-day turnover threshold
- execution strength maintenance
- intraday volume ratio
- bid/ask remaining ratio
- price vs VWAP
- low-break check

The final ranking selects top `N` symbols (`--final-picks`, default `3`).

## Project Structure

```text
.
├─ main.py
├─ run_main.sh
├─ requirements.txt
├─ .env.example
├─ systematic_alpha/
│  ├─ cli.py
│  ├─ selector.py
│  ├─ models.py
│  ├─ credentials.py
│  ├─ dotenv.py
│  ├─ helpers.py
│  └─ mojito_loader.py
└─ mojito/                # vendored upstream wrapper
```

## Prerequisites

- Python `>= 3.10` (tested on `3.12`)
- KIS API credentials
- Network access to KIS OpenAPI endpoints

## Installation

### 1) Create/activate environment (example: conda)

```bash
conda create -n systematic-alpha python=3.12 -y
conda activate systematic-alpha
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill credentials:

```bash
cp .env.example .env
```

Required keys:
- `KIS_APP_KEY`
- `KIS_APP_SECRET`
- `KIS_ACC_NO` (format: `12345678-01`)

Optional:
- `KIS_USER_ID`
- `KIS_MOCK` (`0` or `1`)
- Telegram variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, etc.)
- strategy/runtime override variables listed in `.env.example`
  - monitoring-related: `STAGE1_LOG_INTERVAL`, `REALTIME_LOG_INTERVAL`

### Telegram notifications (optional)

If you want retry/failure/success alerts in Telegram:

1. Create a bot in Telegram (`@BotFather`) and get bot token.
2. Open chat with that bot once (send any message).
3. Get your `chat_id` from bot updates.
4. Add values to `.env`:

```env
TELEGRAM_ENABLED=1
TELEGRAM_BOT_TOKEN=<your_bot_token>   # no leading "bot"
TELEGRAM_CHAT_ID=<your_chat_id>
# optional:
# TELEGRAM_THREAD_ID=<topic_thread_id>
# TELEGRAM_DISABLE_NOTIFICATION=1
```

When configured, `scripts/run_daily.ps1` sends:
- start notification at run begin (default)
- retry notifications with short log tail
- final failure notification with last error tail
- final success notification with top picks summary + output json path

Note: Telegram keys in `.env` are prioritized over inherited OS environment variables.

## Run

### Direct Python

```bash
python main.py --collect-seconds 600 --final-picks 3
```

### Shell wrapper

`run_main.sh` loads `.env` then executes `python main.py ...`:

```bash
./run_main.sh --collect-seconds 600 --final-picks 3
```

See all options:

```bash
python main.py --help
```

### Universe file format (`--universe-file`)

Supported line formats:
- `005930`
- `005930,SAMSUNG_ELEC`
- `005930 SAMSUNG_ELEC`

If name is omitted, the app tries to fill it from KIS symbol master automatically.

## Windows Auto Run at 09:00 (Task Scheduler)

You do not need a 24/7 loop process.  
Use a scheduled task that runs once at market open time.

### 1) Register task

Run PowerShell as your normal user and execute:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_task.ps1
```

Default behavior:
- task name: `SystematicAlpha_0900`
- time: `09:00`
- schedule: weekdays only (Mon-Fri)
- runner: `scripts/run_daily.ps1`
- python: `C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe`
- startup delay: `5 sec` (to avoid exact open-time mismatch)
- internal retries: up to `4` attempts (`30s`, backoff `x2`, max `180s`)
- task-level restart: up to `2` restarts within `10 min`

Custom time/task name example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_task.ps1 -TaskName "SystematicAlpha_Open" -At "09:00"
```

### 2) Verify task registration

```powershell
Get-ScheduledTask -TaskName "SystematicAlpha_0900"
```

### 3) Run once now (manual test)

```powershell
Start-ScheduledTask -TaskName "SystematicAlpha_0900"
```

### 4) Check outputs

- logs: `logs/`
  - python run log: `logs/daily_YYYYMMDD_HHMMSS_tryN.log`
  - runner/notification log: `logs/runner_YYYYMMDD_HHMMSS.log`
- json results: `out/`

### 5) Remove task

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\remove_task.ps1
```

Notes:
- The "PC must be on at 09:00" condition applies only to automatic scheduled execution.
- `remove_task.ps1` is a manual command and works immediately when you run it (not tied to 09:00).
- If the PC is sleeping at 09:00, the scheduled run may be delayed depending on OS power/task settings.
- `scripts/run_daily.ps1` clears proxy env variables and uses project-local cache path for `mojito` token stability.
- `scripts/run_daily.ps1` reads Telegram settings from `.env` and sends notifications automatically when configured.
- If token issuance hits rate-limit (`EGW00133`), retry wait is automatically expanded to `65 sec`.

### Reliability knobs (`scripts/run_daily.ps1`)

You can tune retry behavior manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_daily.ps1 `
  -StartDelaySeconds 5 `
  -MaxAttempts 4 `
  -RetryDelaySeconds 30 `
  -RetryBackoffMultiplier 2 `
  -MaxRetryDelaySeconds 180 `
  -NotifyTailLines 20
```

To disable start notification for a specific manual run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_daily.ps1 -NotifyStart:$false
```

Execution monitor behavior:
- `run_daily.ps1` now streams Python output in real time (no end-of-run dump).
- each attempt prints start time, command line, live log file path, and elapsed seconds.
- startup/retry waiting now prints elapsed/remaining seconds every second.
- stage1 scan prints progress every `STAGE1_LOG_INTERVAL` symbols (default `20`).
- realtime collection prints heartbeat every `REALTIME_LOG_INTERVAL` seconds (default `10`).

## Smoke Run (off-market quick check)

Use very small universe and disable realtime collection:

```bash
python main.py \
  --universe-file smoke_codes.txt \
  --max-symbols-scan 3 \
  --pre-candidates 3 \
  --final-picks 3 \
  --collect-seconds 0 \
  --min-change-pct 0 \
  --min-gap-pct 0 \
  --min-prev-turnover 0 \
  --output-json out/smoke_run.json
```

## Output

- Console table with top picks
- Optional JSON file via `--output-json`

Example JSON path:
- `out/smoke_run.json`

## Troubleshooting

- `KeyError: 'access_token'`:
  - usually token endpoint returned an error payload (not token payload)
  - check credentials and API rate limits
- KIS token issue `EGW00133`:
  - token issuance is limited (retry after around 1 minute)
- Proxy failures (`127.0.0.1:9` etc):
  - clear `HTTP_PROXY` / `HTTPS_PROXY` variables
- Permission errors on cache path:
  - ensure user has write permission to home cache path

## Security Notes

- Never commit `.env` to git
- Rotate keys immediately if exposed
- Treat output files as potentially sensitive (strategy + symbols)

## Third-Party Code

- `mojito/` directory is vendored from `mojito2`-related source.
- Its license is included at `mojito/LICENSE`.

## Disclaimer

This repository is for research/automation purposes only and is not financial advice.
