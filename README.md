# systematic-alpha

Intraday stock candidate selector for Korea Investment & Securities (KIS) using:
- REST data (price, daily data, symbols)
- WebSocket data (KR) / REST polling (US)
- rule-based filtering/scoring for short-term trading candidates

This project is a CLI app (no UI). It reads secrets from `.env` and prints ranked picks.

## Features

- Two-stage filtering (KR/US)
  - Stage 1: daily/price snapshot filters
  - Stage 2: realtime metrics (strength, VWAP, bid/ask ratio, volume ratio)
- `--market kr` (default): domestic KR flow (REST + WebSocket)
- `--market us`: overseas US flow (REST + polling)
- Long-only directional mode by default (`change >= threshold`, `gap >= threshold`)
- Fallback fill rule: if stage1 candidates are fewer than `--final-picks`, thresholds are relaxed step-by-step to fill missing slots.
- Decision-time snapshot refresh: final scoring uses latest snapshot at selection completion time (not only early scan snapshot).
- Realtime quality gate: if eligible realtime coverage is below threshold (default `0.8`), signal can be invalidated for the day.
- Automatic overnight performance report (`selection -> close -> next open`) in CSV.
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

Default directional behavior is long-only.  
At final scoring, `change/gap` are recomputed from the latest snapshot at the actual selection-computation completion time.

The final ranking selects top `N` symbols (`--final-picks`, default `3`).

## Project Structure

```text
.
|-- main.py
|-- run_main.sh
|-- requirements.txt
|-- .env.example
|-- scripts/
|   |-- register_task.ps1
|   |-- register_us_task.ps1
|   |-- run_daily.ps1
|   |-- remove_task.ps1
|   `-- remove_us_task.ps1
`-- systematic_alpha/
    |-- cli.py
    |-- selector.py
    |-- selector_us.py
    |-- models.py
    |-- credentials.py
    |-- dotenv.py
    |-- helpers.py
    |-- data/
    |   `-- us_universe_default.txt
    `-- mojito_loader.py
```

## Prerequisites

- Python `>= 3.10` (tested on `3.12`)
- KIS API credentials
- Network access to KIS OpenAPI endpoints
- `mojito2` package (installed via `requirements.txt`)

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

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Required keys:
- `KIS_APP_KEY`
- `KIS_APP_SECRET`
- `KIS_ACC_NO` (format: `12345678-01`)

Optional:
- `KIS_USER_ID`
- `KIS_MOCK` (`0` or `1`)
- market variables:
  - `MARKET=kr|us`
  - `US_EXCHANGE=NASD|NYSE|AMEX`
  - `US_UNIVERSE_FILE=./systematic_alpha/data/us_universe_default.txt`
  - `US_POLL_INTERVAL=2.0`
- Telegram variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, etc.)
- strategy/runtime override variables listed in `.env.example`
  - quality/monitoring/report-related:
    `LONG_ONLY`, `MIN_EXEC_TICKS`, `MIN_ORDERBOOK_TICKS`,
    `MIN_REALTIME_CUM_VOLUME`, `MIN_REALTIME_COVERAGE_RATIO`,
    `INVALIDATE_ON_LOW_COVERAGE`,
    `STAGE1_LOG_INTERVAL`, `REALTIME_LOG_INTERVAL`,
    `OVERNIGHT_REPORT_PATH`

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

## Run

### Direct Python

```bash
python main.py --collect-seconds 600 --final-picks 3
```

US example:

```bash
python main.py --market us --exchange NASD --collect-seconds 600 --final-picks 3
```

### Shell wrapper

`run_main.sh` loads `.env` then executes `python main.py ...`:

```bash
./run_main.sh --collect-seconds 600 --final-picks 3
./run_main.sh --market us --exchange NASD --collect-seconds 600 --final-picks 3
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

US (`--market us`) format:
- `AAPL`
- `AAPL,Apple Inc.`
- `AAPL Apple Inc.`

## Windows Auto Run (Task Scheduler)

You do not need a 24/7 loop process.  
Use scheduled tasks at market-open times.

### KR 09:00 task registration

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_task.ps1
```

Default behavior:
- task name: `SystematicAlpha_0900`
- time: `09:00`
- schedule: weekdays only (Mon-Fri)
- runner: `scripts/run_daily.ps1 -Market KR`
- python: `C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe`
- startup delay: `5 sec` (to avoid exact open-time mismatch)
- internal retries: up to `4` attempts (`30s`, backoff `x2`, max `180s`)
- task-level restart: up to `2` restarts within `10 min`

### US open task registration (DST/STD dual trigger)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_us_task.ps1
```

Default behavior:
- task name: `SystematicAlpha_US_Open`
- triggers(KST): `22:30` and `23:30` on weekdays
- runner: `scripts/run_daily.ps1 -Market US -RequireUsOpen`
- runtime guard: open-window check (`09:30 ET + 20m`) + ET-day lock(1 run/day)

Custom example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_us_task.ps1 -UsExchange "NASD" -UsOpenWindowMinutes 20 -TaskName "SystematicAlpha_US"
```

### Verify registration

```powershell
Get-ScheduledTask -TaskName "SystematicAlpha_0900"
Get-ScheduledTask -TaskName "SystematicAlpha_US_Open"
```

### Run once now (manual test)

```powershell
Start-ScheduledTask -TaskName "SystematicAlpha_0900"
Start-ScheduledTask -TaskName "SystematicAlpha_US_Open"
```

### Check outputs

- logs: `logs/`
  - python run log: `logs/{kr|us}_daily_YYYYMMDD_HHMMSS_tryN.log`
  - runner log: `logs/runner_{kr|us}_YYYYMMDD_HHMMSS.log`
- json results: `out/`
  - `out/kr_daily_YYYYMMDD_HHMMSS.json`
  - `out/us_daily_YYYYMMDD_HHMMSS.json`

### Remove tasks

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\remove_task.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\remove_us_task.ps1
```

Notes:
- The PC must be on for automatic task execution.
- `remove_task.ps1` / `remove_us_task.ps1` are manual commands and run immediately.
- If the PC is sleeping at trigger time, execution can be delayed by OS/task settings.
- `scripts/run_daily.ps1` clears proxy env variables and uses project-local cache path for `mojito` token stability.
- `scripts/run_daily.ps1` reads Telegram settings from `.env` and sends notifications automatically when configured.
- If token issuance hits rate-limit (`EGW00133`), retry wait is automatically expanded to `65 sec`.
- US holidays/half-days are not fully modeled in this repo.
- US task prevents duplicate same-day runs via `out/us_run_lock_YYYYMMDD.txt` (ET date 기준).

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

US manual run example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_daily.ps1 -Market US -UsExchange "NASD" -RequireUsOpen -UsOpenWindowMinutes 20
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

US quick smoke example:

```bash
python main.py \
  --market us \
  --exchange NASD \
  --universe-file systematic_alpha/data/us_universe_default.txt \
  --max-symbols-scan 3 \
  --pre-candidates 3 \
  --final-picks 3 \
  --collect-seconds 0 \
  --min-change-pct 0 \
  --min-gap-pct 0 \
  --min-prev-turnover 0 \
  --output-json out/us_smoke_run.json
```

## Output

- Console table with top picks
- Optional JSON file via `--output-json`
- Overnight performance report CSV (default): `out/selection_overnight_report.csv`
  - columns include: selection datetime, entry price, same-day close, next-day open, intraday/overnight/total returns

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

## GitHub Publish Checklist

- Confirm secrets are only in `.env` (not in tracked files).
- Confirm generated outputs are ignored (`logs/`, `out/`, `*.mst`, `.cache/`).
- Run:
  - `git status`
  - `python main.py --help`
- Review staged diff before push:
  - `git diff --staged`

## Third-Party Code

- This project depends on `mojito2` from PyPI.
- `systematic_alpha/mojito_loader.py` also supports an optional local `./mojito` override for development; that folder is excluded from git by default.

## Disclaimer

This repository is for research/automation purposes only and is not financial advice.

