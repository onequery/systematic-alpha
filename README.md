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
- Configurable thresholds via CLI arguments or environment variables
- JSON output export for downstream automation
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
- strategy/runtime override variables listed in `.env.example`

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
