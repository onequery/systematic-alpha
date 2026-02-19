#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_DEFAULT="$HOME/anaconda3/envs/systematic-alpha/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
INIT_AGENT_LAB="${INIT_AGENT_LAB:-1}"
INIT_CAPITAL_KRW="${INIT_CAPITAL_KRW:-10000000}"
INIT_AGENTS="${INIT_AGENTS:-3}"
MARK_START="# >>> systematic-alpha tasks start >>>"
MARK_END="# <<< systematic-alpha tasks end <<<"

mkdir -p "$ROOT_DIR/logs/cron"

NEW_BLOCK=$(cat <<EOF
$MARK_START
CRON_TZ=Asia/Seoul
30 7 * * 1-5 cd "$ROOT_DIR" && "$PYTHON_BIN" scripts/prefetch_kr_universe.py >> "$ROOT_DIR/logs/cron/kr_prefetch.log" 2>&1
0 9 * * 1-5 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_daily_wsl.sh" --market KR >> "$ROOT_DIR/logs/cron/kr_daily.log" 2>&1
20 9 * * 1-5 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" --action ingest-propose --market KR --auto-approve >> "$ROOT_DIR/logs/cron/agent_kr.log" 2>&1

CRON_TZ=America/New_York
30 8 * * 1-5 cd "$ROOT_DIR" && "$PYTHON_BIN" scripts/prefetch_us_universe.py --output-csv "$ROOT_DIR/out/us/\$(TZ=Asia/Seoul date +\\%Y\\%m\\%d)/cache/us_sp500_constituents.csv" >> "$ROOT_DIR/logs/cron/us_prefetch.log" 2>&1
30 9 * * 1-5 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_daily_wsl.sh" --market US --exchange NASD >> "$ROOT_DIR/logs/cron/us_daily.log" 2>&1
45 9 * * 1-5 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" --action ingest-propose --market US --auto-approve >> "$ROOT_DIR/logs/cron/agent_us.log" 2>&1

CRON_TZ=Asia/Seoul
10 7 * * * cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" --action daily-review >> "$ROOT_DIR/logs/cron/agent_daily_review.log" 2>&1
0 8 * * 6 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" --action weekly-council >> "$ROOT_DIR/logs/cron/agent_weekly_council.log" 2>&1
@reboot cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" --action telegram-chat >> "$ROOT_DIR/logs/cron/agent_telegram_chat.log" 2>&1
@reboot cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" --action auto-strategy-daemon >> "$ROOT_DIR/logs/cron/agent_auto_strategy.log" 2>&1
$MARK_END
EOF
)

CURRENT="$(crontab -l 2>/dev/null || true)"
if [[ -n "$CURRENT" ]]; then
  CURRENT="$(printf '%s\n' "$CURRENT" | sed "/$MARK_START/,/$MARK_END/d")"
fi

{
  if [[ -n "$CURRENT" ]]; then
    printf "%s\n" "$CURRENT"
  fi
  printf "%s\n" "$NEW_BLOCK"
} | crontab -

echo "WSL cron tasks registered."
echo "Check: crontab -l"

if [[ "$INIT_AGENT_LAB" == "1" ]]; then
  echo "Initializing Agent Lab state..."
  /usr/bin/env bash "$ROOT_DIR/scripts/run_agent_lab_wsl.sh" \
    --action init \
    --capital-krw "$INIT_CAPITAL_KRW" \
    --agents "$INIT_AGENTS"
fi
