#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

pkill -f "run_trader_wsl.sh --action daemon" || true
pkill -f "systematic_alpha.trader.cli .* daemon" || true

existing_cron="$(crontab -l 2>/dev/null || true)"
filtered_cron="$(printf '%s\n' "$existing_cron" | awk 'index($0, "run_trader_wsl.sh")==0')"

if [[ -n "$(printf '%s' "$filtered_cron" | sed '/^[[:space:]]*$/d')" ]]; then
  printf '%s\n' "$filtered_cron" | crontab -
else
  crontab -r 2>/dev/null || true
fi

echo "Trader cron tasks removed."
echo "Check: crontab -l"
echo "Check running: ps -ef | grep run_trader_wsl.sh | grep -v grep"

