#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ts() {
  TZ=Asia/Seoul date '+%F %T KST'
}

log() {
  echo "[remove_trader_tasks] $(ts) $*"
}

log "start root=$ROOT_DIR"

log "pkill run_trader_wsl.sh daemon begin"
pkill -f "run_trader_wsl.sh --action daemon" || true
log "pkill run_trader_wsl.sh daemon done"
log "pkill systematic_alpha.trader.cli daemon begin"
pkill -f "systematic_alpha.trader.cli .* daemon" || true
log "pkill systematic_alpha.trader.cli daemon done"

existing_cron="$(crontab -l 2>/dev/null || true)"
filtered_cron="$(printf '%s\n' "$existing_cron" | awk 'index($0, "run_trader_wsl.sh")==0')"
log "cron entries loaded"

if [[ -n "$(printf '%s' "$filtered_cron" | sed '/^[[:space:]]*$/d')" ]]; then
  log "writing filtered crontab"
  printf '%s\n' "$filtered_cron" | crontab -
else
  log "removing crontab (empty after filter)"
  crontab -r 2>/dev/null || true
fi

log "complete"
echo "Trader cron tasks removed."
echo "Check: crontab -l"
echo "Check running: ps -ef | grep run_trader_wsl.sh | grep -v grep"
