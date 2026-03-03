#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/logs/cron"

existing_cron="$(crontab -l 2>/dev/null || true)"
filtered_cron="$(printf '%s\n' "$existing_cron" | awk 'index($0, "run_trader_wsl.sh")==0')"

new_entries=$(cat <<EOF
5 8 * * 1-5 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_trader_wsl.sh" --profile prod --log-profile trader --action precompute --market ALL >> "$ROOT_DIR/logs/cron/trader_precompute.log" 2>&1
40 8 * * 1-5 cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_trader_wsl.sh" --profile prod --log-profile trader --action snapshot-budget >> "$ROOT_DIR/logs/cron/trader_budget_snapshot.log" 2>&1
@reboot cd "$ROOT_DIR" && /usr/bin/env bash "$ROOT_DIR/scripts/run_trader_wsl.sh" --profile prod --log-profile trader --action daemon >> "$ROOT_DIR/logs/cron/trader_daemon_bootstrap.log" 2>&1
EOF
)

{
  printf '%s\n' "$filtered_cron" | sed '/^[[:space:]]*$/d'
  printf '%s\n' "$new_entries"
} | crontab -

pkill -f "run_trader_wsl.sh --action daemon" || true
nohup /usr/bin/env bash "$ROOT_DIR/scripts/run_trader_wsl.sh" --profile prod --log-profile trader --action daemon >> "$ROOT_DIR/logs/cron/trader_daemon_bootstrap.log" 2>&1 &

echo "Trader cron tasks registered."
echo "Check: crontab -l"
echo "Check running: ps -ef | grep run_trader_wsl.sh | grep -v grep"
