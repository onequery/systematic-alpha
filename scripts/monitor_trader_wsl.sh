#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="dashboard"
INTERVAL=3
ONCE=0
TAIL_LINES=20
EVENT_LIMIT=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE=1
      shift
      ;;
    --interval)
      INTERVAL="${2:-3}"
      shift 2
      ;;
    --tail-lines)
      TAIL_LINES="${2:-20}"
      shift 2
      ;;
    --event-limit)
      EVENT_LIMIT="${2:-20}"
      shift 2
      ;;
    --mode)
      MODE="${2:-dashboard}"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

DB_PATH="$ROOT_DIR/state/trader/trader.sqlite"

print_dashboard() {
  clear
  echo "[monitor_trader_wsl] $(TZ=Asia/Seoul date '+%F %T %Z')"
  echo "root=$ROOT_DIR"
  echo "mode=$MODE interval=${INTERVAL}s tail_lines=$TAIL_LINES event_limit=$EVENT_LIMIT"
  echo
  echo "== Processes =="
  ps -ef | grep -E "run_trader_wsl\\.sh --action daemon|systematic_alpha\\.trader\\.cli .* daemon" | grep -v grep || true
  echo

  if [[ -f "$DB_PATH" ]]; then
    echo "== DB Status =="
    sqlite3 "$DB_PATH" "SELECT COALESCE((SELECT updated_at FROM system_meta WHERE meta_key='precompute_done:'||strftime('%Y%m%d','now','+9 hours') LIMIT 1), '-') AS precompute_done;"
    sqlite3 "$DB_PATH" "SELECT COALESCE((SELECT captured_at FROM day_budget ORDER BY trade_date DESC LIMIT 1), '-') AS last_budget_snapshot;"
    echo
    echo "== Recent Events =="
    sqlite3 -readonly "$DB_PATH" "SELECT created_at || ' | ' || event_type || ' | ' || substr(payload_json,1,180) FROM events ORDER BY event_id DESC LIMIT $EVENT_LIMIT;" || true
  else
    echo "DB missing: $DB_PATH"
  fi
  echo
  echo "== Recent Trader Logs =="
  ls -1dt "$ROOT_DIR"/logs/trader/* 2>/dev/null | head -n 1 | while read -r latest_dir; do
    find "$latest_dir" -maxdepth 1 -type f | sort | tail -n 2 | while read -r f; do
      echo "-- $f"
      tail -n "$TAIL_LINES" "$f"
    done
  done
}

if [[ "$ONCE" == "1" ]]; then
  print_dashboard
  exit 0
fi

while true; do
  print_dashboard
  sleep "$INTERVAL"
done

