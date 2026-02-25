#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

load_env_file_safe() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    return 0
  fi
  # BOM/CRLF-safe load for env-like files.
  # shellcheck disable=SC1090
  source <(awk 'NR==1{sub(/^\xef\xbb\xbf/,"")} {sub(/\r$/,"")}1' "$file_path")
}

set -a
load_env_file_safe "$ROOT_DIR/config/agent_lab.config"
load_env_file_safe "$ROOT_DIR/.env"
set +a

resolve_python_bin() {
  local candidate="${PYTHON_BIN:-}"
  if [[ -n "$candidate" ]]; then
    if [[ "$candidate" =~ ^[A-Za-z]:\\ ]] || [[ "$candidate" == *"\\"* ]] || [[ "$candidate" == *.exe ]]; then
      candidate=""
    fi
  fi
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi
  if [[ -x "$HOME/anaconda3/envs/systematic-alpha/bin/python" ]]; then
    echo "$HOME/anaconda3/envs/systematic-alpha/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

usage() {
  cat <<'EOF'
Usage: scripts/monitor_agent_lab_wsl.sh [options]

Options:
  --mode dashboard|follow   monitor mode (default: dashboard)
  --interval SEC            dashboard refresh interval in seconds (default: 3)
  --tail-lines N            tail lines for each log section (default: 20)
  --event-limit N           number of recent DB events to show (default: 20)
  --payload-max N           max chars for summarized payload/log line (default: 220)
  --once                    render one dashboard frame and exit
  -h, --help                show this help

Examples:
  scripts/monitor_agent_lab_wsl.sh
  scripts/monitor_agent_lab_wsl.sh --mode follow --tail-lines 80
  scripts/monitor_agent_lab_wsl.sh --interval 2 --event-limit 30
EOF
}

MODE="dashboard"
INTERVAL=3
TAIL_LINES=20
EVENT_LIMIT=20
PAYLOAD_MAX=220
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-dashboard}"
      shift 2
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
    --payload-max)
      PAYLOAD_MAX="${2:-220}"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$MODE" in
  dashboard|follow) ;;
  *)
    echo "Unsupported --mode: $MODE (expected: dashboard|follow)" >&2
    exit 2
    ;;
esac

PYTHON_BIN="$(resolve_python_bin)"
DB_PATH="$ROOT_DIR/state/agent_lab/agent_lab.sqlite"
TODAY_KST="$(TZ=Asia/Seoul date +%Y%m%d)"
YDAY_KST="$(TZ=Asia/Seoul date -d '-1 day' +%Y%m%d)"

LOG_FILES=(
  "$ROOT_DIR/logs/cron/agent_auto-strategy-daemon_bootstrap.log"
  "$ROOT_DIR/logs/cron/agent_telegram-chat_bootstrap.log"
  "$ROOT_DIR/logs/cron/agent_kr_preopen.log"
  "$ROOT_DIR/logs/cron/agent_us_preopen.log"
  "$ROOT_DIR/logs/cron/agent_kr_close.log"
  "$ROOT_DIR/logs/cron/agent_us_close.log"
  "$ROOT_DIR/logs/cron/kr_daily.log"
  "$ROOT_DIR/logs/cron/us_daily.log"
)

prepare_logs() {
  local f
  for f in "${LOG_FILES[@]}"; do
    mkdir -p "$(dirname "$f")"
    touch "$f"
  done
}

show_processes() {
  echo "== Processes =="
  local ps_out out
  ps_out="$(ps -ef || true)"
  if command -v rg >/dev/null 2>&1; then
    out="$(
      printf '%s\n' "$ps_out" | rg -n \
        -e 'run_agent_lab_wsl\.sh --action (telegram-chat|auto-strategy-daemon)' \
        -e 'systematic_alpha\.agent_lab\.cli .* (telegram-chat|auto-strategy-daemon)' \
      || true
    )"
  else
    out="$(
      printf '%s\n' "$ps_out" | grep -En \
        'run_agent_lab_wsl\.sh --action (telegram-chat|auto-strategy-daemon)|systematic_alpha\.agent_lab\.cli .* (telegram-chat|auto-strategy-daemon)' \
      || true
    )"
  fi
  if [[ -n "$out" ]]; then
    echo "$out"
  else
    echo "(no matching daemon process found)"
  fi
}

show_daemon_health() {
  echo "== Daemon Health (from DB) =="
  if [[ ! -f "$DB_PATH" ]]; then
    echo "(db not found: $DB_PATH)"
    return
  fi
  DB_PATH="$DB_PATH" "$PYTHON_BIN" - <<'PY'
import os
import sqlite3
from datetime import datetime

db = os.environ["DB_PATH"]
con = sqlite3.connect(db)
cur = con.cursor()

def latest(event_type: str):
    cur.execute(
        "SELECT created_at FROM state_events WHERE event_type = ? ORDER BY event_id DESC LIMIT 1",
        (event_type,),
    )
    row = cur.fetchone()
    return row[0] if row else ""

now = datetime.now()
hb = latest("auto_strategy_heartbeat")
st = latest("auto_strategy_daemon_start")
tc = latest("telegram_chat_worker_start")
ae = latest("auto_strategy_daemon_error")
te = latest("telegram_chat_worker_error")

def age_seconds(iso_text: str) -> str:
    if not iso_text:
        return "-"
    try:
        dt = datetime.fromisoformat(str(iso_text))
        return str(int(max(0.0, (now - dt).total_seconds())))
    except Exception:
        return "?"

print(f"auto_strategy_daemon_start: {st or '-'}")
print(f"auto_strategy_heartbeat:    {hb or '-'} (age_sec={age_seconds(hb)})")
print(f"telegram_chat_worker_start: {tc or '-'}")
print(f"auto_strategy_daemon_error: {ae or '-'}")
print(f"telegram_chat_worker_error: {te or '-'}")
con.close()
PY
}

show_db_events() {
  echo "== Recent DB Events (state_events) =="
  if [[ ! -f "$DB_PATH" ]]; then
    echo "(db not found: $DB_PATH)"
    return
  fi
  DB_PATH="$DB_PATH" EVENT_LIMIT="$EVENT_LIMIT" PAYLOAD_MAX="$PAYLOAD_MAX" "$PYTHON_BIN" - <<'PY'
import json
import os
import sqlite3
from pathlib import Path

db = Path(os.environ["DB_PATH"])
limit = int(float(os.environ.get("EVENT_LIMIT", "20") or 20))
max_chars = int(float(os.environ.get("PAYLOAD_MAX", "220") or 220))
con = sqlite3.connect(str(db))
cur = con.cursor()

HEAVY_KEYS = {
    "proposals",
    "shadow_proposals",
    "execution_results",
    "orders",
    "fills",
    "attempts",
    "discussion",
    "rationale",
    "payload",
    "raw",
    "broker_response_json",
}
PRIORITY_KEYS = [
    "market",
    "date",
    "session_date",
    "status",
    "status_code",
    "signal_status",
    "invalid_reason",
    "ok",
    "matched",
    "blocked",
    "mismatch_count",
    "reason",
    "proposal_id",
    "agent_id",
    "returncode",
    "elapsed_sec",
    "timeout_sec",
    "timed_out",
]

def _clip(text: str) -> str:
    body = str(text or "")
    return body if len(body) <= max_chars else (body[:max_chars] + "...")

def _fmt_value(value):
    if isinstance(value, list):
        return f"[{len(value)}]"
    if isinstance(value, dict):
        return f"{{{len(value)}}}"
    return str(value)

def summarize_payload(raw: str) -> str:
    text = str(raw or "")
    try:
        payload = json.loads(text)
    except Exception:
        return _clip(text.replace("\n", " "))
    if not isinstance(payload, dict):
        return _clip(_fmt_value(payload))
    parts = []
    for key in PRIORITY_KEYS:
        if key in payload:
            parts.append(f"{key}={_fmt_value(payload[key])}")
    for key in HEAVY_KEYS:
        if key in payload:
            val = payload[key]
            if isinstance(val, list):
                parts.append(f"{key}=[{len(val)}]")
            elif isinstance(val, dict):
                parts.append(f"{key}={{{len(val)}}}")
            else:
                parts.append(f"{key}=<omitted>")
    if not parts:
        for idx, (k, v) in enumerate(payload.items()):
            if idx >= 4:
                break
            parts.append(f"{k}={_fmt_value(v)}")
    return _clip(", ".join(parts))

cur.execute(
    """
    SELECT created_at, event_type, payload_json
    FROM state_events
    ORDER BY event_id DESC
    LIMIT ?
    """,
    (limit,),
)
rows = cur.fetchall()
if not rows:
    print("(no rows)")
else:
    for created_at, event_type, payload_json in rows:
        print(f"{created_at} | {event_type} | {summarize_payload(payload_json)}")
con.close()
PY
}

render_jsonl_compact() {
  local path="$1"
  local tail_lines="$2"
  local max_chars="$3"
  FILE_PATH="$path" TAIL_LINES="$tail_lines" PAYLOAD_MAX="$max_chars" "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import deque
from pathlib import Path

path = Path(os.environ["FILE_PATH"])
tail_lines = int(float(os.environ.get("TAIL_LINES", "20") or 20))
max_chars = int(float(os.environ.get("PAYLOAD_MAX", "220") or 220))

HEAVY_KEYS = {
    "proposals",
    "shadow_proposals",
    "execution_results",
    "orders",
    "fills",
    "attempts",
    "discussion",
    "rationale",
    "payload",
    "raw",
    "broker_response_json",
}
PRIORITY_KEYS = [
    "market",
    "date",
    "session_date",
    "status",
    "status_code",
    "signal_status",
    "invalid_reason",
    "ok",
    "matched",
    "blocked",
    "mismatch_count",
    "reason",
    "proposal_id",
    "agent_id",
    "returncode",
    "elapsed_sec",
    "timeout_sec",
    "timed_out",
]

def _clip(text: str) -> str:
    body = str(text or "")
    return body if len(body) <= max_chars else (body[:max_chars] + "...")

def _fmt_value(value):
    if isinstance(value, list):
        return f"[{len(value)}]"
    if isinstance(value, dict):
        return f"{{{len(value)}}}"
    return str(value)

def summarize_payload(payload):
    if not isinstance(payload, dict):
        return _clip(_fmt_value(payload))
    parts = []
    for key in PRIORITY_KEYS:
        if key in payload:
            parts.append(f"{key}={_fmt_value(payload[key])}")
    for key in HEAVY_KEYS:
        if key in payload:
            val = payload[key]
            if isinstance(val, list):
                parts.append(f"{key}=[{len(val)}]")
            elif isinstance(val, dict):
                parts.append(f"{key}={{{len(val)}}}")
            else:
                parts.append(f"{key}=<omitted>")
    if not parts:
        for idx, (k, v) in enumerate(payload.items()):
            if idx >= 4:
                break
            parts.append(f"{k}={_fmt_value(v)}")
    return _clip(", ".join(parts))

buf = deque(maxlen=max(1, tail_lines))
if path.exists():
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line.rstrip("\n"))

if not buf:
    print("(empty)")
else:
    for raw in buf:
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            print(_clip(text))
            continue
        if isinstance(obj, dict):
            ts = str(obj.get("ts") or obj.get("created_at") or obj.get("updated_at") or "-")
            event_type = str(obj.get("event_type") or obj.get("event") or obj.get("type") or "-")
            payload = obj.get("payload")
            if payload is None:
                payload = {k: v for k, v in obj.items() if k not in {"ts", "event_type"}}
            print(f"{ts} | {event_type} | {summarize_payload(payload)}")
        else:
            print(_clip(text))
PY
}

tail_compact() {
  local path="$1"
  local tail_lines="$2"
  local max_chars="$3"
  FILE_PATH="$path" TAIL_LINES="$tail_lines" PAYLOAD_MAX="$max_chars" "$PYTHON_BIN" - <<'PY'
import os
from collections import deque
from pathlib import Path

path = Path(os.environ["FILE_PATH"])
tail_lines = int(float(os.environ.get("TAIL_LINES", "20") or 20))
max_chars = int(float(os.environ.get("PAYLOAD_MAX", "220") or 220))

def _clip(text: str) -> str:
    body = str(text or "")
    return body if len(body) <= max_chars else (body[:max_chars] + "...")

buf = deque(maxlen=max(1, tail_lines))
if path.exists():
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line.rstrip("\n"))

if not buf:
    print("(empty)")
else:
    for line in buf:
        print(_clip(line))
PY
}

show_activity_logs() {
  echo "== Activity Logs (today / yesterday, KST) =="
  local p1="$ROOT_DIR/out/agent_lab/$TODAY_KST/activity_log.jsonl"
  local p2="$ROOT_DIR/out/agent_lab/$YDAY_KST/activity_log.jsonl"
  if [[ -f "$p1" ]]; then
    echo "-- $p1"
    render_jsonl_compact "$p1" "$TAIL_LINES" "$PAYLOAD_MAX" || true
  else
    echo "-- $p1 (missing)"
  fi
  if [[ -f "$p2" ]]; then
    echo "-- $p2"
    render_jsonl_compact "$p2" "$TAIL_LINES" "$PAYLOAD_MAX" || true
  else
    echo "-- $p2 (missing)"
  fi
}

show_main_logs() {
  echo "== Main Cron Logs =="
  local f
  for f in \
    "$ROOT_DIR/logs/cron/agent_auto-strategy-daemon_bootstrap.log" \
    "$ROOT_DIR/logs/cron/agent_telegram-chat_bootstrap.log"; do
    echo "-- $f"
    tail_compact "$f" "$TAIL_LINES" "$PAYLOAD_MAX" || true
  done
}

clear_screen() {
  if [[ -t 1 ]] && [[ -n "${TERM:-}" ]]; then
    clear || true
  else
    printf '\n'
  fi
}

render_dashboard() {
  TODAY_KST="$(TZ=Asia/Seoul date +%Y%m%d)"
  YDAY_KST="$(TZ=Asia/Seoul date -d '-1 day' +%Y%m%d)"
  clear_screen
  echo "[monitor_agent_lab_wsl] $(TZ=Asia/Seoul date '+%F %T %Z')"
  echo "root=$ROOT_DIR"
  echo "mode=$MODE interval=${INTERVAL}s tail_lines=$TAIL_LINES event_limit=$EVENT_LIMIT payload_max=$PAYLOAD_MAX"
  echo
  show_processes
  echo
  show_daemon_health
  echo
  show_db_events
  echo
  show_activity_logs
  echo
  show_main_logs
  echo
  echo "Tip: Ctrl+C to stop monitoring."
}

run_follow_mode() {
  prepare_logs
  echo "[monitor_agent_lab_wsl] follow mode"
  echo "files:"
  printf '  - %s\n' "${LOG_FILES[@]}"
  echo "tail_lines=$TAIL_LINES"
  echo
  tail -n "$TAIL_LINES" -F "${LOG_FILES[@]}"
}

run_dashboard_mode() {
  while true; do
    render_dashboard
    if [[ "$ONCE" == "1" ]]; then
      break
    fi
    sleep "$INTERVAL"
  done
}

if [[ "$MODE" == "follow" ]]; then
  run_follow_mode
else
  run_dashboard_mode
fi
