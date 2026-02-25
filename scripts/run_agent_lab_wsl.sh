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
    # Ignore Windows executables when running inside WSL.
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

PYTHON_BIN="$(resolve_python_bin)"
ACTION="ingest-propose"
MARKET="KR"
RUN_DATE="$(TZ=Asia/Seoul date +%Y%m%d)"
DATE_EXPLICIT="0"
WEEK_ID=""
CAPITAL_KRW=10000000
AGENTS=3
DATE_FROM=""
DATE_TO=""
CHAT_ONCE="0"
AUTO_STRATEGY_ONCE="0"
CHAT_POLL_TIMEOUT=25
CHAT_IDLE_SLEEP=1.0
CHAT_MEMORY_LIMIT=20
AUTO_STRATEGY_POLL="${AGENT_LAB_DAEMON_POLL_SECONDS:-60}"
AUTO_STRATEGY_COOLDOWN=180
AUTO_STRATEGY_MAX_UPDATES=2
RECONCILE_SUBMITTED_APPLY="0"
RECONCILE_SUBMITTED_MAX_AGE_SEC="${AGENT_LAB_SUBMITTED_RECONCILE_AGE_SEC:-1800}"
RECONCILE_SUBMITTED_CLOSE_STATUS="${AGENT_LAB_SUBMITTED_RECONCILE_CLOSE_STATUS:-REJECTED}"
RECONCILE_SUBMITTED_REASON="${AGENT_LAB_SUBMITTED_RECONCILE_REASON:-manual_reconcile_submitted}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action) ACTION="${2:-ingest-propose}"; shift 2 ;;
    --market) MARKET="$(echo "${2:-KR}" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --date) RUN_DATE="${2:-$RUN_DATE}"; DATE_EXPLICIT="1"; shift 2 ;;
    --week) WEEK_ID="${2:-}"; shift 2 ;;
    --capital-krw) CAPITAL_KRW="${2:-10000000}"; shift 2 ;;
    --agents) AGENTS="${2:-3}"; shift 2 ;;
    --from) DATE_FROM="${2:-}"; shift 2 ;;
    --to) DATE_TO="${2:-}"; shift 2 ;;
    --chat-once) CHAT_ONCE="1"; shift ;;
    --auto-strategy-once) AUTO_STRATEGY_ONCE="1"; shift ;;
    --chat-poll-timeout) CHAT_POLL_TIMEOUT="${2:-25}"; shift 2 ;;
    --chat-idle-sleep) CHAT_IDLE_SLEEP="${2:-1.0}"; shift 2 ;;
    --chat-memory-limit) CHAT_MEMORY_LIMIT="${2:-20}"; shift 2 ;;
    --auto-strategy-poll) AUTO_STRATEGY_POLL="${2:-300}"; shift 2 ;;
    --auto-strategy-cooldown) AUTO_STRATEGY_COOLDOWN="${2:-180}"; shift 2 ;;
    --auto-strategy-max-updates) AUTO_STRATEGY_MAX_UPDATES="${2:-2}"; shift 2 ;;
    --reconcile-apply) RECONCILE_SUBMITTED_APPLY="1"; shift ;;
    --reconcile-max-age-sec) RECONCILE_SUBMITTED_MAX_AGE_SEC="${2:-1800}"; shift 2 ;;
    --reconcile-close-status) RECONCILE_SUBMITTED_CLOSE_STATUS="${2:-REJECTED}"; shift 2 ;;
    --reconcile-reason) RECONCILE_SUBMITTED_REASON="${2:-manual_reconcile_submitted}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DATE_EXPLICIT" != "1" && "$MARKET" == "US" ]]; then
  # US close-time tasks run on KST early morning; map to previous KST session date.
  HOUR_KST="$(TZ=Asia/Seoul date +%H)"
  if (( 10#$HOUR_KST < 9 )); then
    RUN_DATE="$(TZ=Asia/Seoul date -d '1 day' +%Y%m%d)"
  fi
fi

RUN_STAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/agent_lab/$RUN_DATE"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agent_lab_${ACTION//-/_}_${RUN_STAMP}.log"

run_cli() {
  local args=("$@")
  {
    echo "[run_agent_lab_wsl] started $(TZ=Asia/Seoul date '+%F %T %Z')"
    echo "[run_agent_lab_wsl] python: $PYTHON_BIN"
    echo "[run_agent_lab_wsl] command: $PYTHON_BIN -m systematic_alpha.agent_lab.cli --project-root $ROOT_DIR ${args[*]}"
  } | tee -a "$LOG_FILE"
  "$PYTHON_BIN" -m systematic_alpha.agent_lab.cli --project-root "$ROOT_DIR" "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
}

case "$ACTION" in
  init)
    run_cli init --capital-krw "$CAPITAL_KRW" --agents "$AGENTS"
    ;;
  ingest-propose)
    run_cli ingest-session --market "$MARKET" --date "$RUN_DATE"
    run_cli propose-orders --market "$MARKET" --date "$RUN_DATE"
    ;;
  daily-review)
    run_cli daily-review --date "$RUN_DATE"
    ;;
  weekly-council)
    if [[ -n "$WEEK_ID" ]]; then
      run_cli weekly-council --week "$WEEK_ID"
    else
      run_cli weekly-council --week "$(TZ=Asia/Seoul date +%G-W%V)"
    fi
    ;;
  report)
    if [[ -z "$DATE_FROM" ]]; then DATE_FROM="$RUN_DATE"; fi
    if [[ -z "$DATE_TO" ]]; then DATE_TO="$RUN_DATE"; fi
    run_cli report --from "$DATE_FROM" --to "$DATE_TO"
    ;;
  preopen-plan)
    run_cli preopen-plan --market "$MARKET" --date "$RUN_DATE"
    ;;
  close-report)
    run_cli close-report --market "$MARKET" --date "$RUN_DATE"
    ;;
  sync-account)
    if [[ "${AGENT_LAB_SYNC_STRICT:-1}" == "1" ]]; then
      run_cli sync-account --market "$MARKET" --strict
    else
      run_cli sync-account --market "$MARKET"
    fi
    ;;
  reconcile-submitted)
    if [[ "$RECONCILE_SUBMITTED_APPLY" == "1" ]]; then
      run_cli reconcile-submitted \
        --market "$MARKET" \
        --max-age-sec "$RECONCILE_SUBMITTED_MAX_AGE_SEC" \
        --close-status "$RECONCILE_SUBMITTED_CLOSE_STATUS" \
        --reason "$RECONCILE_SUBMITTED_REASON" \
        --apply
    else
      run_cli reconcile-submitted \
        --market "$MARKET" \
        --max-age-sec "$RECONCILE_SUBMITTED_MAX_AGE_SEC" \
        --close-status "$RECONCILE_SUBMITTED_CLOSE_STATUS" \
        --reason "$RECONCILE_SUBMITTED_REASON"
    fi
    ;;
  shadow-report)
    if [[ -z "$DATE_FROM" ]]; then DATE_FROM="$RUN_DATE"; fi
    if [[ -z "$DATE_TO" ]]; then DATE_TO="$RUN_DATE"; fi
    run_cli shadow-report --from "$DATE_FROM" --to "$DATE_TO"
    ;;
  cutover-reset)
    run_cli cutover-reset --require-flat --archive --reinit --restart-tasks
    ;;
  telegram-chat)
    if [[ "$CHAT_ONCE" == "1" ]]; then
      run_cli telegram-chat --poll-timeout "$CHAT_POLL_TIMEOUT" --idle-sleep "$CHAT_IDLE_SLEEP" --memory-limit "$CHAT_MEMORY_LIMIT" --once
    else
      run_cli telegram-chat --poll-timeout "$CHAT_POLL_TIMEOUT" --idle-sleep "$CHAT_IDLE_SLEEP" --memory-limit "$CHAT_MEMORY_LIMIT"
    fi
    ;;
  auto-strategy-daemon)
    if [[ "$AUTO_STRATEGY_ONCE" == "1" ]]; then
      run_cli auto-strategy-daemon --poll-seconds "$AUTO_STRATEGY_POLL" --cooldown-minutes "$AUTO_STRATEGY_COOLDOWN" --max-updates-per-day "$AUTO_STRATEGY_MAX_UPDATES" --once
    else
      run_cli auto-strategy-daemon --poll-seconds "$AUTO_STRATEGY_POLL" --cooldown-minutes "$AUTO_STRATEGY_COOLDOWN" --max-updates-per-day "$AUTO_STRATEGY_MAX_UPDATES"
    fi
    ;;
  *)
    echo "Unsupported action: $ACTION" >&2
    exit 2
    ;;
esac
