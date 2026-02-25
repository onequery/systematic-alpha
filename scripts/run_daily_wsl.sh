#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # BOM/CRLF-safe load for .env.
  # shellcheck disable=SC1090
  source <(awk 'NR==1{sub(/^\xef\xbb\xbf/,"")} {sub(/\r$/,"")}1' ".env")
  set +a
fi

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

is_truthy() {
  local raw="${1:-}"
  local normalized
  normalized="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$normalized" in
    1|true|yes|on|y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

telegram_daily_patch_enabled() {
  local token chat_id enabled_raw notify_raw
  token="$(printf '%s' "${TELEGRAM_BOT_TOKEN:-}" | xargs)"
  chat_id="$(printf '%s' "${TELEGRAM_CHAT_ID:-}" | xargs)"
  enabled_raw="${TELEGRAM_ENABLED:-}"
  notify_raw="${AGENT_LAB_NOTIFY_DAILY_PATCH:-1}"
  if ! is_truthy "$notify_raw"; then
    return 1
  fi
  if [[ -n "$enabled_raw" ]] && ! is_truthy "$enabled_raw"; then
    return 1
  fi
  [[ -n "$token" && -n "$chat_id" ]]
}

send_telegram_notice() {
  local text="${1:-}"
  if [[ -z "$text" ]]; then
    return 0
  fi
  if ! telegram_daily_patch_enabled; then
    return 0
  fi

  local token chat_id thread_id
  token="$(printf '%s' "${TELEGRAM_BOT_TOKEN:-}" | xargs)"
  chat_id="$(printf '%s' "${TELEGRAM_CHAT_ID:-}" | xargs)"
  thread_id="$(printf '%s' "${TELEGRAM_THREAD_ID:-}" | xargs)"

  local -a curl_cmd=(
    curl -sS --max-time 10 --retry 1 --retry-delay 1
    -X POST "https://api.telegram.org/bot${token}/sendMessage"
    --data-urlencode "chat_id=${chat_id}"
    --data-urlencode "text=${text}"
  )
  if [[ -n "$thread_id" ]]; then
    curl_cmd+=(--data-urlencode "message_thread_id=${thread_id}")
  fi
  if is_truthy "${TELEGRAM_DISABLE_NOTIFICATION:-0}"; then
    curl_cmd+=(--data-urlencode "disable_notification=true")
  fi

  if is_truthy "${AGENT_LAB_TELEGRAM_USE_ENV_PROXY:-0}"; then
    "${curl_cmd[@]}" >/dev/null 2>&1 || true
  else
    env \
      -u http_proxy -u https_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY \
      -u no_proxy -u NO_PROXY \
      "${curl_cmd[@]}" >/dev/null 2>&1 || true
  fi
}

PYTHON_BIN="$(resolve_python_bin)"
MARKET="KR"
US_EXCHANGE="NASD"
COLLECT_SECONDS=600
FINAL_PICKS=3
PRE_CANDIDATES=40
MAX_SYMBOLS_SCAN=500
KR_UNIVERSE_SIZE=500
US_UNIVERSE_SIZE=500
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --market)
      MARKET="$(echo "${2:-KR}" | tr '[:lower:]' '[:upper:]')"
      shift 2
      ;;
    --exchange)
      US_EXCHANGE="${2:-NASD}"
      shift 2
      ;;
    --collect-seconds)
      COLLECT_SECONDS="${2:-600}"
      shift 2
      ;;
    --final-picks)
      FINAL_PICKS="${2:-3}"
      shift 2
      ;;
    --pre-candidates)
      PRE_CANDIDATES="${2:-40}"
      shift 2
      ;;
    --max-symbols-scan)
      MAX_SYMBOLS_SCAN="${2:-500}"
      shift 2
      ;;
    --kr-universe-size)
      KR_UNIVERSE_SIZE="${2:-500}"
      shift 2
      ;;
    --us-universe-size)
      US_UNIVERSE_SIZE="${2:-500}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MARKET_LC="$(echo "$MARKET" | tr '[:upper:]' '[:lower:]')"
RUN_DATE="$(TZ=Asia/Seoul date +%Y%m%d)"
RUN_STAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"

OUT_BASE="$ROOT_DIR/out/$MARKET_LC/$RUN_DATE"
LOG_DIR="$ROOT_DIR/logs/$MARKET_LC/$RUN_DATE"
RESULTS_DIR="$OUT_BASE/results"

mkdir -p "$OUT_BASE" "$LOG_DIR" "$RESULTS_DIR"

OUTPUT_JSON="$RESULTS_DIR/${MARKET_LC}_daily_${RUN_STAMP}.json"
LOG_FILE="$LOG_DIR/${MARKET_LC}_daily_${RUN_STAMP}.log"
RUN_START_EPOCH="$(date +%s)"

CMD=(
  "$PYTHON_BIN" -u main.py
  --market "$MARKET_LC"
  --collect-seconds "$COLLECT_SECONDS"
  --final-picks "$FINAL_PICKS"
  --pre-candidates "$PRE_CANDIDATES"
  --max-symbols-scan "$MAX_SYMBOLS_SCAN"
  --kr-universe-size "$KR_UNIVERSE_SIZE"
  --us-universe-size "$US_UNIVERSE_SIZE"
  --output-json "$OUTPUT_JSON"
  --analytics-dir "$OUT_BASE/analytics"
  --overnight-report-path "$OUT_BASE/selection_overnight_report.csv"
  --allow-short-bias
  --invalidate-on-low-coverage
)

if [[ "$MARKET" == "US" ]]; then
  CMD+=(--exchange "$US_EXCHANGE")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

notify_daily_patch_start() {
  local now_kst
  now_kst="$(TZ=Asia/Seoul date '+%F %T %Z')"
  local msg
  msg=$'[이벤트] [AgentLab] 일일 패치 시작\n'
  msg+=$'시장='"$MARKET"$'\n'
  msg+=$'일자='"$RUN_DATE"$'\n'
  msg+=$'시각='"$now_kst"$'\n'
  msg+=$'로그='"$LOG_FILE"
  send_telegram_notice "$msg"
}

notify_daily_patch_end() {
  local exit_code="${1:-0}"
  local now_kst elapsed result title
  now_kst="$(TZ=Asia/Seoul date '+%F %T %Z')"
  elapsed="$(( $(date +%s) - RUN_START_EPOCH ))"
  if [[ "$exit_code" -eq 0 ]]; then
    result="성공"
    title="[이벤트] [AgentLab] 일일 패치 종료"
  else
    result="실패"
    title="[Action required] [AgentLab] 일일 패치 실패"
  fi

  local msg
  msg="$title"$'\n'
  msg+=$'시장='"$MARKET"$'\n'
  msg+=$'일자='"$RUN_DATE"$'\n'
  msg+=$'시각='"$now_kst"$'\n'
  msg+=$'결과='"$result"$'\n'
  msg+=$'종료코드='"$exit_code"$'\n'
  msg+=$'경과시간='"${elapsed}"$'s\n'
  msg+=$'결과파일='"$OUTPUT_JSON"$'\n'
  msg+=$'로그='"$LOG_FILE"
  send_telegram_notice "$msg"
}

on_script_exit() {
  local rc=$?
  trap - EXIT
  notify_daily_patch_end "$rc"
}

trap on_script_exit EXIT

{
  echo "[run_daily_wsl] started $(TZ=Asia/Seoul date '+%F %T %Z')"
  echo "[run_daily_wsl] python: $PYTHON_BIN"
  echo "[run_daily_wsl] command: ${CMD[*]}"
} | tee -a "$LOG_FILE"

notify_daily_patch_start

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
