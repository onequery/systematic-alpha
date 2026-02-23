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

{
  echo "[run_daily_wsl] started $(TZ=Asia/Seoul date '+%F %T %Z')"
  echo "[run_daily_wsl] python: $PYTHON_BIN"
  echo "[run_daily_wsl] command: ${CMD[*]}"
} | tee -a "$LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
