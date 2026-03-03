#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

load_env_file_safe() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    return 0
  fi
  # shellcheck disable=SC1090
  source <(awk 'NR==1{sub(/^\xef\xbb\xbf/,"")} {sub(/\r$/,"")}1' "$file_path")
}

set -a
load_env_file_safe "$ROOT_DIR/config/trader.config"
load_env_file_safe "$ROOT_DIR/.env"
set +a

resolve_python_bin() {
  local candidate="${PYTHON_BIN:-}"
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

ACTION="status"
MARKET="ALL"
DATE_ARG=""
PHASE="manual"
STRICT=1
POLL_SECONDS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action)
      ACTION="${2:-status}"
      shift 2
      ;;
    --market)
      MARKET="${2:-ALL}"
      shift 2
      ;;
    --date)
      DATE_ARG="${2:-}"
      shift 2
      ;;
    --phase)
      PHASE="${2:-manual}"
      shift 2
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --no-strict)
      STRICT=0
      shift
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:-}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

PYTHON_BIN="$(resolve_python_bin)"
RUN_DATE="$(TZ=Asia/Seoul date +%Y%m%d)"
RUN_STAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/trader/$RUN_DATE"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/trader_${ACTION//-/_}_${RUN_STAMP}.log"

ARGS=()
case "$ACTION" in
  sync-account)
    ARGS+=(sync-account --market "$MARKET")
    if [[ "$STRICT" == "1" ]]; then
      ARGS+=(--strict)
    fi
    ;;
  precompute)
    ARGS+=(precompute --market "$MARKET")
    [[ -n "$DATE_ARG" ]] && ARGS+=(--date "$DATE_ARG")
    ;;
  snapshot-budget)
    ARGS+=(snapshot-budget)
    [[ -n "$DATE_ARG" ]] && ARGS+=(--date "$DATE_ARG")
    ;;
  run-cycle)
    ARGS+=(run-cycle --market "$MARKET")
    [[ -n "$DATE_ARG" ]] && ARGS+=(--date "$DATE_ARG")
    ;;
  liquidate)
    ARGS+=(liquidate --market "$MARKET" --phase "$PHASE")
    [[ -n "$DATE_ARG" ]] && ARGS+=(--date "$DATE_ARG")
    ;;
  report)
    ARGS+=(report)
    [[ -n "$DATE_ARG" ]] && ARGS+=(--date "$DATE_ARG")
    ;;
  daemon)
    ARGS+=(daemon)
    [[ -n "$POLL_SECONDS" ]] && ARGS+=(--poll-seconds "$POLL_SECONDS")
    ;;
  archive-reset|status)
    ARGS+=("$ACTION")
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

{
  echo "[run_trader_wsl] started $(TZ=Asia/Seoul date '+%F %T %Z')"
  echo "[run_trader_wsl] python: $PYTHON_BIN"
  echo "[run_trader_wsl] command: $PYTHON_BIN -m systematic_alpha.trader.cli --project-root $ROOT_DIR ${ARGS[*]}"
} | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m systematic_alpha.trader.cli --project-root "$ROOT_DIR" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

