#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

INCLUDE_LIVE=1
KEEP_ARTIFACTS=0
EXTRA_PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-live)
      INCLUDE_LIVE=0
      shift
      ;;
    --keep-artifacts)
      KEEP_ARTIFACTS=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_PYTEST_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

PYTHON_BIN="$(resolve_python_bin)"
RUN_STAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/trader_test"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_run_${RUN_STAMP}.log"

echo "[run_all_tests_wsl] started $(TZ=Asia/Seoul date '+%F %T %Z')" | tee -a "$LOG_FILE"
echo "[run_all_tests_wsl] python: $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "[run_all_tests_wsl] include_live: $INCLUDE_LIVE" | tee -a "$LOG_FILE"
echo "[run_all_tests_wsl] keep_artifacts: $KEEP_ARTIFACTS" | tee -a "$LOG_FILE"

if [[ "$KEEP_ARTIFACTS" != "1" ]]; then
  echo "[run_all_tests_wsl] cleaning previous test artifacts..." | tee -a "$LOG_FILE"
  rm -rf "$ROOT_DIR/logs/trader_test" "$ROOT_DIR/out/trader_test" "$ROOT_DIR/state/trader_test" "$ROOT_DIR/.pytest_cache"
  mkdir -p "$ROOT_DIR/logs/trader_test" "$ROOT_DIR/out/trader_test" "$ROOT_DIR/state/trader_test"
  LOG_FILE="$ROOT_DIR/logs/trader_test/test_run_${RUN_STAMP}.log"
fi

if [[ "$INCLUDE_LIVE" == "1" ]]; then
  CMD=("$PYTHON_BIN" -m pytest -q tests)
else
  CMD=("$PYTHON_BIN" -m pytest -q -m "not live_api" tests)
fi
if [[ ${#EXTRA_PYTEST_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_PYTEST_ARGS[@]}")
fi

echo "[run_all_tests_wsl] command: ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[run_all_tests_wsl] finished $(TZ=Asia/Seoul date '+%F %T %Z')" | tee -a "$LOG_FILE"
echo "[run_all_tests_wsl] log: $LOG_FILE"

