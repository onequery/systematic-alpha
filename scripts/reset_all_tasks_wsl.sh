#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/reset_all_tasks_wsl.sh [options]

Options:
  --force                 proceed even if active run_daily/prefetch jobs are running
  --no-wait               do not wait for active run_daily/prefetch jobs (warn only)
  --wait-timeout-sec N    max wait time for active jobs (default: 900)
  --wait-poll-sec N       polling interval while waiting (default: 5)
  -h, --help              show this help

Env overrides:
  RESET_FORCE=1
  RESET_WAIT_FOR_ACTIVE_JOBS=0|1
  RESET_ACTIVE_WAIT_TIMEOUT_SEC=900
  RESET_ACTIVE_WAIT_POLL_SEC=5
EOF
}

FORCE_RESET="${RESET_FORCE:-0}"
WAIT_FOR_ACTIVE_JOBS="${RESET_WAIT_FOR_ACTIVE_JOBS:-1}"
ACTIVE_WAIT_TIMEOUT_SEC="${RESET_ACTIVE_WAIT_TIMEOUT_SEC:-900}"
ACTIVE_WAIT_POLL_SEC="${RESET_ACTIVE_WAIT_POLL_SEC:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_RESET=1
      shift
      ;;
    --no-wait)
      WAIT_FOR_ACTIVE_JOBS=0
      shift
      ;;
    --wait-timeout-sec)
      ACTIVE_WAIT_TIMEOUT_SEC="${2:-900}"
      shift 2
      ;;
    --wait-poll-sec)
      ACTIVE_WAIT_POLL_SEC="${2:-5}"
      shift 2
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

if ! [[ "$ACTIVE_WAIT_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  ACTIVE_WAIT_TIMEOUT_SEC=900
fi
if ! [[ "$ACTIVE_WAIT_POLL_SEC" =~ ^[0-9]+$ ]] || (( ACTIVE_WAIT_POLL_SEC < 1 )); then
  ACTIVE_WAIT_POLL_SEC=5
fi

list_active_market_jobs() {
  ps -eo pid=,args= | awk -v self_pid="$$" '
    {
      pid = $1
      $1 = ""
      sub(/^[[:space:]]+/, "", $0)
      cmd = $0
      if (pid == self_pid) next
      if (cmd ~ /reset_all_tasks_wsl\.sh|reset_tasks_preserve_state_wsl\.sh/) next
      if (cmd ~ /run_daily_wsl\.sh|prefetch_kr_universe\.py|prefetch_us_universe\.py|prefetch_us_market_cache\.py|python[^ ]* .*main\.py --market/) {
        print pid " " cmd
      }
    }
  '
}

wait_for_active_market_jobs() {
  local jobs
  jobs="$(list_active_market_jobs)"
  if [[ -z "$jobs" ]]; then
    return 0
  fi

  if [[ "$FORCE_RESET" == "1" || "$WAIT_FOR_ACTIVE_JOBS" != "1" ]]; then
    echo "[reset] warning: active run_daily/prefetch job(s) detected; proceeding by force/no-wait."
    echo "$jobs"
    return 0
  fi

  local started_at now elapsed remain
  started_at="$(date +%s)"
  while [[ -n "$jobs" ]]; do
    now="$(date +%s)"
    elapsed=$((now - started_at))
    if (( elapsed >= ACTIVE_WAIT_TIMEOUT_SEC )); then
      echo "[reset] ERROR: active run_daily/prefetch job(s) still running after ${ACTIVE_WAIT_TIMEOUT_SEC}s."
      echo "$jobs"
      echo "[reset] aborting reset to avoid interrupting market jobs."
      echo "[reset] use --force (or RESET_FORCE=1) to override."
      return 1
    fi

    remain=$((ACTIVE_WAIT_TIMEOUT_SEC - elapsed))
    echo "[reset] active run_daily/prefetch job(s) detected; waiting ${ACTIVE_WAIT_POLL_SEC}s (remaining ${remain}s)"
    echo "$jobs"
    sleep "$ACTIVE_WAIT_POLL_SEC"
    jobs="$(list_active_market_jobs)"
  done
}

ensure_cron_running() {
  if pgrep -x cron >/dev/null 2>&1; then
    echo "[reset] cron daemon: running"
    return 0
  fi

  echo "[reset] cron daemon: not running, attempting to start..."
  local started=0
  local allow_interactive_sudo="${RESET_ALLOW_INTERACTIVE_SUDO:-1}"
  local try_sudo_interactive=0
  if [[ "$allow_interactive_sudo" == "1" && -t 0 && -t 1 ]]; then
    try_sudo_interactive=1
  fi

  if [[ "$started" -eq 0 ]] && command -v service >/dev/null 2>&1; then
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      service cron start >/dev/null 2>&1 || true
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      sudo -n service cron start >/dev/null 2>&1 || true
    elif [[ "$try_sudo_interactive" -eq 1 ]] && command -v sudo >/dev/null 2>&1; then
      echo "[reset] trying: sudo service cron start"
      sudo service cron start >/dev/null 2>&1 || true
    fi
  fi

  if [[ "$started" -eq 0 ]] && [[ -x /etc/init.d/cron ]]; then
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      /etc/init.d/cron start >/dev/null 2>&1 || true
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      sudo -n /etc/init.d/cron start >/dev/null 2>&1 || true
    elif [[ "$try_sudo_interactive" -eq 1 ]] && command -v sudo >/dev/null 2>&1; then
      echo "[reset] trying: sudo /etc/init.d/cron start"
      sudo /etc/init.d/cron start >/dev/null 2>&1 || true
    fi
  fi

  if pgrep -x cron >/dev/null 2>&1; then
    started=1
  fi

  # Last resort: try direct launch (some WSL setups allow this without service manager).
  if [[ "$started" -eq 0 ]] && command -v cron >/dev/null 2>&1; then
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      cron >/dev/null 2>&1 || true
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      sudo -n cron >/dev/null 2>&1 || true
    elif [[ "$try_sudo_interactive" -eq 1 ]] && command -v sudo >/dev/null 2>&1; then
      echo "[reset] trying: sudo cron"
      sudo cron >/dev/null 2>&1 || true
    else
      cron >/dev/null 2>&1 || true
    fi
  fi

  sleep 1
  if pgrep -x cron >/dev/null 2>&1; then
    echo "[reset] cron daemon: running"
    return 0
  fi

  echo "[reset] ERROR: cron daemon is still not running." >&2
  echo "[reset] Try: sudo service cron start" >&2
  return 1
}

wait_for_active_market_jobs
ensure_cron_running
/usr/bin/env bash "$ROOT_DIR/scripts/remove_all_tasks_wsl.sh"
/usr/bin/env bash "$ROOT_DIR/scripts/register_all_tasks_wsl.sh"
ensure_cron_running

echo "WSL tasks reset complete (remove + register + cron check)."
