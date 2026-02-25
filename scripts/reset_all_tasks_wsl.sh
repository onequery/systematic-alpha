#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

ensure_cron_running
/usr/bin/env bash "$ROOT_DIR/scripts/remove_all_tasks_wsl.sh"
/usr/bin/env bash "$ROOT_DIR/scripts/register_all_tasks_wsl.sh"
ensure_cron_running

echo "WSL tasks reset complete (remove + register + cron check)."
