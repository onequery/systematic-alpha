#!/usr/bin/env bash
set -euo pipefail

MARK_START="# >>> systematic-alpha tasks start >>>"
MARK_END="# <<< systematic-alpha tasks end <<<"

stop_agent_lab_processes() {
  local patterns=(
    "run_agent_lab_wsl.sh --action telegram-chat"
    "run_agent_lab_wsl.sh --action auto-strategy-daemon"
    "systematic_alpha.agent_lab.cli .* telegram-chat"
    "systematic_alpha.agent_lab.cli .* auto-strategy-daemon"
  )
  local killed=0
  for pattern in "${patterns[@]}"; do
    if pids=$(pgrep -f "$pattern" 2>/dev/null); then
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
      killed=1
    fi
  done
  if [[ "$killed" -eq 1 ]]; then
    sleep 0.3
    for pattern in "${patterns[@]}"; do
      if pids=$(pgrep -f "$pattern" 2>/dev/null); then
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
      fi
    done
  fi
}

CURRENT="$(crontab -l 2>/dev/null || true)"
if [[ -z "$CURRENT" ]]; then
  echo "No crontab entries."
  stop_agent_lab_processes
  echo "Check: ps -ef | grep run_agent_lab_wsl.sh | grep -v grep"
  exit 0
fi

printf '%s\n' "$CURRENT" | sed "/$MARK_START/,/$MARK_END/d" | crontab -
stop_agent_lab_processes
echo "WSL cron tasks removed."
echo "Check: crontab -l"
echo "Check: ps -ef | grep run_agent_lab_wsl.sh | grep -v grep"
