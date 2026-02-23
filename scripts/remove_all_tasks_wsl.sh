#!/usr/bin/env bash
set -euo pipefail

MARK_START="# >>> systematic-alpha tasks start >>>"
MARK_END="# <<< systematic-alpha tasks end <<<"

CURRENT="$(crontab -l 2>/dev/null || true)"
if [[ -z "$CURRENT" ]]; then
  echo "No crontab entries."
  exit 0
fi

printf '%s\n' "$CURRENT" | sed "/$MARK_START/,/$MARK_END/d" | crontab -
echo "WSL cron tasks removed."
echo "Check: crontab -l"
