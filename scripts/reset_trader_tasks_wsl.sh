#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ts() {
  TZ=Asia/Seoul date '+%F %T KST'
}

log() {
  echo "[reset_trader_tasks] $(ts) $*"
}

log "start root=$ROOT_DIR"
log "step=remove begin"
/usr/bin/env bash "$ROOT_DIR/scripts/remove_trader_tasks_wsl.sh"
log "step=remove done"
log "step=register begin"
/usr/bin/env bash "$ROOT_DIR/scripts/register_trader_tasks_wsl.sh"
log "step=register done"
log "complete"
