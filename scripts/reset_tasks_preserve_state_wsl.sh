#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Preserve existing Agent Lab state (capital/strategy/memory) while refreshing
# cron registrations and daemon processes.
INIT_AGENT_LAB=0 /usr/bin/env bash "$ROOT_DIR/scripts/reset_all_tasks_wsl.sh"

