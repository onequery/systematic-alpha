#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

/usr/bin/env bash "$ROOT_DIR/scripts/remove_all_tasks_wsl.sh"
/usr/bin/env bash "$ROOT_DIR/scripts/register_all_tasks_wsl.sh"

echo "WSL tasks reset complete (remove + register)."
